"""Convert the HumanEval corpus into isolated three-tool coding tasks."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
import random

import httpx

from .config import AppConfig
from .coding_agent import AgentError

HUMANEVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/463c980b59e818ace59f6f9803cd92c749ceae61/data/HumanEval.jsonl.gz"
HUMANEVAL_SHA256 = "b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef"
HUMANEVAL_REVISION = "463c980b59e818ace59f6f9803cd92c749ceae61"
LICENSE_SOURCE = 'https://raw.githubusercontent.com/openai/human-eval/d321ec0b6c23dec317337be99f6d0c45ca73f3d5/LICENSE'
LICENSE_SHA256 = "bcba3de214851cce46ed5af42d6698044616eeace887c3231bc7a20474ab639e"
MIT_LICENSE = 'The MIT License\n\nCopyright (c) OpenAI (https://openai.com)\n\nPermission is hereby granted, free of charge, to any person obtaining a copy\nof this software and associated documentation files (the "Software"), to deal\nin the Software without restriction, including without limitation the rights\nto use, copy, modify, merge, publish, distribute, sublicense, and/or sell\ncopies of the Software, and to permit persons to whom the Software is\nfurnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in\nall copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR\nIMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,\nFITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE\nAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER\nLIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,\nOUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN\nTHE SOFTWARE.\n'


def prepare_agent_dataset(cfg: AppConfig, *, source_file: Path | None = None, output_dir: Path | None = None,
                          num_samples: int | None = None, seed: int | None = None) -> Path:
    options = cfg.prepare_agent
    if options.source != "humaneval":
        raise AgentError("prepare_agent.source must be humaneval")
    count = num_samples if num_samples is not None else options.num_samples
    if count < 1:
        raise AgentError("num_samples must be positive")
    selected_seed = seed if seed is not None else options.seed
    if source_file:
        raw = source_file.read_bytes()
        source = str(source_file.resolve())
    else:
        response = httpx.get(options.source_url, follow_redirects=True, timeout=30)
        response.raise_for_status()
        raw = response.content
        source = options.source_url
    digest = hashlib.sha256(raw).hexdigest()
    if not source_file and source == HUMANEVAL_URL and digest != HUMANEVAL_SHA256:
        raise AgentError("official HumanEval source SHA256 differs from the pinned snapshot")
    content = gzip.decompress(raw) if raw.startswith(b"\x1f\x8b") else raw
    if len(content) > 10_000_000:
        raise AgentError("HumanEval source exceeds the supported size")
    rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip()]
    ids = set()
    for row in rows:
        for key in ("task_id", "prompt", "entry_point", "test"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise AgentError(f"HumanEval row requires {key}")
        if row["task_id"] in ids or not row["entry_point"].isidentifier():
            raise AgentError("HumanEval source contains duplicate IDs or invalid entry points")
        ids.add(row["task_id"])
        ast.parse(row["prompt"] + "    raise NotImplementedError\n")
        ast.parse(row["test"])
    if count > len(rows):
        raise AgentError("num_samples exceeds the HumanEval source size")
    rows.sort(key=lambda row: (int(row["task_id"].rsplit("/", 1)[-1]) if row["task_id"].rsplit("/", 1)[-1].isdigit() else 10**9, row["task_id"]))
    chosen = random.Random(selected_seed).sample(rows, count) if selected_seed is not None else rows[:count]
    out = (output_dir or cfg.path(options.output_dir)).resolve()
    if out.exists():
        raise AgentError("output directory already exists; select a new output directory")
    out.mkdir(parents=True)
    provenance = {"dataset": "openai/openai_humaneval", "source": source, "source_sha256": digest,
                  "source_revision": HUMANEVAL_REVISION if source == HUMANEVAL_URL else None,
                  "license_source": LICENSE_SOURCE, "license_sha256": LICENSE_SHA256,
                  "num_samples": count, "seed": selected_seed, "selection": "seeded" if selected_seed is not None else "numeric_first"}
    tasks = []
    for index, row in enumerate(chosen):
        repository = out / "repositories" / str(index)
        repository.mkdir(parents=True)
        (repository / "solution.py").write_text(row["prompt"] + "    raise NotImplementedError\n", encoding="utf-8")
        verifier = ("from solution import " + row["entry_point"] + "\n" + row["test"] + "\ncheck(" + row["entry_point"] + ")\nprint('HumanEval checks passed')\n")
        (repository / "verify.py").write_text(verifier, encoding="utf-8")
        (repository / "LICENSE.humaneval").write_text(MIT_LICENSE, encoding="utf-8")
        tasks.append({"task_id": row["task_id"], "prompt": "Use read_file to read solution.py and verify.py. Implement the requested function with write_file. "
                      "Use run_command with argv [\"python3\", \"verify.py\"] to test the implementation. "
                      "Repair failures before your final answer. Do not modify verify.py.",
                      "repository": f"repositories/{index}", "verify_command": ["python3", "verify.py"],
                      "verification_files": ["verify.py"], "required_tools": ["read_file", "write_file", "run_command"],
                      "provenance": {**provenance, "source_task_id": row["task_id"],
                                     "sample_sha256": hashlib.sha256(json.dumps({key: row[key] for key in ("task_id", "prompt", "entry_point", "test")}, sort_keys=True).encode()).hexdigest()}})
    (out / "LICENSE.humaneval").write_text(MIT_LICENSE, encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    path = out / "tasks.jsonl"
    path.write_text("".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), encoding="utf-8")
    return path
