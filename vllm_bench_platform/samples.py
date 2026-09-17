"""Benchmark prompt samples: preparation, loading, and request alignment.

Why alignment needs care
------------------------
``vllm bench serve --save-detailed`` writes per-request arrays that follow the
order in which requests were *issued*, not the order of lines in the dataset
file. For ``--dataset-name custom`` vLLM 0.11.0 does (see
``vllm/benchmarks/datasets.py``)::

    class CustomDataset(BenchmarkDataset):
        def load_data(self):
            ...                       # read JSONL lines in file order
            random.seed(self.random_seed)
            random.shuffle(self.data)
        def sample(self, num_requests, ...):
            for i, item in enumerate(self.data):     # first num_requests items
                ...
            self.maybe_oversample_requests(...)      # random.choices() if short

and ``get_samples()`` builds it as ``CustomDataset(dataset_path=...)`` — note
that **no** ``random_seed`` is passed, so the shuffle always uses
``BenchmarkDataset.DEFAULT_SEED == 0`` **regardless of ``--seed``**. Verified
empirically: ``--seed 0`` and ``--seed 42`` both produce the order
``[4, 1, 5, 2, 0, 3, 7, 6]`` for an 8-line file. :func:`align_requests_to_samples`
therefore takes no seed argument at all — passing ``benchmark.seed`` would
silently mislabel every request.

``random.shuffle`` / ``random.choices`` on a list depend only on the list
*length*, so the permutation is fully reproducible here from (n, seed) alone:
:func:`custom_dataset_order` returns exactly the file-line index that produced
request *i*. :func:`align_requests_to_samples` uses it to attach the original
prompt and a stable ``sample_id`` to every parsed request.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

# BenchmarkDataset.DEFAULT_SEED in vllm/benchmarks/datasets.py
VLLM_DEFAULT_DATASET_SEED = 0

SONNET_URL = "https://raw.githubusercontent.com/vllm-project/vllm/main/benchmarks/sonnet.txt"
SHAREGPT_URL = (
    "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/"
    "resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
)


@dataclass
class Sample:
    """One prompt from the benchmark dataset file."""

    line_index: int
    sample_id: str
    prompt: str

    def as_dataset_item(self, source: Optional[str] = None) -> dict:
        # NB: "source" is a reserved DatasetItem field on the platform (MANUAL/TRACE/
        # SPAN/SDK), so the provenance label goes into "dataset_source".
        item = {
            "sample_id": self.sample_id,
            "line_index": self.line_index,
            "prompt": self.prompt,
        }
        if source:
            item["dataset_source"] = source
        return item


def make_sample_id(line_index: int, prompt: str) -> str:
    """A stable id that changes when the prompt text changes.

    Positional-only ids (``sample-0004``) collide across regenerated dataset
    files: re-running `prepare-dataset` with different content but the same
    platform dataset name would produce two items sharing an id, and the
    ``sample_id -> dataset_item_id`` mapping would pick an arbitrary one.
    Mixing in a digest of the prompt makes that impossible, while keeping the
    line index for readability and for disambiguating duplicate prompts.
    """
    digest = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
    return f"s{line_index:04d}-{digest}"


# ---------------------------------------------------------------------------
# loading / writing
# ---------------------------------------------------------------------------


def load_samples(path: str | Path) -> List[Sample]:
    """Read a vLLM ``custom`` dataset JSONL file (one ``{"prompt": ...}`` per line)."""
    p = Path(path)
    samples: List[Sample] = []
    with p.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                raise ValueError(f"{p}: every JSONL line must contain a 'prompt' key")
            idx = len(samples)
            prompt = str(obj["prompt"])
            samples.append(
                Sample(
                    line_index=idx,
                    sample_id=str(obj.get("sample_id") or make_sample_id(idx, prompt)),
                    prompt=prompt,
                )
            )
    if not samples:
        raise ValueError(f"{p}: no samples found")
    return samples


def write_samples(path: str | Path, prompts: Sequence[str]) -> Path:
    """Write prompts as a vLLM ``custom`` JSONL dataset."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            fh.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
    return p


# ---------------------------------------------------------------------------
# replicating vLLM's CustomDataset sampling order
# ---------------------------------------------------------------------------


def custom_dataset_order(
    num_samples: int,
    num_requests: int,
    seed: int = VLLM_DEFAULT_DATASET_SEED,
) -> List[int]:
    """Return ``order[i]`` = file line index of the *i*-th issued request.

    Replicates ``CustomDataset.load_data`` + ``sample`` + ``maybe_oversample_requests``.
    """
    if num_samples <= 0:
        return []
    if num_requests <= 0:
        num_requests = num_samples

    rng = random.Random(seed)
    order = list(range(num_samples))
    rng.shuffle(order)
    sampled = order[:num_requests]

    if len(sampled) < num_requests:
        # maybe_oversample_requests: random.seed(seed); random.choices(...)
        rng2 = random.Random(seed)
        sampled = sampled + rng2.choices(sampled, k=num_requests - len(sampled))
    return sampled


def align_by_prompt(
    requests: Iterable,
    samples: Sequence[Sample],
) -> tuple[int, List[Sample]]:
    """Attach ``sample_id`` by matching the captured prompt text, in place.

    This is the preferred path: when the capture sidecar exists we know the
    exact prompt each request was sent, so alignment needs no replay of vLLM's
    shuffling and works for *any* dataset type (sharegpt / random / hf ...),
    not just ``custom``.

    Prompts not present in ``samples`` are returned as new :class:`Sample`
    objects (deduplicated by content) so the caller can upload them too.

    Returns ``(matched_count, extra_samples)``.
    """
    requests = list(requests)
    by_prompt: dict[str, Sample] = {}
    for sample in samples:
        by_prompt.setdefault(sample.prompt, sample)

    extra: List[Sample] = []
    matched = 0
    next_index = len(samples)
    for req in requests:
        prompt = getattr(req, "prompt", None)
        if prompt is None:
            continue
        sample = by_prompt.get(prompt)
        if sample is None:
            sample = Sample(
                line_index=next_index,
                sample_id=make_sample_id(next_index, prompt),
                prompt=prompt,
            )
            by_prompt[prompt] = sample
            extra.append(sample)
            next_index += 1
        req.sample_id = sample.sample_id
        matched += 1
    return matched, extra


def align_requests_to_samples(
    requests: Iterable,
    samples: Sequence[Sample],
    dataset_name: str = "custom",
) -> bool:
    """Attach ``prompt`` / ``sample_id`` to each parsed request, in place.

    Deliberately takes **no** seed: vLLM's ``CustomDataset`` always shuffles
    with ``DEFAULT_SEED = 0`` (``--seed`` is not threaded into it), so using
    ``benchmark.seed`` here would compute the wrong permutation while still
    reporting success.

    Returns True when every request could be mapped back to a dataset sample.
    """
    requests = list(requests)
    if dataset_name != "custom" or not samples or not requests:
        return False

    order = custom_dataset_order(
        len(samples), len(requests), seed=VLLM_DEFAULT_DATASET_SEED
    )
    if len(order) != len(requests):
        return False

    for req, line_index in zip(requests, order):
        sample = samples[line_index]
        req.prompt = sample.prompt
        req.sample_id = sample.sample_id
    return True


# ---------------------------------------------------------------------------
# preparing a sample set from an official vLLM benchmark dataset
# ---------------------------------------------------------------------------


def _fetch_text(url: str, timeout: float = 60.0) -> str:
    import httpx

    resp = httpx.get(url, timeout=timeout, follow_redirects=True)
    resp.raise_for_status()
    return resp.text


def build_sonnet_prompts(
    text: str,
    num_samples: int,
    lines_per_prompt: int = 8,
    seed: int = 1234,
) -> List[str]:
    """Slice vLLM's official ``benchmarks/sonnet.txt`` into prompts."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("sonnet source is empty")
    chunks = [
        "\n".join(lines[i : i + lines_per_prompt])
        for i in range(0, len(lines), lines_per_prompt)
    ]
    chunks = [c for c in chunks if c]
    rng = random.Random(seed)
    if num_samples < len(chunks):
        picked = rng.sample(range(len(chunks)), num_samples)
        picked.sort()
        chunks = [chunks[i] for i in picked]
    return chunks[:num_samples] if num_samples else chunks


def build_sharegpt_prompts(
    payload: str,
    num_samples: int,
    seed: int = 1234,
) -> List[str]:
    """Take the first human turn of N ShareGPT conversations."""
    data = json.loads(payload)
    prompts: List[str] = []
    for conv in data:
        turns = conv.get("conversations") or []
        if not turns:
            continue
        first = turns[0]
        value = (first.get("value") or "").strip()
        if value:
            prompts.append(value)
    rng = random.Random(seed)
    if num_samples < len(prompts):
        picked = sorted(rng.sample(range(len(prompts)), num_samples))
        prompts = [prompts[i] for i in picked]
    return prompts[:num_samples] if num_samples else prompts


def prepare_dataset(
    source: str,
    output_path: str | Path,
    num_samples: int,
    seed: int = 1234,
    lines_per_prompt: int = 8,
    source_url: Optional[str] = None,
) -> List[Sample]:
    """Download an official vLLM benchmark dataset and write a small JSONL slice."""
    if source == "sonnet":
        text = _fetch_text(source_url or SONNET_URL)
        prompts = build_sonnet_prompts(text, num_samples, lines_per_prompt, seed)
    elif source == "sharegpt":
        payload = _fetch_text(source_url or SHAREGPT_URL, timeout=600.0)
        prompts = build_sharegpt_prompts(payload, num_samples, seed)
    elif source == "local":
        return load_samples(output_path)
    else:
        raise ValueError(f"unknown prepare.source: {source!r} (use sonnet|sharegpt|local)")

    write_samples(output_path, prompts)
    return load_samples(output_path)
