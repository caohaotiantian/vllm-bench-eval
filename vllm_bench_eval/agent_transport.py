"""Measure chat requests and assemble streamed assistant messages."""
from __future__ import annotations

import datetime as dt
import json
import time
from typing import Any

import httpx

from .agent_metrics import normalize_timing


def chat_request(client: httpx.Client, url: str, payload: dict, timeout: float, cancel=None) -> tuple[dict, dict]:
    """Return a chat response and measured client timing."""
    started = time.monotonic()
    first = last = None
    first_wall = None
    intervals = []
    usage = {}
    try:
        if not payload.get("stream"):
            response = client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
        else:
            message: dict[str, Any] = {"role": "assistant", "content": ""}
            calls: dict[int, dict] = {}
            usage = {}
            finish = None
            done = False
            data_lines: list[str] = []
            with client.stream("POST", url, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                def consume(data: str) -> None:
                    nonlocal first, last, first_wall, usage, finish, done
                    if data == "[DONE]":
                        done = True
                        return
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise ValueError(f"stream error: {chunk['error']}")
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    meaningful = False
                    for choice in chunk.get("choices", []):
                        if choice.get("index", 0) != 0:
                            continue
                        if choice.get("finish_reason") is not None:
                            finish = choice["finish_reason"]
                        delta = choice.get("delta") or {}
                        if "role" in delta:
                            message["role"] = delta["role"]
                        if delta.get("reasoning_content") or delta.get("reasoning"):
                            message["_reasoning_present"] = True
                        if delta.get("function_call"):
                            message["function_call"] = True
                        content = delta.get("content")
                        if content:
                            if not isinstance(content, str):
                                raise ValueError("stream content must be a string")
                            message["content"] += content
                            meaningful = True
                        for part in delta.get("tool_calls") or []:
                            index = part.get("index")
                            if not isinstance(index, int) or index < 0:
                                raise ValueError("stream tool call requires a nonnegative index")
                            call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                            if part.get("id"):
                                call["id"] += part["id"]
                            function = part.get("function") or {}
                            for key in ("name", "arguments"):
                                fragment = function.get(key)
                                if fragment:
                                    call["function"][key] += fragment
                            meaningful = meaningful or bool(function.get("name") or function.get("arguments"))
                    if meaningful:
                        observed = time.monotonic()
                        if last is not None:
                            intervals.append((observed - last) * 1000)
                        last = observed
                        if first is None:
                            first = last
                            first_wall = dt.datetime.now(dt.timezone.utc).isoformat()
                for line in response.iter_lines():
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("evaluation cancelled")
                    if time.monotonic() - started > timeout:
                        raise TimeoutError("request timeout exceeded")
                    if line == "":
                        if data_lines:
                            consume("\n".join(data_lines))
                            data_lines.clear()
                        if done:
                            break
                    elif line.startswith("data:"):
                        data_lines.append(line[5:].lstrip())
                if data_lines and not done:
                    consume("\n".join(data_lines))
                if not done or finish is None:
                    raise ValueError("stream ended without DONE and a finish reason")
            if calls:
                message["tool_calls"] = [calls[index] for index in sorted(calls)]
                if not message["content"]:
                    message["content"] = None
            body = {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}
        ended = time.monotonic()
        completion_tokens = (body.get("usage") or {}).get("completion_tokens")
        timing = {"stream": bool(payload.get("stream")), "e2e_ms": (ended - started) * 1000,
                  "ttft_ms": (first - started) * 1000 if first is not None else None,
                  "tpot_ms": (last - first) * 1000 / (completion_tokens - 1)
                  if first is not None and isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool) and completion_tokens > 1 else None,
                  "chunk_itl_ms": intervals, "itl_kind": "client_chunk_estimate",
                  "first_token_time": first_wall, "completion_tokens": completion_tokens,
                  "decode_ms": (last - first) * 1000 if first is not None else None,
                  "ttft_null_reason": "nonstream" if not payload.get("stream") else "no_output" if first is None else None,
                  "tpot_null_reason": "nonstream" if not payload.get("stream") else "no_output" if first is None else
                  "missing_completion_tokens" if not isinstance(completion_tokens, int) or isinstance(completion_tokens, bool) else
                  "fewer_than_two_tokens" if completion_tokens < 2 else None}
        return body, normalize_timing(timing, body.get("usage"))
    except Exception as exc:
        exc.agent_timing = {"stream": bool(payload.get("stream")), "e2e_ms": (time.monotonic() - started) * 1000,
                            "ttft_ms": (first - started) * 1000 if first is not None else None,
                            "tpot_ms": None, "chunk_itl_ms": intervals, "first_token_time": first_wall,
                            "tpot_null_reason": "incomplete_request"}
        exc.agent_timing = normalize_timing(exc.agent_timing, usage)
        exc.agent_usage = usage
        raise
