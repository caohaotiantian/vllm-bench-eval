"""Measure chat requests and assemble streamed assistant messages."""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import time

import httpx

from .agent_metrics import normalize_timing


class _ResponseState:
    def __init__(self, stream):
        self.started = time.monotonic()
        self.stream = stream
        self.first = self.last = self.first_wall = None
        self.intervals = []
        self.usage = {}
        self.message = {"role": "assistant", "content": ""}
        self.calls = {}
        self.finish = None
        self.done = False
        self.data_lines = []

    def consume(self, data: str) -> None:
        if data == "[DONE]":
            self.done = True
            return
        chunk = json.loads(data)
        if chunk.get("error"):
            raise ValueError(f"stream error: {chunk['error']}")
        if chunk.get("usage"):
            self.usage = chunk["usage"]
        meaningful = False
        for choice in chunk.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            if choice.get("finish_reason") is not None:
                self.finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if "role" in delta:
                self.message["role"] = delta["role"]
            if delta.get("reasoning_content") or delta.get("reasoning"):
                self.message["_reasoning_present"] = True
            if delta.get("function_call"):
                self.message["function_call"] = True
            content = delta.get("content")
            if content:
                if not isinstance(content, str):
                    raise ValueError("stream content must be a string")
                self.message["content"] += content
                meaningful = True
            for part in delta.get("tool_calls") or []:
                index = part.get("index")
                if not isinstance(index, int) or index < 0:
                    raise ValueError("stream tool call requires a nonnegative index")
                call = self.calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
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
            if self.last is not None:
                self.intervals.append((observed - self.last) * 1000)
            self.last = observed
            if self.first is None:
                self.first = self.last
                self.first_wall = dt.datetime.now(dt.timezone.utc).isoformat()

    def line(self, line):
        if line == "":
            if self.data_lines:
                self.consume("\n".join(self.data_lines))
                self.data_lines.clear()
        elif line.startswith("data:"):
            self.data_lines.append(line[5:].lstrip())

    def body(self):
        if self.data_lines and not self.done:
            self.consume("\n".join(self.data_lines))
        if not self.done or self.finish is None:
            raise ValueError("stream ended without DONE and a finish reason")
        if self.calls:
            self.message["tool_calls"] = [self.calls[index] for index in sorted(self.calls)]
            if not self.message["content"]:
                self.message["content"] = None
        return {"choices": [{"message": self.message, "finish_reason": self.finish}], "usage": self.usage}

    def timing(self, usage, *, incomplete=False):
        ended = time.monotonic()
        tokens = (usage or {}).get("completion_tokens")
        timing = {"stream": self.stream, "e2e_ms": (ended - self.started) * 1000,
                  "ttft_ms": (self.first - self.started) * 1000 if self.first is not None else None,
                  "decode_ms": (self.last - self.first) * 1000 if self.first is not None else None,
                  "tpot_ms": None, "chunk_itl_ms": self.intervals, "itl_kind": "client_chunk_estimate",
                  "first_token_time": self.first_wall, "completion_tokens": tokens,
                  "ttft_null_reason": "nonstream" if not self.stream else "no_output" if self.first is None else None,
                  "tpot_null_reason": "incomplete_request" if incomplete else None}
        return normalize_timing(timing, usage)

    def failure(self, exc):
        exc.agent_usage = self.usage
        exc.agent_timing = self.timing(self.usage, incomplete=True)


def chat_request(client: httpx.Client, url: str, payload: dict, timeout: float, cancel=None) -> tuple[dict, dict]:
    """Return a chat response and measured client timing."""
    state = _ResponseState(bool(payload.get("stream")))
    try:
        if not state.stream:
            response = client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            body = response.json()
        else:
            with client.stream("POST", url, json=payload, timeout=timeout) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("evaluation cancelled")
                    if time.monotonic() - state.started > timeout:
                        raise TimeoutError("request timeout exceeded")
                    state.line(line)
                    if state.done:
                        break
            body = state.body()
        return body, state.timing(body.get("usage"))
    except Exception as exc:
        state.failure(exc)
        raise


async def async_chat_request(client: httpx.AsyncClient, url: str, payload: dict, timeout: float) -> tuple[dict, dict]:
    """Return a cancellable response with the same parser and timing rules."""
    state = _ResponseState(bool(payload.get("stream")))
    async def request():
        if not state.stream:
            response = await client.post(url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        async with client.stream("POST", url, json=payload, timeout=timeout) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                state.line(line)
                if state.done:
                    break
        return state.body()
    try:
        body = await asyncio.wait_for(request(), timeout)
        return body, state.timing(body.get("usage"))
    except (Exception, asyncio.CancelledError) as exc:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) and not str(exc):
            exc.args = ("request timeout exceeded",)
        state.failure(exc)
        raise
