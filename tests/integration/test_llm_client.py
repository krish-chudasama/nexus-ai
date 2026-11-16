"""Exercises the real OpenAI SDK + our client against a local server that speaks the
OpenAI chat-completions wire protocol. This verifies request/response handling, tool-call
parsing, streaming and error mapping. It does NOT verify behaviour of any real provider."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.core.config import Settings
from app.core.errors import ConfigurationError, LLMError
from app.llm.factory import get_llm_client

SEEN: list[dict] = []


def _completion(message: dict, finish: str = "stop") -> dict:
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "stub-1",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
    }  # fmt: skip


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append({"auth": self.headers.get("Authorization"), "body": body})
        last = body["messages"][-1]["content"]
        if last == "boom":
            return self._send(500, {"error": {"message": "upstream exploded", "type": "server"}})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for tok in ("Hel", "lo", "!"):
                delta = {"index": 0, "delta": {"content": tok}}
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": 0,
                         "model": "stub-1", "choices": [delta]}  # fmt: skip
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        if body.get("tools") and last == "use tool":
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_9", "type": "function",
                 "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'}},
                {"id": "call_10", "type": "function",
                 "function": {"name": "calculator", "arguments": "{not json"}},
            ]}  # fmt: skip
            return self._send(200, _completion(msg, "tool_calls"))
        self._send(200, _completion({"role": "assistant", "content": "plain answer"}))

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def base_url():
    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()


@pytest.fixture
def client(base_url, tmp_path):
    SEEN.clear()
    s = Settings(_env_file=None, data_dir=tmp_path, llm_base_url=base_url,
                 llm_api_key="test-key-123", llm_model="stub-1")  # fmt: skip
    return get_llm_client(s)


def test_chat_returns_content_usage_and_sends_auth(client):
    r = client.chat([{"role": "user", "content": "hi"}], max_tokens=50)
    assert r.content == "plain answer" and r.model == "stub-1"
    assert (r.usage.prompt_tokens, r.usage.total_tokens) == (11, 18)
    assert SEEN[0]["auth"] == "Bearer test-key-123"
    assert SEEN[0]["body"]["model"] == "stub-1" and SEEN[0]["body"]["max_tokens"] == 50


def test_tool_calls_parsed_and_invalid_json_flagged(client):
    schema = [
        {"type": "function", "function": {"name": "calculator", "parameters": {"type": "object"}}}
    ]
    r = client.chat([{"role": "user", "content": "use tool"}], tools=schema)
    assert SEEN[0]["body"]["tools"] == schema and SEEN[0]["body"]["tool_choice"] == "auto"
    assert r.tool_calls[0].name == "calculator" and r.tool_calls[0].arguments == {
        "expression": "2+2"
    }
    assert "__invalid_json__" in r.tool_calls[1].arguments  # registry will reject, not crash


def test_streaming_yields_tokens(client):
    assert "".join(client.stream([{"role": "user", "content": "hi"}])) == "Hello!"


def test_provider_errors_are_normalised_without_leaking_key(client):
    with pytest.raises(LLMError) as exc:
        client.chat([{"role": "user", "content": "boom"}])
    assert "test-key-123" not in str(exc.value)


def test_unreachable_provider_is_llm_error(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path, llm_base_url="http://127.0.0.1:9/v1",
                 llm_api_key="k", llm_timeout_s=2)  # fmt: skip
    with pytest.raises(LLMError):
        get_llm_client(s).chat([{"role": "user", "content": "hi"}])


def test_missing_key_is_configuration_error(tmp_path):
    with pytest.raises(ConfigurationError, match="LLM_API_KEY"):
        get_llm_client(Settings(_env_file=None, data_dir=tmp_path, llm_api_key=""))


def test_full_agent_loop_over_the_wire(client, indexed):
    """Agent -> real SDK -> stub server -> tool execution -> final answer."""
    from app.agents.agent import Agent
    from app.schemas.models import AgentRequest

    res = Agent(indexed.settings, indexed, lambda: client).run(AgentRequest(question="use tool"))
    kinds = [(s.kind, s.tool, s.ok) for s in res.steps]
    assert kinds[0] == ("tool_call", "calculator", True)
    assert kinds[1] == ("tool_call", "calculator", False)  # malformed JSON args rejected
    assert res.usage and res.usage.total_tokens >= 18
