import http.client
import io
import json
import os
import socket
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from local_llm import MODE_INSTRUCT, MODE_RIVA, LocalChatClient, LocalLLMClient, LocalLLMError


class FakeResponse:
	def __init__(self, body):
		self.body = body
	def __enter__(self):
		return self
	def __exit__(self, *_):
		return False
	def read(self):
		if isinstance(self.body, BaseException):
			raise self.body
		return self.body


class FakeOpener:
	def __init__(self, result):
		self.result = result
		self.calls = []
	def open(self, request, timeout):
		self.calls.append((request, timeout))
		if isinstance(self.result, BaseException):
			raise self.result
		return FakeResponse(self.result)


def response(content="翻譯結果", **extra):
	body = {"content": content, "stop_type": "eos"}
	body.update(extra)
	return json.dumps(body, ensure_ascii=False).encode("utf-8")


def chat_response(content="翻譯結果", finish_reason="stop"):
	body = {"choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}]}
	return json.dumps(body, ensure_ascii=False).encode("utf-8")


class LocalLLMClientContractTests(unittest.TestCase):
	def setUp(self):
		self.environment = mock.patch.dict(os.environ, {}, clear=True)
		self.environment.start()
		self.addCleanup(self.environment.stop)

	def make_client(self, result=response(), **kwargs):
		opener = FakeOpener(result)
		with mock.patch("local_llm.urllib.request.build_opener", return_value=opener):
			client = LocalLLMClient(**kwargs)
		return client, opener

	def test_riva_chat_uses_exact_prompt_and_sampling_contract(self):
		client, opener = self.make_client(timeout=12.5, max_tokens=200, mode="riva")
		self.assertEqual(client.chat("只翻譯目前句子", "前一句：田中です。\n目前：よろしく。"), "翻譯結果")
		request, timeout = opener.calls[0]
		payload = json.loads(request.data.decode("utf-8"))
		self.assertEqual(request.full_url, "http://127.0.0.1:8766/completion")
		self.assertEqual(payload["prompt"], "<s>System\n只翻譯目前句子</s>\n<s>User\n前一句：田中です。\n目前：よろしく。</s>\n<s>Assistant\n")
		self.assertEqual({key: payload[key] for key in ("temperature", "top_p", "top_k", "repeat_penalty", "seed", "n_predict", "stop", "stream")}, {"temperature": 0, "top_p": 1, "top_k": 0, "repeat_penalty": 1, "seed": 1234, "n_predict": 200, "stop": ["</s>", "<s>"], "stream": False})
		self.assertEqual(timeout, 12.5)

	def test_plain_ontime_prompt_shape_is_reproduced(self):
		client, opener = self.make_client(mode="riva")
		client.chat("", "Translate this into Traditional Chinese: こんにちは。")
		payload = json.loads(opener.calls[0][0].data.decode("utf-8"))
		self.assertEqual(payload["prompt"], "<s>System\n</s>\n<s>User\nTranslate this into Traditional Chinese: こんにちは。</s>\n<s>Assistant\n")

	def test_instruct_chat_sends_gemini_style_messages(self):
		client, opener = self.make_client(chat_response("  請多指教。 "), mode="instruct", timeout=30)
		self.assertEqual(client.chat("系統提示", "請翻譯這一句：\nよろしく。", max_tokens=64, timeout=5), "請多指教。")
		request, timeout = opener.calls[0]
		payload = json.loads(request.data.decode("utf-8"))
		self.assertEqual(request.full_url, "http://127.0.0.1:8766/v1/chat/completions")
		self.assertEqual(payload["messages"], [
			{"role": "system", "content": "系統提示"},
			{"role": "user", "content": "請翻譯這一句：\nよろしく。"},
		])
		self.assertEqual(
			{key: payload[key] for key in ("stream", "temperature", "max_tokens", "seed", "chat_template_kwargs")},
			{"stream": False, "temperature": 0.3, "max_tokens": 64, "seed": 1234, "chat_template_kwargs": {"enable_thinking": False}},
		)
		self.assertNotIn("model", payload)
		self.assertEqual(timeout, 5)

	def test_instruct_chat_names_configured_model_and_omits_empty_system(self):
		client, opener = self.make_client(chat_response(), mode="instruct", model=" qwen2.5:14b ")
		client.chat("", "hello")
		payload = json.loads(opener.calls[0][0].data.decode("utf-8"))
		self.assertEqual(payload["model"], "qwen2.5:14b")
		self.assertEqual(payload["messages"], [{"role": "user", "content": "hello"}])

	def test_instruct_chat_rejects_length_stop_and_strips_reasoning(self):
		client, _ = self.make_client(chat_response("部分", finish_reason="length"), mode="instruct")
		with self.assertRaises(LocalLLMError) as caught:
			client.chat("s", "u")
		self.assertEqual(caught.exception.kind, "truncated")
		self.assertTrue(caught.exception.too_large)
		client, _ = self.make_client(chat_response("<think>\n推理\n</think>\n\n譯文。"), mode="instruct")
		self.assertEqual(client.chat("s", "u"), "譯文。")
		for body in (b"{}", b'{"choices": []}', json.dumps({"choices": [{"message": {"content": None}}]}).encode()):
			with self.subTest(body=body):
				client, _ = self.make_client(body, mode="instruct")
				with self.assertRaisesRegex(LocalLLMError, "格式"):
					client.chat("s", "u")

	def test_complete_sends_raw_prompt(self):
		client, opener = self.make_client()
		client.complete("raw prompt 中文")
		self.assertEqual(json.loads(opener.calls[0][0].data.decode("utf-8"))["prompt"], "raw prompt 中文")

	def test_per_call_budget_overrides_default_and_is_validated(self):
		client, opener = self.make_client(max_tokens=256)
		client.complete("prompt", max_tokens=1024, timeout=3)
		self.assertEqual(json.loads(opener.calls[0][0].data.decode("utf-8"))["n_predict"], 1024)
		self.assertEqual(opener.calls[0][1], 3)
		for budget in (0, -5, True, 1.5):
			with self.assertRaisesRegex(LocalLLMError, "上限"):
				client.complete("prompt", max_tokens=budget)

	def test_alias_and_stripped_result(self):
		self.assertIs(LocalChatClient, LocalLLMClient)
		client, _ = self.make_client(response("  完成。\n"))
		self.assertEqual(client.complete("prompt"), "完成。")

	def test_invalid_json_and_malformed_content_are_rejected_without_secrets(self):
		client, _ = self.make_client(b"not-json SECRET_BODY")
		with self.assertRaisesRegex(LocalLLMError, "JSON") as caught:
			client.complete("SECRET_PROMPT")
		self.assertNotIn("SECRET_BODY", str(caught.exception))
		self.assertNotIn("SECRET_PROMPT", str(caught.exception))
		for body in ([], {}, {"content": None}, {"content": 7}, {"content": "  "}):
			with self.subTest(body=body):
				client, _ = self.make_client(json.dumps(body).encode("utf-8"))
				with self.assertRaisesRegex(LocalLLMError, "格式"):
					client.complete("prompt")

	def test_truncation_is_rejected(self):
		for extra, kind in (({"stop_type": "limit"}, "truncated"), ({"stopped_limit": True, "stop_type": None}, "truncated"), ({"truncated": True}, "context")):
			client, _ = self.make_client(response("partial", **extra))
			with self.assertRaisesRegex(LocalLLMError, "截斷") as caught:
				client.complete("prompt")
			self.assertEqual(caught.exception.kind, kind)
			self.assertTrue(caught.exception.too_large)

	def test_http_and_connection_failures_do_not_expose_body_or_retry(self):
		error = urllib.error.HTTPError("http://127.0.0.1:8766/completion", 503, "failure", {}, io.BytesIO(b"SECRET_BODY"))
		client, opener = self.make_client(error)
		with self.assertRaisesRegex(LocalLLMError, "HTTP 503") as caught:
			client.complete("SECRET_PROMPT")
		self.assertTrue(caught.exception.retryable)
		self.assertNotIn("SECRET_BODY", str(caught.exception))
		self.assertEqual(len(opener.calls), 1)
		client, opener = self.make_client(urllib.error.URLError(socket.timeout("timed out")))
		with mock.patch("time.sleep") as sleep:
			with self.assertRaisesRegex(LocalLLMError, "連線") as caught:
				client.complete("prompt")
		self.assertTrue(caught.exception.retryable)
		self.assertEqual(caught.exception.kind, "timeout")
		sleep.assert_not_called()
		self.assertEqual(len(opener.calls), 1)
		client, opener = self.make_client(http.client.IncompleteRead(b"SECRET", 100))
		with self.assertRaisesRegex(LocalLLMError, "連線") as caught:
			client.complete("prompt")
		self.assertTrue(caught.exception.retryable)
		self.assertTrue(caught.exception.unreachable)
		self.assertEqual(len(opener.calls), 1)

	def test_context_overflow_is_classified_for_smaller_batches(self):
		body = json.dumps({"error": {"code": 400, "message": "request (5001 tokens) exceeds the available context size (2048 tokens), try increasing it", "type": "exceed_context_size_error"}}).encode()
		error = urllib.error.HTTPError("http://127.0.0.1:8766/v1/chat/completions", 400, "bad", {}, io.BytesIO(body))
		client, _ = self.make_client(error, mode="instruct")
		with self.assertRaises(LocalLLMError) as caught:
			client.chat("s", "SECRET_PROMPT")
		self.assertEqual(caught.exception.kind, "context")
		self.assertTrue(caught.exception.too_large)
		self.assertFalse(caught.exception.retryable)
		self.assertNotIn("SECRET_PROMPT", str(caught.exception))

	def test_server_error_message_is_reported_but_bounded(self):
		body = json.dumps({"error": {"code": 400, "message": "model is required" + "x" * 500, "type": "invalid_request_error"}}).encode()
		error = urllib.error.HTTPError("http://127.0.0.1:8766/v1/chat/completions", 400, "bad", {}, io.BytesIO(body))
		client, _ = self.make_client(error, mode="instruct")
		with self.assertRaises(LocalLLMError) as caught:
			client.chat("s", "u")
		self.assertIn("model is required", str(caught.exception))
		self.assertLess(len(str(caught.exception)), 400)
		self.assertEqual(caught.exception.kind, "http")

	def test_non_retryable_response_errors_are_marked_permanent(self):
		for status in (400, 401, 404, 422):
			error = urllib.error.HTTPError(
				"http://127.0.0.1:8766/completion", status, "failure", {}, io.BytesIO(b"bad")
			)
			client, _ = self.make_client(error)
			with self.assertRaises(LocalLLMError) as caught:
				client.complete("prompt")
			self.assertFalse(caught.exception.retryable)
		client, _ = self.make_client(response("partial", stop_type="limit"))
		with self.assertRaises(LocalLLMError) as caught:
			client.complete("prompt")
		self.assertFalse(caught.exception.retryable)

	def test_configuration_defaults_environment_and_validation(self):
		client, _ = self.make_client()
		self.assertEqual((client.base_url, client.timeout, client.max_tokens, client.requested_mode, client.model), ("http://127.0.0.1:8766", 120.0, 256, "auto", None))
		with mock.patch.dict(os.environ, {"LOCAL_LLM_BASE_URL": "https://llm.example.test/", "LOCAL_LLM_TIMEOUT": "42.5", "LOCAL_LLM_MAX_TOKENS": "512", "LOCAL_LLM_MODE": " Riva ", "LOCAL_LLM_MODEL": "m"}):
			client, _ = self.make_client()
		self.assertEqual((client.base_url, client.timeout, client.max_tokens, client.mode, client.model), ("https://llm.example.test", 42.5, 512, "riva", "m"))
		for base_url in ("ftp://localhost", "http:///", "localhost:8766", "http://localhost/completion", "http://localhost?x=1"):
			with self.assertRaisesRegex(LocalLLMError, "網址"):
				self.make_client(base_url=base_url)
		for timeout in (0, -1, float("nan"), float("inf"), "soon"):
			with self.assertRaisesRegex(LocalLLMError, "逾時"):
				self.make_client(timeout=timeout)
		for max_tokens in (0, -1, "0", "1.5", "soon", True):
			with self.assertRaisesRegex(LocalLLMError, "上限"):
				self.make_client(max_tokens=max_tokens)
		for mode in ("chat", "", 3):
			with self.assertRaisesRegex(LocalLLMError, "模式") as caught:
				self.make_client(mode=mode)
			self.assertEqual(caught.exception.kind, "config")


class FakeLlamaServer(BaseHTTPRequestHandler):
	"""Minimal llama-server stand-in covering the endpoints the client uses."""

	requests = []
	props = {}
	health_statuses = []
	has_health = True
	has_props = True
	redirect = False

	def _send_json(self, status, body):
		raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(raw)))
		self.end_headers()
		self.wfile.write(raw)

	def do_GET(self):
		type(self).requests.append(("GET", self.path, b""))
		if self.path == "/health" and type(self).has_health:
			status = type(self).health_statuses.pop(0) if type(self).health_statuses else 200
			if status == 503:
				self._send_json(503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
			else:
				self._send_json(200, {"status": "ok"})
		elif self.path == "/props" and type(self).has_props:
			self._send_json(200, type(self).props)
		elif self.path == "/v1/models":
			self._send_json(200, {"object": "list", "data": [{"id": "qwen2.5:14b"}]})
		else:
			self._send_json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

	def do_POST(self):
		length = int(self.headers["Content-Length"])
		type(self).requests.append(("POST", self.path, self.rfile.read(length)))
		if type(self).redirect:
			self.send_response(307)
			self.send_header("Location", "/completion")
			self.end_headers()
			return
		if self.path == "/completion":
			self._send_json(200, {"content": " 本機回覆 ", "stop_type": "eos", "truncated": False})
		elif self.path == "/v1/chat/completions":
			self._send_json(200, {"choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": " 聊天回覆 "}}]})
		else:
			self._send_json(404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})

	def log_message(self, *_):
		pass


class LocalLLMClientLoopbackTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaServer)
		cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
		cls.thread.start()
	@classmethod
	def tearDownClass(cls):
		cls.server.shutdown()
		cls.server.server_close()
		cls.thread.join(timeout=5)
	def setUp(self):
		FakeLlamaServer.requests = []
		FakeLlamaServer.props = {"chat_template": "{{ messages }}<|im_start|>", "model_alias": "qwen2.5-14b-instruct", "model_path": "/models/qwen2.5-14b-instruct-q4_k_m.gguf", "total_slots": 1, "default_generation_settings": {"n_ctx": 8192}}
		FakeLlamaServer.health_statuses = []
		FakeLlamaServer.has_health = True
		FakeLlamaServer.has_props = True
		FakeLlamaServer.redirect = False
	def base_url(self):
		return f"http://127.0.0.1:{self.server.server_port}"
	def test_real_http_preserves_non_ascii_payload(self):
		client = LocalLLMClient(base_url=self.base_url(), timeout=2, mode="riva")
		self.assertEqual(client.chat("日本語を繁體中文に翻譯", "よろしく。"), "本機回覆")
		method, path, raw_body = FakeLlamaServer.requests[0]
		self.assertEqual((method, path), ("POST", "/completion"))
		self.assertIn("よろしく。", json.loads(raw_body.decode("utf-8"))["prompt"])
	def test_auto_mode_uses_chat_for_general_instruction_models(self):
		client = LocalLLMClient(base_url=self.base_url(), timeout=2)
		self.assertEqual(client.chat("系統", "使用者"), "聊天回覆")
		self.assertEqual(client.mode, MODE_INSTRUCT)
		self.assertEqual([(m, p) for m, p, _ in FakeLlamaServer.requests], [("GET", "/props"), ("POST", "/v1/chat/completions")])
		self.assertEqual(client.server_info["n_ctx"], 8192)
	def test_auto_mode_detects_riva_by_template_or_model_name(self):
		for props in (
			{"chat_template": "{{ '<s>System\\n' }}{{ system }}</s>\n<s>User\n{{ user }}</s>\n<s>Assistant\n", "model_path": "/m/model.gguf"},
			{"chat_template": "", "model_alias": "Riva-Translate-4B-Instruct"},
			{"model_path": "/models/riva-translate-4b-instruct-q8_0.gguf"},
		):
			with self.subTest(props=props):
				FakeLlamaServer.props = props
				client = LocalLLMClient(base_url=self.base_url(), timeout=2)
				self.assertEqual(client.mode, MODE_RIVA)
	def test_auto_mode_falls_back_to_chat_without_props(self):
		FakeLlamaServer.has_props = False
		client = LocalLLMClient(base_url=self.base_url(), timeout=2)
		self.assertEqual(client.mode, MODE_INSTRUCT)
	def test_ensure_ready_waits_for_model_loading(self):
		FakeLlamaServer.health_statuses = [503, 503, 200]
		client = LocalLLMClient(base_url=self.base_url(), timeout=2)
		waits = []
		client.ensure_ready(wait_seconds=30, poll_interval=0, on_wait=waits.append)
		self.assertEqual(len(waits), 2)
		self.assertTrue(all(error.kind == "not_ready" for error in waits))
	def test_ensure_ready_gives_up_after_deadline(self):
		FakeLlamaServer.health_statuses = [503] * 10
		client = LocalLLMClient(base_url=self.base_url(), timeout=2)
		ticks = iter([0.0, 1.0, 2.0, 50.0])
		with self.assertRaises(LocalLLMError) as caught:
			client.ensure_ready(wait_seconds=5, poll_interval=0, clock=lambda: next(ticks), sleep=lambda _: None)
		self.assertEqual(caught.exception.kind, "not_ready")
	def test_ensure_ready_accepts_openai_compatible_servers_without_health(self):
		FakeLlamaServer.has_health = False
		LocalLLMClient(base_url=self.base_url(), timeout=2).ensure_ready()
		self.assertEqual([p for _, p, _ in FakeLlamaServer.requests], ["/health", "/v1/models"])
	def test_unreachable_server_is_reported_without_waiting_when_no_budget(self):
		with socket.socket() as probe:
			probe.bind(("127.0.0.1", 0))
			port = probe.getsockname()[1]
		client = LocalLLMClient(base_url=f"http://127.0.0.1:{port}", timeout=2)
		with self.assertRaises(LocalLLMError) as caught:
			client.ensure_ready()
		self.assertEqual(caught.exception.kind, "connection")
	def test_redirect_is_rejected(self):
		FakeLlamaServer.redirect = True
		client = LocalLLMClient(base_url=self.base_url(), timeout=2, mode="riva")
		with self.assertRaisesRegex(LocalLLMError, "HTTP 307"):
			client.complete("prompt")
		self.assertEqual(len(FakeLlamaServer.requests), 1)
	def test_proxy_environment_is_ignored(self):
		with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1", "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}, clear=False):
			self.assertEqual(LocalLLMClient(base_url=self.base_url(), timeout=2, mode="riva").complete("prompt"), "本機回覆")


if __name__ == "__main__":
	unittest.main()
