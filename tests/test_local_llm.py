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

from local_llm import LocalChatClient, LocalLLMClient, LocalLLMError


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

	def test_chat_uses_exact_prompt_and_sampling_contract(self):
		client, opener = self.make_client(timeout=12.5, max_tokens=200)
		self.assertEqual(client.chat("只翻譯目前句子", "前一句：田中です。\n目前：よろしく。"), "翻譯結果")
		request, timeout = opener.calls[0]
		payload = json.loads(request.data.decode("utf-8"))
		self.assertEqual(request.full_url, "http://127.0.0.1:8766/completion")
		self.assertEqual(payload["prompt"], "<s>System\n只翻譯目前句子</s>\n<s>User\n前一句：田中です。\n目前：よろしく。</s>\n<s>Assistant\n")
		self.assertEqual({key: payload[key] for key in ("temperature", "top_p", "top_k", "repeat_penalty", "seed", "n_predict", "stop", "stream")}, {"temperature": 0, "top_p": 1, "top_k": 0, "repeat_penalty": 1, "seed": 1234, "n_predict": 200, "stop": ["</s>", "<s>"], "stream": False})
		self.assertEqual(timeout, 12.5)

	def test_complete_sends_raw_prompt(self):
		client, opener = self.make_client()
		client.complete("raw prompt 中文")
		self.assertEqual(json.loads(opener.calls[0][0].data.decode("utf-8"))["prompt"], "raw prompt 中文")

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
		for extra in ({"stop_type": "limit"}, {"truncated": True}):
			client, _ = self.make_client(response("partial", **extra))
			with self.assertRaisesRegex(LocalLLMError, "截斷"):
				client.complete("prompt")

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
		sleep.assert_not_called()
		self.assertEqual(len(opener.calls), 1)
		client, opener = self.make_client(http.client.IncompleteRead(b"SECRET", 100))
		with self.assertRaisesRegex(LocalLLMError, "連線") as caught:
			client.complete("prompt")
		self.assertTrue(caught.exception.retryable)
		self.assertEqual(len(opener.calls), 1)

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
		self.assertEqual((client.base_url, client.timeout, client.max_tokens), ("http://127.0.0.1:8766", 120.0, 256))
		with mock.patch.dict(os.environ, {"LOCAL_LLM_BASE_URL": "https://llm.example.test/", "LOCAL_LLM_TIMEOUT": "42.5", "LOCAL_LLM_MAX_TOKENS": "512"}):
			client, _ = self.make_client()
		self.assertEqual((client.base_url, client.timeout, client.max_tokens), ("https://llm.example.test", 42.5, 512))
		for base_url in ("ftp://localhost", "http:///", "localhost:8766", "http://localhost/completion", "http://localhost?x=1"):
			with self.assertRaisesRegex(LocalLLMError, "網址"):
				self.make_client(base_url=base_url)
		for timeout in (0, -1, float("nan"), float("inf"), "soon"):
			with self.assertRaisesRegex(LocalLLMError, "逾時"):
				self.make_client(timeout=timeout)
		for max_tokens in (0, -1, "0", "1.5", "soon", True):
			with self.assertRaisesRegex(LocalLLMError, "上限"):
				self.make_client(max_tokens=max_tokens)


class LoopbackHandler(BaseHTTPRequestHandler):
	requests = []
	redirect = False
	def do_POST(self):
		length = int(self.headers["Content-Length"])
		type(self).requests.append((self.path, self.rfile.read(length)))
		if type(self).redirect:
			self.send_response(307)
			self.send_header("Location", "/completion")
			self.end_headers()
			return
		body = response(" 本機回覆 ")
		self.send_response(200)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(body)))
		self.end_headers()
		self.wfile.write(body)
	def log_message(self, *_):
		pass


class LocalLLMClientLoopbackTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.server = ThreadingHTTPServer(("127.0.0.1", 0), LoopbackHandler)
		cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
		cls.thread.start()
	@classmethod
	def tearDownClass(cls):
		cls.server.shutdown()
		cls.server.server_close()
		cls.thread.join(timeout=5)
	def setUp(self):
		LoopbackHandler.requests = []
		LoopbackHandler.redirect = False
	def base_url(self):
		return f"http://127.0.0.1:{self.server.server_port}"
	def test_real_http_preserves_non_ascii_payload(self):
		client = LocalLLMClient(base_url=self.base_url(), timeout=2)
		self.assertEqual(client.chat("日本語を繁體中文に翻譯", "よろしく。"), "本機回覆")
		path, raw_body = LoopbackHandler.requests[0]
		self.assertEqual(path, "/completion")
		self.assertIn("よろしく。", json.loads(raw_body.decode("utf-8"))["prompt"])
	def test_redirect_is_rejected(self):
		LoopbackHandler.redirect = True
		client = LocalLLMClient(base_url=self.base_url(), timeout=2)
		with self.assertRaisesRegex(LocalLLMError, "HTTP 307"):
			client.complete("prompt")
		self.assertEqual(len(LoopbackHandler.requests), 1)
	def test_proxy_environment_is_ignored(self):
		with mock.patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:1", "HTTPS_PROXY": "http://127.0.0.1:1", "ALL_PROXY": "http://127.0.0.1:1", "NO_PROXY": ""}, clear=False):
			self.assertEqual(LocalLLMClient(base_url=self.base_url(), timeout=2).complete("prompt"), "本機回覆")


if __name__ == "__main__":
	unittest.main()
