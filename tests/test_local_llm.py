import io
import json
import os
import socket
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from local_llm import LocalChatClient, LocalLLMError


class FakeResponse:
	def __init__(self, body):
		self._body = body

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_value, traceback):
		return False

	def read(self):
		return self._body


class FakeOpener:
	def __init__(self, result):
		self.result = result
		self.calls = []

	def open(self, request, timeout):
		self.calls.append((request, timeout))
		if isinstance(self.result, BaseException):
			raise self.result
		return FakeResponse(self.result)


def chat_response(content="翻譯結果", finish_reason="stop"):
	return json.dumps({
		"choices": [{
			"message": {"role": "assistant", "content": content},
			"finish_reason": finish_reason,
		}],
	}).encode("utf-8")


class LocalChatClientContractTests(unittest.TestCase):
	def setUp(self):
		self.environment = mock.patch.dict(os.environ, {}, clear=True)
		self.environment.start()
		self.addCleanup(self.environment.stop)

	def make_client(self, result=chat_response(), **kwargs):
		opener = FakeOpener(result)
		with mock.patch("local_llm.urllib.request.build_opener", return_value=opener):
			client = LocalChatClient(**kwargs)
		return client, opener

	def test_chat_preserves_utf8_prompts_and_sends_one_nonstreaming_request(self):
		client, opener = self.make_client(
			base_url="http://127.0.0.1:8080/v1",
			model="local-model",
			timeout=12.5,
		)

		result = client.chat("只翻譯目前句子", "前一句：田中です。\n目前：よろしく。")

		self.assertEqual(result, "翻譯結果")
		self.assertEqual(len(opener.calls), 1)
		request, timeout = opener.calls[0]
		payload = json.loads(request.data.decode("utf-8"))
		self.assertEqual(request.full_url, "http://127.0.0.1:8080/v1/chat/completions")
		self.assertEqual(payload["messages"], [
			{"role": "system", "content": "只翻譯目前句子"},
			{"role": "user", "content": "前一句：田中です。\n目前：よろしく。"},
		])
		self.assertEqual(payload["model"], "local-model")
		self.assertEqual(payload["temperature"], 0.3)
		self.assertIs(payload["stream"], False)
		self.assertEqual(request.method, "POST")
		self.assertEqual(request.headers["Content-type"], "application/json")
		self.assertEqual(timeout, 12.5)

	def test_chat_returns_stripped_first_choice_content(self):
		client, _ = self.make_client(chat_response("  完成。\n"))

		self.assertEqual(client.chat("system", "user"), "完成。")

	def test_invalid_json_is_reported_without_response_body(self):
		client, _ = self.make_client(b"not-json SECRET_BODY")

		with self.assertRaisesRegex(LocalLLMError, "JSON") as caught:
			client.chat("SECRET_PROMPT", "user")
		self.assertNotIn("SECRET_BODY", str(caught.exception))
		self.assertNotIn("SECRET_PROMPT", str(caught.exception))

	def test_empty_choices_is_a_response_shape_error(self):
		client, _ = self.make_client(json.dumps({"choices": []}).encode("utf-8"))

		with self.assertRaisesRegex(LocalLLMError, "格式"):
			client.chat("system", "user")

	def test_malformed_response_containers_are_response_shape_errors(self):
		malformed = [
			[],
			{"choices": None},
			{"choices": [None]},
			{"choices": [{"message": None}]},
			{"choices": [{"message": {}}]},
		]
		for body in malformed:
			with self.subTest(body=body):
				client, _ = self.make_client(json.dumps(body).encode("utf-8"))
				with self.assertRaisesRegex(LocalLLMError, "格式"):
					client.chat("system", "user")

	def test_null_nonstring_and_empty_content_are_response_shape_errors(self):
		for content in (None, 7, "", " \n"):
			with self.subTest(content=content):
				client, _ = self.make_client(chat_response(content))
				with self.assertRaisesRegex(LocalLLMError, "格式"):
					client.chat("system", "user")

	def test_length_finish_reason_is_a_truncation_error(self):
		client, _ = self.make_client(chat_response("partial", "length"))

		with self.assertRaisesRegex(LocalLLMError, "截斷"):
			client.chat("system", "user")

	def test_refusal_is_reported_without_returning_empty_content(self):
		body = json.dumps({
			"choices": [{
				"message": {"role": "assistant", "content": None, "refusal": "cannot comply"},
				"finish_reason": "stop",
			}],
		}).encode("utf-8")
		client, _ = self.make_client(body)

		with self.assertRaisesRegex(LocalLLMError, "拒絕"):
			client.chat("system", "user")

	def test_http_failures_include_status_and_endpoint_guidance(self):
		for status in (400, 404, 429, 503):
			with self.subTest(status=status):
				error = urllib.error.HTTPError(
					"http://127.0.0.1:8080/v1/chat/completions",
					status,
					"failure",
					{},
					io.BytesIO(b"SECRET_BODY"),
				)
				client, _ = self.make_client(error)
				with self.assertRaises(LocalLLMError) as caught:
					client.chat("SECRET_PROMPT", "user")
				message = str(caught.exception)
				self.assertIn(f"HTTP {status}", message)
				self.assertIn("/v1/chat/completions", message)
				self.assertNotIn("SECRET_BODY", message)
				self.assertNotIn("SECRET_PROMPT", message)

	def test_timeout_is_a_connection_error_and_is_not_retried(self):
		client, opener = self.make_client(urllib.error.URLError(socket.timeout("timed out")))

		with mock.patch("time.sleep") as sleep:
			with self.assertRaisesRegex(LocalLLMError, "連線"):
				client.chat("system", "user")
		sleep.assert_not_called()
		self.assertEqual(len(opener.calls), 1)

	def test_defaults_are_read_when_configuration_is_absent(self):
		client, _ = self.make_client()

		self.assertEqual(client.base_url, "http://127.0.0.1:8080/v1")
		self.assertEqual(client.model, "local-model")
		self.assertEqual(client.timeout, 30.0)

	def test_environment_configuration_overrides_defaults(self):
		with mock.patch.dict(os.environ, {
			"LOCAL_LLM_BASE_URL": "https://llm.example.test/v1",
			"LOCAL_LLM_MODEL": "env-model",
			"LOCAL_LLM_TIMEOUT": "42.5",
		}):
			client, _ = self.make_client()

		self.assertEqual(client.base_url, "https://llm.example.test/v1")
		self.assertEqual(client.model, "env-model")
		self.assertEqual(client.timeout, 42.5)

	def test_explicit_configuration_overrides_environment(self):
		with mock.patch.dict(os.environ, {
			"LOCAL_LLM_BASE_URL": "https://wrong.example/v1",
			"LOCAL_LLM_MODEL": "wrong-model",
			"LOCAL_LLM_TIMEOUT": "99",
		}):
			client, _ = self.make_client(
				base_url="http://localhost:9090",
				model="explicit-model",
				timeout=4,
			)

		self.assertEqual(client.base_url, "http://localhost:9090/v1")
		self.assertEqual(client.model, "explicit-model")
		self.assertEqual(client.timeout, 4.0)

	def test_base_url_is_normalized_once(self):
		for supplied in (
			"http://localhost:8080",
			"http://localhost:8080/",
			"http://localhost:8080/v1",
			"http://localhost:8080/v1/",
		):
			with self.subTest(supplied=supplied):
				client, _ = self.make_client(base_url=supplied)
				self.assertEqual(client.base_url, "http://localhost:8080/v1")

	def test_invalid_base_urls_are_rejected(self):
		invalid = (
			"ftp://localhost:8080",
			"http:///v1",
			"localhost:8080/v1",
			"http://localhost:8080/api",
			"http://localhost:8080/v1/translate",
			"http://localhost:8080/completion",
			"http://localhost:8080/v1/chat/completions",
			"http://localhost:8080/v1?key=value",
			"http://localhost:8080/v1#fragment",
		)
		for supplied in invalid:
			with self.subTest(supplied=supplied):
				with self.assertRaisesRegex(LocalLLMError, "網址"):
					self.make_client(base_url=supplied)

	def test_empty_model_is_rejected(self):
		for model in ("", " \t"):
			with self.subTest(model=model):
				with self.assertRaisesRegex(LocalLLMError, "模型"):
					self.make_client(model=model)

	def test_nonpositive_and_nonfinite_timeouts_are_rejected(self):
		for timeout in (0, -1, float("nan"), float("inf"), float("-inf")):
			with self.subTest(timeout=timeout):
				with self.assertRaisesRegex(LocalLLMError, "逾時"):
					self.make_client(timeout=timeout)

	def test_nonnumeric_environment_timeout_is_rejected_as_configuration(self):
		with mock.patch.dict(os.environ, {"LOCAL_LLM_TIMEOUT": "soon"}):
			with self.assertRaisesRegex(LocalLLMError, "逾時"):
				self.make_client()


class LoopbackHandler(BaseHTTPRequestHandler):
	requests = []
	redirect = False

	def do_POST(self):
		length = int(self.headers["Content-Length"])
		body = self.rfile.read(length)
		type(self).requests.append((self.path, body))
		if type(self).redirect:
			self.send_response(307)
			self.send_header("Location", "/v1/chat/completions")
			self.end_headers()
			return
		response = chat_response(" 本機回覆 ")
		self.send_response(200)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(response)))
		self.end_headers()
		self.wfile.write(response)

	def log_message(self, format, *args):
		pass


class LocalChatClientLoopbackTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		LoopbackHandler.requests = []
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

	def test_chat_traverses_real_http_and_preserves_non_ascii_payload(self):
		client = LocalChatClient(base_url=self.base_url(), timeout=2)

		result = client.chat("日本語を繁體中文に翻譯", "よろしく。")

		self.assertEqual(result, "本機回覆")
		self.assertEqual(len(LoopbackHandler.requests), 1)
		path, raw_body = LoopbackHandler.requests[0]
		self.assertEqual(path, "/v1/chat/completions")
		payload = json.loads(raw_body.decode("utf-8"))
		self.assertEqual(payload["messages"][1]["content"], "よろしく。")

	def test_redirect_is_rejected(self):
		LoopbackHandler.redirect = True
		client = LocalChatClient(base_url=self.base_url(), timeout=2)

		with self.assertRaisesRegex(LocalLLMError, "HTTP 307"):
			client.chat("system", "user")
		self.assertEqual(len(LoopbackHandler.requests), 1)

	def test_proxy_environment_is_ignored_for_local_inference(self):
		proxy = "http://127.0.0.1:1"
		with mock.patch.dict(os.environ, {
			"HTTP_PROXY": proxy,
			"HTTPS_PROXY": proxy,
			"ALL_PROXY": proxy,
			"NO_PROXY": "",
		}, clear=False):
			client = LocalChatClient(base_url=self.base_url(), timeout=2)
			self.assertEqual(client.chat("system", "user"), "本機回覆")

	def base_url(self, path=""):
		return f"http://127.0.0.1:{self.server.server_port}{path}"


if __name__ == "__main__":
	unittest.main()
