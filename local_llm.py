import json
import math
import os
import socket
import urllib.error
import urllib.parse
import urllib.request


class LocalLLMError(RuntimeError):
	"""Report local language-model configuration and transport failures."""


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
	def redirect_request(self, req, fp, code, msg, headers, newurl):
		return None


class LocalChatClient:
	"""Call a local OpenAI-compatible chat-completions endpoint."""

	def __init__(self, base_url=None, model=None, timeout=None):
		configured_base_url = (
			base_url if base_url is not None
			else os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080/v1")
		)
		configured_model = (
			model if model is not None
			else os.environ.get("LOCAL_LLM_MODEL", "local-model")
		)
		configured_timeout = (
			timeout if timeout is not None
			else os.environ.get("LOCAL_LLM_TIMEOUT", "30")
		)

		self.base_url = self._validate_base_url(configured_base_url)
		self.model = self._validate_model(configured_model)
		self.timeout = self._validate_timeout(configured_timeout)
		self._opener = urllib.request.build_opener(
			urllib.request.ProxyHandler({}),
			_RejectRedirects(),
		)

	@staticmethod
	def _validate_base_url(value):
		if not isinstance(value, str):
			raise LocalLLMError("本機模型網址設定無效：請使用 HTTP(S) 網址。")
		try:
			parsed = urllib.parse.urlsplit(value)
			_ = parsed.port
		except ValueError as error:
			raise LocalLLMError("本機模型網址設定無效：請檢查主機與連接埠。") from error
		if (
			parsed.scheme not in ("http", "https")
			or not parsed.hostname
			or parsed.query
			or parsed.fragment
			or parsed.path.rstrip("/") not in ("", "/v1")
		):
			raise LocalLLMError(
				"本機模型網址設定無效：請指定服務根網址或 /v1，不可指定 translate 或 completion 路徑。"
			)
		return value.rstrip("/") + ("" if parsed.path.rstrip("/") == "/v1" else "/v1")

	@staticmethod
	def _validate_model(value):
		if not isinstance(value, str) or not value.strip():
			raise LocalLLMError("本機模型名稱設定無效：請指定非空白模型名稱。")
		return value

	@staticmethod
	def _validate_timeout(value):
		try:
			numeric = float(value)
		except (TypeError, ValueError) as error:
			raise LocalLLMError("本機模型逾時設定無效：請指定正數秒數。") from error
		if not math.isfinite(numeric) or numeric <= 0:
			raise LocalLLMError("本機模型逾時設定無效：請指定有限的正數秒數。")
		return numeric

	def chat(self, system_prompt, user_prompt):
		payload = {
			"model": self.model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"temperature": 0.3,
			"stream": False,
		}
		request = urllib.request.Request(
			self.base_url + "/chat/completions",
			data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		try:
			with self._opener.open(request, timeout=self.timeout) as response:
				raw_response = response.read()
		except urllib.error.HTTPError as error:
			error.close()
			raise LocalLLMError(
				f"本機模型 HTTP {error.code} 錯誤：請確認 {request.full_url} 端點與服務狀態。"
			) from error
		except (urllib.error.URLError, socket.timeout, TimeoutError, OSError) as error:
			raise LocalLLMError(
				f"本機模型連線失敗：請確認服務已啟動且可連線至 {request.full_url}。"
			) from error

		try:
			result = json.loads(raw_response.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError) as error:
			raise LocalLLMError(
				f"本機模型 JSON 回應無效：請確認 {request.full_url} 為相容的 chat 端點。"
			) from error

		choice = self._first_choice(result, request.full_url)
		if choice.get("finish_reason") == "length":
			raise LocalLLMError("本機模型回應遭長度限制截斷，未回傳不完整結果。")
		message = choice.get("message")
		if isinstance(message, dict) and message.get("refusal"):
			raise LocalLLMError("本機模型拒絕此要求；請檢查模型能力或提示內容。")
		if not isinstance(message, dict):
			raise self._shape_error(request.full_url)
		content = message.get("content")
		if not isinstance(content, str) or not content.strip():
			raise self._shape_error(request.full_url)
		return content.strip()

	@classmethod
	def _first_choice(cls, result, endpoint):
		if not isinstance(result, dict):
			raise cls._shape_error(endpoint)
		choices = result.get("choices")
		if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
			raise cls._shape_error(endpoint)
		return choices[0]

	@staticmethod
	def _shape_error(endpoint):
		return LocalLLMError(
			f"本機模型回應格式無效：請確認 {endpoint} 提供 OpenAI 相容的 chat 回應。"
		)
