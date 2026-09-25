import http.client
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


class LocalLLMClient:
	"""Call a local llama-server completion endpoint."""

	DEFAULT_BASE_URL = "http://127.0.0.1:8766"
	DEFAULT_TIMEOUT = 120.0
	DEFAULT_MAX_TOKENS = 256
	STOP_STRINGS = ["</s>", "<s>"]

	def __init__(self, base_url=None, timeout=None, max_tokens=None):
		base_url = base_url if base_url is not None else os.environ.get("LOCAL_LLM_BASE_URL", self.DEFAULT_BASE_URL)
		timeout = timeout if timeout is not None else os.environ.get("LOCAL_LLM_TIMEOUT", str(self.DEFAULT_TIMEOUT))
		max_tokens = max_tokens if max_tokens is not None else os.environ.get("LOCAL_LLM_MAX_TOKENS", str(self.DEFAULT_MAX_TOKENS))
		self.base_url = self._validate_base_url(base_url)
		self.timeout = self._validate_timeout(timeout)
		self.max_tokens = self._validate_max_tokens(max_tokens)
		self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirects())

	@staticmethod
	def _validate_base_url(value):
		if not isinstance(value, str):
			raise LocalLLMError("本機模型網址設定無效：請使用 HTTP(S) 網址。")
		try:
			parsed = urllib.parse.urlsplit(value)
			_ = parsed.port
		except ValueError as error:
			raise LocalLLMError("本機模型網址設定無效：請檢查主機與連接埠。") from error
		if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.path.rstrip("/")):
			raise LocalLLMError("本機模型網址設定無效：請指定服務根網址，不可指定 completion 路徑。")
		return value.rstrip("/")

	@staticmethod
	def _validate_timeout(value):
		if isinstance(value, bool):
			raise LocalLLMError("本機模型逾時設定無效：請指定正數秒數。")
		try:
			numeric = float(value)
		except (TypeError, ValueError) as error:
			raise LocalLLMError("本機模型逾時設定無效：請指定正數秒數。") from error
		if not math.isfinite(numeric) or numeric <= 0:
			raise LocalLLMError("本機模型逾時設定無效：請指定有限的正數秒數。")
		return numeric

	@staticmethod
	def _validate_max_tokens(value):
		try:
			if isinstance(value, bool):
				raise ValueError
			numeric = int(value)
		except (TypeError, ValueError, OverflowError) as error:
			raise LocalLLMError("本機模型輸出上限設定無效：請指定正整數。") from error
		if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
			raise LocalLLMError("本機模型輸出上限設定無效：請指定正整數。")
		if numeric <= 0 or (isinstance(value, str) and str(numeric) != value.strip()):
			raise LocalLLMError("本機模型輸出上限設定無效：請指定正整數。")
		return numeric

	def complete(self, prompt):
		if not isinstance(prompt, str):
			raise LocalLLMError("本機模型提示內容無效：請提供文字。")
		payload = {
			"prompt": prompt, "stream": False, "temperature": 0, "top_p": 1,
			"top_k": 0, "repeat_penalty": 1, "seed": 1234,
			"n_predict": self.max_tokens, "stop": self.STOP_STRINGS,
		}
		request = urllib.request.Request(
			self.base_url + "/completion",
			data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
			headers={"Content-Type": "application/json"}, method="POST",
		)
		try:
			with self._opener.open(request, timeout=self.timeout) as response:
				raw_response = response.read()
		except urllib.error.HTTPError as error:
			error.close()
			raise LocalLLMError(f"本機模型 HTTP {error.code} 錯誤：請確認 {request.full_url} 端點與服務狀態。") from error
		except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError, OSError) as error:
			raise LocalLLMError(f"本機模型連線失敗：請確認服務已啟動且可連線至 {request.full_url}。") from error
		try:
			result = json.loads(raw_response.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError) as error:
			raise LocalLLMError(f"本機模型 JSON 回應無效：請確認 {request.full_url} 為相容的 completion 端點。") from error
		if not isinstance(result, dict):
			raise self._shape_error(request.full_url)
		if result.get("stop_type") in ("limit", "length") or result.get("truncated") is True:
			raise LocalLLMError("本機模型回應遭長度限制截斷，未回傳不完整結果。")
		content = result.get("content")
		if not isinstance(content, str) or not content.strip():
			raise self._shape_error(request.full_url)
		return content.strip()

	@staticmethod
	def _shape_error(endpoint):
		return LocalLLMError(f"本機模型回應格式無效：請確認 {endpoint} 提供 completion 回應。")

	def chat(self, system_prompt, user_prompt):
		if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
			raise LocalLLMError("本機模型提示內容無效：請提供文字。")
		prompt = "<s>System\n" + system_prompt + "</s>\n<s>User\n" + user_prompt + "</s>\n<s>Assistant\n"
		return self.complete(prompt)


# Keep the old import name for callers that only depend on the client boundary.
LocalChatClient = LocalLLMClient
