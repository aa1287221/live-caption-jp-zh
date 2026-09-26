"""Standard-library client for a separately started local language-model server.

The primary target is llama.cpp's ``llama-server``. Two request styles are supported:

- ``instruct``: ``POST /v1/chat/completions``. The server applies the loaded model's own
  chat template, so a general instruction model receives exactly the system/user prompts
  the Gemini backend sends. Ollama, LM Studio and other OpenAI-compatible servers also work
  in this mode when ``LOCAL_LLM_MODEL`` names the model to use.
- ``riva``: ``POST /completion`` with Ontime-Translator's Riva-Translate turn markers and
  greedy sampling contract, for translation-only models.

``auto`` (the default) asks the server what it loaded (``GET /props``) and picks ``riva``
for Riva-Translate style templates or model names, otherwise ``instruct``.

This module never starts, stops or downloads anything. Errors never include prompts or
model output.
"""

import http.client
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request


MODE_AUTO = "auto"
MODE_INSTRUCT = "instruct"
MODE_RIVA = "riva"
_MODES = (MODE_AUTO, MODE_INSTRUCT, MODE_RIVA)

RIVA_TRANSLATION_INSTRUCTION = "Translate this into Traditional Chinese:"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_MAX_ERROR_DETAIL_CHARS = 200


class LocalLLMError(RuntimeError):
	"""Report a local model failure and how a caller may react to it.

	``kind`` is one of: ``config``, ``connection``, ``timeout``, ``http``, ``context``
	(prompt larger than the server context), ``truncated`` (output hit the token limit),
	``protocol``, ``empty`` or ``not_ready``.
	"""

	def __init__(
		self,
		message: str,
		*,
		retryable: bool = False,
		kind: str = "protocol",
		status: int | None = None,
	) -> None:
		super().__init__(message)
		self.retryable = retryable
		self.kind = kind
		self.status = status

	@property
	def too_large(self) -> bool:
		"""True when a smaller request (shorter prompt or output) could succeed."""
		return self.kind in ("context", "truncated")

	@property
	def unreachable(self) -> bool:
		"""True when the server could not be reached or did not answer in time."""
		return self.kind in ("connection", "timeout")


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
	def redirect_request(self, req, fp, code, msg, headers, newurl):
		return None


class LocalLLMClient:
	"""Call a local model server that the user started independently."""

	DEFAULT_BASE_URL = "http://127.0.0.1:8766"
	DEFAULT_TIMEOUT = 120.0
	DEFAULT_MAX_TOKENS = 256
	# Matches the Gemini backend's generation config so both backends sample alike.
	DEFAULT_TEMPERATURE = 0.3
	STOP_STRINGS = ["</s>", "<s>"]
	PROBE_TIMEOUT = 10.0

	def __init__(
		self,
		base_url: str | None = None,
		timeout: float | str | None = None,
		max_tokens: int | str | None = None,
		mode: str | None = None,
		model: str | None = None,
	) -> None:
		base_url = base_url if base_url is not None else os.environ.get("LOCAL_LLM_BASE_URL", self.DEFAULT_BASE_URL)
		timeout = timeout if timeout is not None else os.environ.get("LOCAL_LLM_TIMEOUT", str(self.DEFAULT_TIMEOUT))
		max_tokens = max_tokens if max_tokens is not None else os.environ.get("LOCAL_LLM_MAX_TOKENS", str(self.DEFAULT_MAX_TOKENS))
		mode = mode if mode is not None else os.environ.get("LOCAL_LLM_MODE", MODE_AUTO)
		model = model if model is not None else os.environ.get("LOCAL_LLM_MODEL", "")
		self.base_url = self._validate_base_url(base_url)
		self.timeout = self._validate_timeout(timeout)
		self.max_tokens = self._validate_max_tokens(max_tokens)
		self.requested_mode = self._validate_mode(mode)
		self.model = self._validate_model(model)
		self.server_info: dict = {}
		self._resolved_mode = None if self.requested_mode == MODE_AUTO else self.requested_mode
		self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirects())

	# ---------- configuration ----------

	@staticmethod
	def _validate_base_url(value: object) -> str:
		if not isinstance(value, str):
			raise LocalLLMError("本機模型網址設定無效：請使用 HTTP(S) 網址。", kind="config")
		try:
			parsed = urllib.parse.urlsplit(value)
			_ = parsed.port
		except ValueError as error:
			raise LocalLLMError("本機模型網址設定無效：請檢查主機與連接埠。", kind="config") from error
		if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.query or parsed.fragment or parsed.path.rstrip("/")):
			raise LocalLLMError("本機模型網址設定無效：請指定服務根網址，不可指定 completion 路徑。", kind="config")
		return value.rstrip("/")

	@staticmethod
	def _validate_timeout(value: object) -> float:
		if isinstance(value, bool):
			raise LocalLLMError("本機模型逾時設定無效：請指定正數秒數。", kind="config")
		try:
			numeric = float(value)
		except (TypeError, ValueError) as error:
			raise LocalLLMError("本機模型逾時設定無效：請指定正數秒數。", kind="config") from error
		if not math.isfinite(numeric) or numeric <= 0:
			raise LocalLLMError("本機模型逾時設定無效：請指定有限的正數秒數。", kind="config")
		return numeric

	@staticmethod
	def _validate_max_tokens(value: object) -> int:
		try:
			if isinstance(value, bool):
				raise ValueError
			numeric = int(value)
		except (TypeError, ValueError, OverflowError) as error:
			raise LocalLLMError("本機模型輸出上限設定無效：請指定正整數。", kind="config") from error
		if isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
			raise LocalLLMError("本機模型輸出上限設定無效：請指定正整數。", kind="config")
		if numeric <= 0 or (isinstance(value, str) and str(numeric) != value.strip()):
			raise LocalLLMError("本機模型輸出上限設定無效：請指定正整數。", kind="config")
		return numeric

	@staticmethod
	def _validate_mode(value: object) -> str:
		mode = value.strip().lower() if isinstance(value, str) else ""
		if mode not in _MODES:
			raise LocalLLMError("本機模型模式設定無效：LOCAL_LLM_MODE 請使用 auto、instruct 或 riva。", kind="config")
		return mode

	@staticmethod
	def _validate_model(value: object) -> str | None:
		if not isinstance(value, str):
			raise LocalLLMError("本機模型名稱設定無效：LOCAL_LLM_MODEL 必須是文字。", kind="config")
		return value.strip() or None

	# ---------- server discovery ----------

	@property
	def mode(self) -> str:
		"""The effective request style; ``auto`` is resolved on first use."""
		if self._resolved_mode is None:
			self._resolved_mode = self.detect_mode()
		return self._resolved_mode

	def detect_mode(self) -> str:
		"""Return ``riva`` for Riva-Translate style servers, otherwise ``instruct``."""
		try:
			props = self._get_json("/props", timeout=min(self.timeout, self.PROBE_TIMEOUT))
		except LocalLLMError as error:
			if error.unreachable:
				raise
			if error.status == 503:
				raise LocalLLMError("本機模型服務仍在載入模型，請稍候。", retryable=True, kind="not_ready", status=503) from error
			# OpenAI-compatible servers without /props (Ollama, LM Studio) are chat servers.
			return MODE_INSTRUCT
		if not isinstance(props, dict):
			return MODE_INSTRUCT
		settings = props.get("default_generation_settings")
		self.server_info = {
			"model_alias": props.get("model_alias"),
			"model_path": props.get("model_path"),
			"total_slots": props.get("total_slots"),
			"n_ctx": settings.get("n_ctx") if isinstance(settings, dict) else None,
		}
		template = props.get("chat_template") if isinstance(props.get("chat_template"), str) else ""
		names = " ".join(str(props.get(key) or "") for key in ("model_alias", "model_path")).lower()
		if ("<s>System" in template and "<s>Assistant" in template) or "riva" in names:
			return MODE_RIVA
		return MODE_INSTRUCT

	def ensure_ready(
		self,
		wait_seconds: float = 0.0,
		*,
		poll_interval: float = 1.0,
		on_wait=None,
		clock=time.monotonic,
		sleep=time.sleep,
	) -> None:
		"""Wait until the server answers its health check, up to ``wait_seconds``."""
		deadline = clock() + max(0.0, wait_seconds)
		while True:
			try:
				self._probe_health()
				return
			except LocalLLMError as error:
				waiting = error.unreachable or error.kind == "not_ready"
				if not waiting or clock() >= deadline:
					raise
				if on_wait is not None:
					on_wait(error)
				sleep(poll_interval)

	def _probe_health(self) -> None:
		timeout = min(self.timeout, self.PROBE_TIMEOUT)
		try:
			self._get_json("/health", timeout=timeout)
		except LocalLLMError as error:
			if error.kind == "http" and error.status == 503:
				raise LocalLLMError("本機模型服務仍在載入模型，請稍候。", retryable=True, kind="not_ready", status=503) from error
			if error.kind == "http" and error.status == 404:
				# OpenAI-compatible servers without /health still list their models.
				self._get_json("/v1/models", timeout=timeout)
				return
			raise

	# ---------- inference ----------

	def complete(self, prompt: str, *, max_tokens: int | None = None, timeout: float | None = None) -> str:
		"""Send one raw prompt to ``/completion`` using the Riva sampling contract."""
		if not isinstance(prompt, str):
			raise LocalLLMError("本機模型提示內容無效：請提供文字。")
		payload = {
			"prompt": prompt, "stream": False, "temperature": 0, "top_p": 1,
			"top_k": 0, "repeat_penalty": 1, "seed": 1234,
			"n_predict": self._budget(max_tokens), "stop": self.STOP_STRINGS,
		}
		result = self._post_json("/completion", payload, timeout)
		if not isinstance(result, dict):
			raise self._shape_error("/completion")
		if result.get("truncated") is True:
			raise LocalLLMError("本機模型提示內容超過服務的 context 長度而遭截斷，未回傳不完整結果。", kind="context")
		if result.get("stop_type") in ("limit", "length") or result.get("stopped_limit") is True:
			raise LocalLLMError("本機模型回應遭長度限制截斷，未回傳不完整結果。", kind="truncated")
		return self._validated_text(result.get("content"), "/completion")

	def chat(
		self,
		system_prompt: str,
		user_prompt: str,
		*,
		max_tokens: int | None = None,
		timeout: float | None = None,
		temperature: float | None = None,
	) -> str:
		"""Send one system/user exchange using the resolved request style."""
		if not isinstance(system_prompt, str) or not isinstance(user_prompt, str):
			raise LocalLLMError("本機模型提示內容無效：請提供文字。")
		if self.mode == MODE_RIVA:
			return self.complete(self.frame_riva_prompt(system_prompt, user_prompt), max_tokens=max_tokens, timeout=timeout)
		messages = [{"role": "user", "content": user_prompt}]
		if system_prompt:
			messages.insert(0, {"role": "system", "content": system_prompt})
		payload = {
			"messages": messages,
			"stream": False,
			"temperature": self.DEFAULT_TEMPERATURE if temperature is None else temperature,
			"max_tokens": self._budget(max_tokens),
			"seed": 1234,
			# Reasoning models would otherwise spend the caption's time budget thinking.
			"chat_template_kwargs": {"enable_thinking": False},
		}
		if self.model:
			payload["model"] = self.model
		result = self._post_json("/v1/chat/completions", payload, timeout)
		try:
			choice = result["choices"][0]
			content = choice["message"]["content"]
		except (KeyError, IndexError, TypeError) as error:
			raise self._shape_error("/v1/chat/completions") from error
		if choice.get("finish_reason") == "length":
			raise LocalLLMError("本機模型回應遭長度限制截斷，未回傳不完整結果。", kind="truncated")
		if isinstance(content, str):
			content = _THINK_BLOCK_RE.sub("", content)
		return self._validated_text(content, "/v1/chat/completions")

	@staticmethod
	def frame_riva_prompt(system_prompt: str, user_prompt: str) -> str:
		"""Frame a system/user exchange with the Riva-Translate turn markers."""
		return "<s>System\n" + system_prompt + "</s>\n<s>User\n" + user_prompt + "</s>\n<s>Assistant\n"

	# ---------- HTTP plumbing ----------

	def _budget(self, max_tokens: int | None) -> int:
		if max_tokens is None:
			return self.max_tokens
		if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
			raise LocalLLMError("本機模型輸出上限設定無效：請指定正整數。", kind="config")
		return max_tokens

	def _post_json(self, path: str, payload: dict, timeout: float | None):
		request = urllib.request.Request(
			self.base_url + path,
			data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
			headers={"Content-Type": "application/json"}, method="POST",
		)
		return self._decode(self._send(request, timeout), path)

	def _get_json(self, path: str, *, timeout: float | None):
		request = urllib.request.Request(self.base_url + path, method="GET")
		return self._decode(self._send(request, timeout), path)

	def _send(self, request: urllib.request.Request, timeout: float | None) -> bytes:
		timeout = self.timeout if timeout is None else timeout
		try:
			with self._opener.open(request, timeout=timeout) as response:
				return response.read()
		except urllib.error.HTTPError as error:
			detail, error_type = self._error_detail(error)
			error.close()
			status = error.code
			if status == 400 and (error_type == "exceed_context_size_error" or "context" in detail.lower()):
				raise LocalLLMError(
					"本機模型提示內容超過服務的 context 長度：請以較大的 -c 啟動 llama-server，程式會改用較小的批次。",
					kind="context", status=status,
				) from error
			suffix = f"（{detail}）" if detail else ""
			raise LocalLLMError(
				f"本機模型 HTTP {status} 錯誤{suffix}：請確認 {request.full_url} 端點與服務狀態。",
				retryable=status == 429 or status >= 500, kind="http", status=status,
			) from error
		except (socket.timeout, TimeoutError) as error:
			raise self._timeout_error(request.full_url, timeout) from error
		except urllib.error.URLError as error:
			if isinstance(error.reason, (socket.timeout, TimeoutError)):
				raise self._timeout_error(request.full_url, timeout) from error
			raise self._connection_error(request.full_url) from error
		except (http.client.HTTPException, OSError) as error:
			raise self._connection_error(request.full_url) from error

	@staticmethod
	def _error_detail(error: urllib.error.HTTPError) -> tuple[str, str]:
		"""Read the server's own error message; request bodies are never echoed back."""
		try:
			raw = error.read(4096)
			body = json.loads(raw.decode("utf-8"))
		except Exception:
			return "", ""
		inner = body.get("error") if isinstance(body, dict) else None
		if isinstance(inner, dict):
			message = inner.get("message")
			error_type = inner.get("type")
		elif isinstance(inner, str):
			message, error_type = inner, ""
		else:
			return "", ""
		message = message if isinstance(message, str) else ""
		return message[:_MAX_ERROR_DETAIL_CHARS], error_type if isinstance(error_type, str) else ""

	def _decode(self, raw: bytes, path: str):
		try:
			return json.loads(raw.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError) as error:
			raise LocalLLMError(f"本機模型 JSON 回應無效：請確認 {self.base_url}{path} 為相容的端點。") from error

	def _validated_text(self, content: object, path: str) -> str:
		if not isinstance(content, str):
			raise self._shape_error(path)
		if not content.strip():
			raise LocalLLMError(f"本機模型回應格式無效：{self.base_url}{path} 回傳了空白內容。", kind="empty")
		return content.strip()

	def _shape_error(self, path: str) -> LocalLLMError:
		return LocalLLMError(f"本機模型回應格式無效：請確認 {self.base_url}{path} 提供相容的回應。")

	@staticmethod
	def _timeout_error(endpoint: str, timeout: float) -> LocalLLMError:
		return LocalLLMError(
			f"本機模型連線逾時（{timeout:g} 秒）：請確認 {endpoint} 的服務沒有卡住或過載。",
			retryable=True, kind="timeout",
		)

	@staticmethod
	def _connection_error(endpoint: str) -> LocalLLMError:
		return LocalLLMError(
			f"本機模型連線失敗：請確認服務已啟動且可連線至 {endpoint}。",
			retryable=True, kind="connection",
		)


# Keep the old import name for callers that only depend on the client boundary.
LocalChatClient = LocalLLMClient
