"""HTTP client and lifecycle manager for the installed Ontime Riva relay."""

from __future__ import annotations

import http.client
import json
import logging
import math
import os
import platform
import re
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


LOGGER = logging.getLogger(__name__)
_MAX_ITEMS = 32
_MAX_CHARS = 1200
_SENTENCE_END = re.compile(r"(?<=[。！？!?\n])")


class RivaError(RuntimeError):
	def __init__(self, message: str, *, retryable: bool = False):
		super().__init__(message)
		self.retryable = retryable


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
	def redirect_request(self, req, fp, code, msg, headers, newurl):
		return None


class OntimeRivaClient:
	def __init__(self, base_url: str | None = None, timeout: float | None = None):
		self.base_url = self._validate_origin(
			base_url if base_url is not None else os.environ.get("ONTIME_RELAY_URL", "http://127.0.0.1:8765")
		)
		self.timeout = self._positive_timeout(
			timeout if timeout is not None else os.environ.get("ONTIME_TRANSLATE_TIMEOUT", "120"),
			"翻譯",
		)
		self.start_timeout = self._positive_timeout(os.environ.get("ONTIME_START_TIMEOUT", "180"), "啟動")
		self.repo_path = self._wsl_posix_path(os.environ.get("ONTIME_REPO_PATH", "/project/Ontime-Translator"))
		self.wsl_distro = os.environ.get("ONTIME_WSL_DISTRO", "Ubuntu-26.04")
		self.auto_start = self._boolean(os.environ.get("ONTIME_AUTO_START", "1"))
		self.log_path = Path(__file__).resolve().parent / "ontime-riva.log"
		self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _RejectRedirects())
		self._last_health_detail = ""

	@staticmethod
	def _validate_origin(value):
		if not isinstance(value, str):
			raise RivaError("Riva 網址設定無效：請使用 HTTP(S) 服務根網址。")
		try:
			parsed = urllib.parse.urlsplit(value)
			_ = parsed.port
		except ValueError as error:
			raise RivaError("Riva 網址設定無效：請檢查主機與連接埠。") from error
		if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path.rstrip("/") or parsed.query or parsed.fragment:
			raise RivaError("Riva 網址設定無效：請指定不含路徑的 HTTP(S) 服務根網址。")
		return value.rstrip("/")

	@staticmethod
	def _positive_timeout(value, label):
		try:
			result = float(value)
		except (TypeError, ValueError) as error:
			raise RivaError(f"Riva {label}逾時設定無效：請指定有限的正數秒數。") from error
		if not math.isfinite(result) or result <= 0:
			raise RivaError(f"Riva {label}逾時設定無效：請指定有限的正數秒數。")
		return result

	@staticmethod
	def _boolean(value):
		if value in ("1", "true", "TRUE", "yes", "YES"):
			return True
		if value in ("0", "false", "FALSE", "no", "NO"):
			return False
		raise RivaError("ONTIME_AUTO_START 設定無效：請使用 1 或 0。")

	@staticmethod
	def _wsl_posix_path(value):
		if not isinstance(value, str) or not value.startswith("/") or "\x00" in value:
			raise RivaError("ONTIME_REPO_PATH 設定無效：請使用 WSL 的絕對 POSIX 路徑。")
		return value.rstrip("/") or "/"

	def translate(self, text: str, glossary: dict | None = None) -> str:
		return self.translate_batch([text], glossary)[0]

	def translate_batch(self, texts: list[str], glossary: dict | None = None) -> list[str]:
		if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
			raise RivaError("翻譯輸入格式無效：必須是字串清單。")
		glossary_map = self._validate_glossary(glossary)
		results = list(texts)
		segments = []
		states = {}
		for original_index, text in enumerate(texts):
			if not text.strip():
				continue
			masked, markers = self._mask(text, glossary_map)
			states[original_index] = (markers, [])
			for segment in self._split(masked):
				segments.append((original_index, segment))
		for offset in range(0, len(segments), _MAX_ITEMS):
			group = segments[offset:offset + _MAX_ITEMS]
			translated = self._request(
				[segment for _, segment in group],
				[original_index for original_index, _ in group],
			)
			for (original_index, _), output in zip(group, translated):
				states[original_index][1].append(output)
		for original_index, (markers, pieces) in states.items():
			combined = "".join(pieces)
			restored = self._restore(combined, markers)
			if texts[original_index].strip() and not restored:
				raise RivaError(f"Riva 第 {original_index + 1} 句未回傳有效譯文。")
			results[original_index] = restored
		return results

	@staticmethod
	def _validate_glossary(glossary):
		if glossary is None:
			return {}
		if not isinstance(glossary, dict) or any(
			not isinstance(source, str) or not source or not isinstance(target, str)
			for source, target in glossary.items()
		):
			raise RivaError("術語表格式無效：來源必須是非空字串，目標必須是字串。")
		return glossary

	@staticmethod
	def _mask(text, glossary):
		literal_markers = re.findall(r"__GLOSSARY_\d+__", text)
		if not glossary and not literal_markers:
			return text, {}
		existing_numbers = [int(value) for value in re.findall(r"__GLOSSARY_(\d+)__", text)]
		start = max(existing_numbers, default=-1) + 1
		markers = {}
		counter = start
		for literal in literal_markers:
			marker = f"__GLOSSARY_{counter}__"
			counter += 1
			text = text.replace(literal, marker, 1)
			markers[marker] = literal
		if not glossary:
			return text, markers
		pattern = re.compile("|".join(re.escape(term) for term in sorted(glossary, key=len, reverse=True)))
		def replace(match):
			nonlocal counter
			marker = f"__GLOSSARY_{counter}__"
			while marker in text or marker in markers:
				counter += 1
				marker = f"__GLOSSARY_{counter}__"
			markers[marker] = glossary[match.group(0)]
			counter += 1
			return marker
		return pattern.sub(replace, text), markers

	@staticmethod
	def _split(text):
		if len(text) <= _MAX_CHARS:
			return [text]
		units = [unit for unit in _SENTENCE_END.split(text) if unit]
		pieces = []
		current = ""
		for unit in units:
			while len(unit) > _MAX_CHARS:
				if current:
					pieces.append(current)
					current = ""
				cut = _MAX_CHARS
				# A marker is short and atomic; move the boundary before it when necessary.
				open_marker = unit.rfind("__GLOSSARY_", 0, cut)
				if open_marker >= 0 and unit.find("__", open_marker + 11) >= cut:
					cut = open_marker
				if cut <= 0:
					cut = _MAX_CHARS
				pieces.append(unit[:cut])
				unit = unit[cut:]
			if len(current) + len(unit) > _MAX_CHARS:
				pieces.append(current)
				current = unit
			else:
				current += unit
		if current:
			pieces.append(current)
		return pieces

	def _restore(self, text, markers):
		for marker in re.findall(r"__GLOSSARY_\d+__", text):
			if marker not in markers:
				raise RivaError(f"術語標記完整性失敗：出現非預期標記 {marker}。")
		for marker, target in markers.items():
			if text.count(marker) != 1:
				raise RivaError(f"術語標記完整性失敗：{marker} 數量不符。")
		return re.sub(r"__GLOSSARY_\d+__", lambda match: markers[match.group(0)], text)

	def _request(self, texts, original_indices):
		endpoint = self.base_url + "/v1/translate"
		payload = {"text": texts, "source_lang": "JA", "target_lang": "zh-TW"}
		request = urllib.request.Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
		try:
			with self._opener.open(request, timeout=self.timeout) as response:
				raw = response.read()
		except urllib.error.HTTPError as error:
			status = error.code
			error.close()
			raise RivaError(f"Riva HTTP {status} 錯誤：{endpoint}", retryable=status in (429, 503)) from error
		except (urllib.error.URLError, http.client.HTTPException, socket.timeout, TimeoutError, OSError) as error:
			raise RivaError(f"Riva 連線或逾時失敗：{endpoint}", retryable=True) from error
		try:
			body = json.loads(raw.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError) as error:
			raise RivaError("Riva JSON 回應無效。") from error
		items = body.get("translations") if isinstance(body, dict) else None
		if not isinstance(items, list) or len(items) != len(texts):
			raise RivaError("Riva 回應筆數與輸入不一致。")
		outputs = []
		for index, item in enumerate(items):
			display_index = original_indices[index] + 1
			if not isinstance(item, dict) or not isinstance(item.get("ontime"), dict):
				raise RivaError(f"Riva 第 {display_index} 句回應格式無效。")
			meta = item["ontime"]
			verdict, reasons = meta.get("verdict"), meta.get("reasons")
			if verdict not in ("ok", "warn", "reject", "skipped") or not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
				raise RivaError(f"Riva 第 {display_index} 句判定資訊無效。")
			if verdict == "reject" or "input:truncated" in reasons:
				raise RivaError(f"第 {display_index} 句被拒絕或截斷：{reasons}")
			output = item.get("text")
			if not isinstance(output, str):
				raise RivaError(f"Riva 第 {display_index} 句未回傳有效譯文。")
			if verdict == "skipped":
				if texts[index].strip() and any(char.isalnum() for char in texts[index]):
					raise RivaError(f"Riva 第 {display_index} 句略過實質內容。")
				output = texts[index]
			if verdict == "warn":
				LOGGER.warning("Riva 第 %d 句警告：%s", display_index, reasons)
			outputs.append(output)
		return outputs

	def ensure_ready(self) -> None:
		state = self._probe_health()
		if state == "ready":
			return
		if state in ("failed", "malformed"):
			raise self._startup_error("健康檢查顯示 NMT 無法使用")
		process = None
		log_handle = None
		if state == "absent":
			parsed = urllib.parse.urlsplit(self.base_url)
			if parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
				raise RivaError(f"遠端 Riva 服務無法連線，不會在本機自動啟動：{self.base_url}", retryable=True)
			if not self.auto_start:
				raise RivaError(f"Riva 服務未啟動，且 ONTIME_AUTO_START=0：{self.base_url}", retryable=True)
			launcher = f"{self.repo_path}/start-relay.sh"
			if platform.system() != "Windows" and not Path(launcher).is_file():
				raise self._startup_error(f"找不到啟動程式 {launcher}")
			port = parsed.port or (443 if parsed.scheme == "https" else 80)
			if platform.system() == "Windows":
				argv = ["wsl.exe", "-d", self.wsl_distro, "-e", "bash", launcher, "--no-preload", "--port", str(port)]
				flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
				kwargs = {"creationflags": flags}
			else:
				argv = ["bash", launcher, "--no-preload", "--port", str(port)]
				kwargs = {"start_new_session": True}
			self.log_path.parent.mkdir(parents=True, exist_ok=True)
			log_handle = self.log_path.open("ab")
			try:
				process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log_handle, stderr=log_handle, **kwargs)
			except OSError as error:
				log_handle.close()
				raise self._startup_error(f"無法啟動 relay：{error}") from error
		deadline = time.monotonic() + self.start_timeout
		try:
			while time.monotonic() < deadline:
				if process is not None and process.poll() is not None:
					raise self._startup_error(f"relay 啟動程序提前結束，代碼 {process.returncode}")
				state = self._probe_health()
				if state == "ready":
					return
				if state in ("failed", "malformed"):
					raise self._startup_error("relay 已回應，但 NMT 健康狀態無效")
				time.sleep(0.25)
			raise self._startup_error("等待 NMT 就緒逾時")
		finally:
			if log_handle is not None:
				log_handle.close()

	def _probe_health(self):
		request = urllib.request.Request(self.base_url + "/health", method="GET")
		try:
			with self._opener.open(request, timeout=min(self.timeout, 5.0)) as response:
				raw = response.read()
		except urllib.error.HTTPError as error:
			error.close()
			self._last_health_detail = f"HTTP {error.code}"
			return "malformed"
		except urllib.error.URLError as error:
			self._last_health_detail = str(error)
			return "absent" if isinstance(error.reason, ConnectionRefusedError) else "malformed"
		except ConnectionRefusedError as error:
			self._last_health_detail = str(error)
			return "absent"
		except (http.client.HTTPException, socket.timeout, TimeoutError, OSError) as error:
			self._last_health_detail = str(error)
			return "malformed"
		try:
			body = json.loads(raw.decode("utf-8"))
		except (UnicodeDecodeError, json.JSONDecodeError):
			return "malformed"
		nmt = body.get("nmt") if isinstance(body, dict) else None
		if body.get("service") != "ontime-translator-relay" or not isinstance(nmt, dict) or not isinstance(nmt.get("loaded"), bool) or not isinstance(nmt.get("state"), str) or "error" not in nmt:
			return "malformed"
		if nmt["loaded"] is True and nmt["state"] == "ready" and not nmt["error"]:
			return "ready"
		if nmt["state"] in ("starting", "loading") and not nmt["error"]:
			return "starting"
		return "failed"

	def _startup_error(self, stage):
		tail = ""
		try:
			with self.log_path.open("rb") as handle:
				handle.seek(0, os.SEEK_END)
				handle.seek(max(0, handle.tell() - 4096))
				tail = handle.read().decode("utf-8", "replace")[-2000:]
		except OSError:
			pass
		return RivaError(f"Riva 啟動失敗：{stage}；網址 {self.base_url}；記錄 {self.log_path}；末尾：{tail}")
