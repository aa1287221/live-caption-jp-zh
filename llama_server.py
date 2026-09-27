"""Start, wait for and stop a llama-server this app spawned itself.

Standard-library only. An already-running server the app did not start (Ontime, Ollama,
WSL, a manually launched llama-server) is used as-is and never touched here: this module
only manages a child process it created.

Configuration resolves env var -> ``local_llm_server.json`` (program folder) -> default,
per key (see ``LlamaServerConfig.resolve``). ``ensure_llama_server`` is the entry point:
it probes the base URL, and only spawns when nothing answers and auto-start is configured
(a model path is set). On Windows the spawned child is placed in a Job Object with
``KILL_ON_JOB_CLOSE`` so a crashed app cannot leave an orphan holding VRAM.
"""

import atexit
import ctypes
import ipaddress
import json
import math
import os
import shlex
import subprocess
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

from local_llm import LocalLLMClient, LocalLLMError


PROGRAM_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROGRAM_DIR / "local_llm_server.json"
LOG_PATH = PROGRAM_DIR / "logs" / "llama-server.log"

DEFAULT_SERVER_EXE = "tools/llama/llama-server.exe"
DEFAULT_SERVER_ARGS = "-c 8192 -np 1 -ngl 99 --jinja --reasoning off"
DEFAULT_STARTUP_WAIT = 300.0

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class LlamaServerError(RuntimeError):
	"""Config, validation or spawn-time failure raised directly by this module."""


# ---------- server_args: accept a shell-like string or a JSON/native list ----------

def parse_server_args(value) -> list:
	"""Turn ``server_args`` (string or list, from env/JSON) into argv tokens.

	A string starting with ``[`` is tried as a JSON list first; otherwise it is split
	shell-style. ``posix=False`` keeps Windows path backslashes intact (posix mode treats
	backslash as an escape character, which would corrupt paths like ``C:\\models\\x.gguf``).
	"""
	if isinstance(value, list):
		return [str(item) for item in value]
	if not isinstance(value, str):
		raise LlamaServerError(f"server_args 設定無效：必須是字串或字串陣列，收到 {value!r}。")
	stripped = value.strip()
	if stripped.startswith("["):
		try:
			parsed = json.loads(stripped)
		except json.JSONDecodeError:
			parsed = None
		if isinstance(parsed, list):
			return [str(item) for item in parsed]
	try:
		tokens = shlex.split(stripped, posix=False)
	except ValueError as error:
		raise LlamaServerError(f"server_args 設定無效（引號沒有配對）：{value!r}") from error
	return [_strip_quotes(token) for token in tokens]


def _strip_quotes(token: str) -> str:
	if len(token) >= 2 and token[0] == token[-1] and token[0] in "'\"":
		return token[1:-1]
	return token


# ---------- config resolution ----------

def _load_json_config(path: Path) -> dict:
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
	except (OSError, json.JSONDecodeError):
		return {}
	return data if isinstance(data, dict) else {}


def _resolve_path_value(value, program_dir: Path) -> Path | None:
	if value is None:
		return None
	path = Path(str(value)).expanduser()
	return path if path.is_absolute() else (program_dir / path)


def _resolve_startup_wait(value) -> float:
	try:
		numeric = float(value)
	except (TypeError, ValueError) as error:
		raise LlamaServerError(f"startup_wait 設定無效：請指定正數秒數，收到 {value!r}。") from error
	if not math.isfinite(numeric) or numeric <= 0:
		raise LlamaServerError(f"startup_wait 設定無效：請指定正數秒數，收到 {value!r}。")
	return numeric


def _parse_host_port(base_url: str) -> tuple:
	parsed = urllib.parse.urlsplit(base_url)
	host = parsed.hostname or "127.0.0.1"
	port = parsed.port or (443 if parsed.scheme == "https" else 80)
	return host, port


def _is_loopback_host(hostname: str) -> bool:
	if hostname.lower() == "localhost":
		return True
	try:
		return ipaddress.ip_address(hostname).is_loopback
	except ValueError:
		return False


@dataclass
class LlamaServerConfig:
	"""Resolved settings for one auto-start attempt."""

	server_exe: Path | None
	model_path: Path | None
	server_args: list = field(default_factory=list)
	startup_wait: float = DEFAULT_STARTUP_WAIT
	base_url: str = LocalLLMClient.DEFAULT_BASE_URL
	host: str = "127.0.0.1"
	port: int = 8766

	@property
	def auto_start_enabled(self) -> bool:
		"""No model path -> the user has not opted into auto-start."""
		return self.model_path is not None

	@property
	def is_loopback(self) -> bool:
		return _is_loopback_host(self.host)

	@classmethod
	def resolve(cls, *, base_url=None, environ=None, config_path=None, program_dir=None) -> "LlamaServerConfig":
		"""env var -> local_llm_server.json -> default, per key.

		``environ``/``config_path``/``program_dir`` are injectable so callers (and tests)
		never have to mutate the real process environment or touch the real program folder.
		"""
		environ = os.environ if environ is None else environ
		program_dir = PROGRAM_DIR if program_dir is None else Path(program_dir)
		resolved_config_path = Path(config_path) if config_path is not None else (program_dir / "local_llm_server.json")
		json_config = _load_json_config(resolved_config_path)

		def value_for(env_name, json_key, default):
			env_value = environ.get(env_name)
			if env_value is not None and str(env_value).strip() != "":
				return env_value
			json_value = json_config.get(json_key)
			if json_value not in (None, ""):
				return json_value
			return default

		resolved_base_url = base_url if base_url is not None else environ.get("LOCAL_LLM_BASE_URL", LocalLLMClient.DEFAULT_BASE_URL)
		host, port = _parse_host_port(resolved_base_url)
		return cls(
			server_exe=_resolve_path_value(value_for("LOCAL_LLM_SERVER_EXE", "server_exe", DEFAULT_SERVER_EXE), program_dir),
			model_path=_resolve_path_value(value_for("LOCAL_LLM_MODEL_PATH", "model_path", None), program_dir),
			server_args=parse_server_args(value_for("LOCAL_LLM_SERVER_ARGS", "server_args", DEFAULT_SERVER_ARGS)),
			startup_wait=_resolve_startup_wait(value_for("LOCAL_LLM_SPAWN_WAIT", "startup_wait", DEFAULT_STARTUP_WAIT)),
			base_url=resolved_base_url, host=host, port=port,
		)


# ---------- process management ----------

class ManagedLlamaServer:
	"""Owns one llama-server child process this app spawned: start, wait, stop."""

	def __init__(self, config: LlamaServerConfig, *, command_prefix=(), log_path=None, log=print) -> None:
		self.config = config
		self.command_prefix = list(command_prefix)
		self.log_path = Path(log_path) if log_path is not None else LOG_PATH
		self.log = log
		self.process: subprocess.Popen | None = None
		self._log_file = None
		self._job_handle = None
		self._stopped = False

	@property
	def command(self) -> list:
		config = self.config
		return [
			*self.command_prefix, str(config.server_exe), *config.server_args,
			"--host", config.host, "--port", str(config.port), "-m", str(config.model_path),
		]

	def start(self) -> None:
		self.log_path.parent.mkdir(parents=True, exist_ok=True)
		self._log_file = open(self.log_path, "a", encoding="utf-8", errors="replace")
		self._log_file.write(f"\n----- 啟動：{' '.join(self.command)} -----\n")
		self._log_file.flush()
		self.log(f"自動啟動本機模型服務：{' '.join(self.command)}")
		creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
		self.process = subprocess.Popen(
			self.command, stdin=subprocess.DEVNULL, stdout=self._log_file, stderr=subprocess.STDOUT,
			cwd=str(PROGRAM_DIR), creationflags=creationflags,
		)
		atexit.register(self.stop)
		self._assign_job_object()

	def wait_ready(self, client, *, poll_interval: float = 2.0, on_wait=None, clock=time.monotonic, sleep=time.sleep) -> None:
		"""Wait for ``client``'s health check, failing fast if the child already exited."""

		def guarded_on_wait(error):
			self._raise_if_exited()
			if on_wait is not None:
				on_wait(error)

		client.ensure_ready(self.config.startup_wait, poll_interval=poll_interval, on_wait=guarded_on_wait, clock=clock, sleep=sleep)
		self._raise_if_exited()

	def stop(self) -> None:
		"""Terminate the child if still running; safe to call more than once."""
		if self._stopped:
			return
		self._stopped = True
		try:
			atexit.unregister(self.stop)
		except ValueError:
			pass
		if self.process is not None and self.process.poll() is None:
			try:
				self.process.terminate()
				self.process.wait(timeout=5)
			except subprocess.TimeoutExpired:
				try:
					self.process.kill()
					self.process.wait(timeout=5)
				except subprocess.TimeoutExpired as error:
					self.log(f"提醒：llama-server 子行程 kill() 後仍未在時限內結束：{error}")
			except OSError as error:
				self.log(f"提醒：終止 llama-server 子行程時發生錯誤：{error}")
		if self._log_file is not None:
			try:
				self._log_file.close()
			except OSError:
				pass
			self._log_file = None
		if self._job_handle is not None:
			try:
				ctypes.windll.kernel32.CloseHandle(self._job_handle)
			except OSError:
				pass
			self._job_handle = None

	# ---------- internals ----------

	def _raise_if_exited(self) -> None:
		if self.process is None:
			return
		code = self.process.poll()
		if code is None:
			return
		raise LlamaServerError(
			f"本機模型服務啟動後提早結束（exit code {code}）：\n{self._log_tail()}\n"
			"請確認 server_exe／model_path／server_args 設定正確，或直接在終端機執行同一個指令排查。"
		)

	def _log_tail(self, lines: int = 20) -> str:
		try:
			if self._log_file is not None:
				self._log_file.flush()
			text = self.log_path.read_text(encoding="utf-8", errors="replace")
		except OSError:
			return f"（無法讀取 log：{self.log_path}）"
		tail = text.splitlines()[-lines:]
		return "\n".join(tail) if tail else "（log 檔是空的）"

	def _assign_job_object(self) -> None:
		"""Best-effort: a failure here is a warning, never fatal (no-op off Windows)."""
		if os.name != "nt" or self.process is None:
			return
		try:
			self._job_handle = _assign_to_windows_job(self.process.pid)
		except OSError as error:
			self.log(f"提醒：無法把 llama-server 子行程放進 Job Object（{error}），程式異常結束時可能不會自動終止子行程。")
			self._job_handle = None


def _assign_to_windows_job(pid: int) -> int:
	"""Create a Job Object with KILL_ON_JOB_CLOSE and put ``pid`` in it; return the job handle."""

	class _BasicLimits(ctypes.Structure):
		_fields_ = [
			("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
			("LimitFlags", ctypes.c_uint32), ("MinimumWorkingSetSize", ctypes.c_size_t),
			("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", ctypes.c_uint32),
			("Affinity", ctypes.c_size_t), ("PriorityClass", ctypes.c_uint32), ("SchedulingClass", ctypes.c_uint32),
		]

	class _IoCounters(ctypes.Structure):
		_fields_ = [(name, ctypes.c_uint64) for name in (
			"ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
			"ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
		)]

	class _ExtendedLimits(ctypes.Structure):
		_fields_ = [
			("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
			("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
			("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t),
		]

	kernel32 = ctypes.windll.kernel32
	# HANDLE is pointer-sized; ctypes' default restype (c_int, 32-bit) can truncate/misread
	# a handle value on Win64 without this -- explicit is correct, not just "happens to work".
	kernel32.CreateJobObjectW.restype = ctypes.c_void_p
	kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
	kernel32.OpenProcess.restype = ctypes.c_void_p
	kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
	kernel32.SetInformationJobObject.restype = ctypes.c_int
	kernel32.SetInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32)
	kernel32.AssignProcessToJobObject.restype = ctypes.c_int
	kernel32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
	kernel32.CloseHandle.restype = ctypes.c_int
	kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)

	job = kernel32.CreateJobObjectW(None, None)
	if not job:
		raise ctypes.WinError()
	info = _ExtendedLimits()
	info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if not kernel32.SetInformationJobObject(job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)):
		error = ctypes.WinError()
		kernel32.CloseHandle(job)
		raise error
	process_handle = kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid)
	if not process_handle:
		error = ctypes.WinError()
		kernel32.CloseHandle(job)
		raise error
	try:
		if not kernel32.AssignProcessToJobObject(job, process_handle):
			raise ctypes.WinError()
	finally:
		kernel32.CloseHandle(process_handle)
	return job


# ---------- entry point ----------

def ensure_llama_server(
	client, *, config: LlamaServerConfig | None = None, log=print,
	command_prefix=(), log_path=None, poll_interval: float = 2.0,
) -> ManagedLlamaServer | None:
	"""Spawn llama-server iff nothing answers ``client``'s base URL and auto-start is set up.

	Returns the spawned, already-ready ``ManagedLlamaServer``, or ``None`` when an existing
	server answered (never spawned, never stopped) or auto-start is not configured (today's
	behaviour: the caller's own readiness wait raises its usual, unchanged error).
	"""
	try:
		client.ensure_ready(wait_seconds=0)
		return None
	except LocalLLMError as error:
		if error.kind == "not_ready":
			# Something is already there and loading (Ontime/Ollama/user-started) -- use it.
			return None
		if not error.unreachable:
			raise
	# Only resolve config once we know we might actually need to spawn: a malformed
	# LOCAL_LLM_SPAWN_WAIT/server_args/JSON file must never block an already-reachable
	# server (spec rule 2) just because ensure_llama_server() also resolves config.
	config = config if config is not None else LlamaServerConfig.resolve(base_url=client.base_url)
	if not config.auto_start_enabled:
		return None
	if not config.is_loopback:
		raise LlamaServerError(
			f"本機模型服務網址「{config.base_url}」不是本機位址：自動啟動只支援 loopback"
			"（127.0.0.1／localhost），請自行啟動遠端服務，或把 LOCAL_LLM_BASE_URL 改回本機網址。"
		)
	if config.server_exe is None or not config.server_exe.is_file():
		raise LlamaServerError(
			f"找不到 llama-server 執行檔：{config.server_exe}\n"
			"請先執行 setup_local_llm.py 下載，或設定 LOCAL_LLM_SERVER_EXE 指向正確路徑。"
		)
	if config.model_path is None or not config.model_path.is_file():
		raise LlamaServerError(
			f"找不到模型檔：{config.model_path}\n"
			"請先執行 setup_local_llm.py 下載，或設定 LOCAL_LLM_MODEL_PATH 指向正確的 GGUF 檔案。"
		)
	server = ManagedLlamaServer(config, command_prefix=command_prefix, log_path=log_path, log=log)
	notices = []

	def on_wait(error):
		if not notices:
			log(f"等待本機模型服務就緒（最多 {config.startup_wait:g} 秒）：{error}")
		notices.append(error)

	try:
		# start() can fail (bad exe, unwritable logs/, ...): keep it in the try too, or a
		# raw OSError escapes uncaught and the log handle open() already did stays leaked.
		server.start()
		server.wait_ready(client, poll_interval=poll_interval, on_wait=on_wait)
	except OSError as error:
		server.stop()
		raise LlamaServerError(f"啟動 llama-server 失敗：{config.server_exe}（{error}）") from error
	except Exception:
		# Release the log handle / Job Object now; the caller never gets a handle to this
		# instance to clean up itself since we raise instead of returning it.
		server.stop()
		raise
	return server
