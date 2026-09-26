"""Path-matrix tests for llama_server.py (spec: docs/superpowers/specs/2026-09-26-managed-llama-server-design.md).

Rows 1-4 and 9 (no spawn) use a duck-typed fake client and spy on ManagedLlamaServer so
they run with no sockets and no subprocess at all. Rows 5-7 (a real spawn) run the fake
server in tests/fake_llama_server.py as a real child process, controlled entirely through
env vars it inherits so the built argv (``server.command``) stays exactly what production
builds. Row 8 (config precedence) and row 10 (server_args parsing) are pure-function tests.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from local_llm import LocalLLMClient, LocalLLMError
from llama_server import (
	DEFAULT_SERVER_ARGS, DEFAULT_SERVER_EXE, LlamaServerConfig, LlamaServerError,
	ManagedLlamaServer, ensure_llama_server, parse_server_args,
)


FAKE_SERVER = Path(__file__).parent / "fake_llama_server.py"
_UNSET = object()  # distinguishes "use the default" from "explicitly None" in make_config
# Connecting to a closed loopback port on this Windows box burns the full client timeout
# instead of failing fast (the known pre-existing 'timeout' != 'connection' quirk) -- keep
# it short so "unreachable" rows do not slow the suite down.
PROBE_TIMEOUT = 0.5


def free_port() -> int:
	with socket.socket() as probe:
		probe.bind(("127.0.0.1", 0))
		return probe.getsockname()[1]


class FakeProbeClient:
	"""Duck-typed stand-in for LocalLLMClient's ``ensure_ready``/``base_url`` contract."""

	def __init__(self, base_url: str, outcome: LocalLLMError | None) -> None:
		self.base_url = base_url
		self._outcome = outcome

	def ensure_ready(self, wait_seconds: float = 0.0, **_kwargs) -> None:
		if self._outcome is not None:
			raise self._outcome


class LlamaServerTestCase(unittest.TestCase):
	def setUp(self) -> None:
		self.tmp = tempfile.TemporaryDirectory()
		self.addCleanup(self.tmp.cleanup)
		self.tmp_path = Path(self.tmp.name)
		self.log_path = self.tmp_path / "logs" / "llama-server.log"
		# Only existence is checked by ensure_llama_server; contents never matter.
		self.model_path = self.tmp_path / "model.gguf"
		self.model_path.write_bytes(b"fake-gguf")
		self.other_exe = self.tmp_path / "other.exe"
		self.other_exe.write_bytes(b"fake-exe")

	def make_config(self, port, *, model_path=_UNSET, server_exe=_UNSET, server_args=(), startup_wait=10.0, host="127.0.0.1") -> LlamaServerConfig:
		return LlamaServerConfig(
			server_exe=FAKE_SERVER if server_exe is _UNSET else server_exe,
			model_path=self.model_path if model_path is _UNSET else model_path,
			server_args=list(server_args), startup_wait=startup_wait,
			base_url=f"http://{host}:{port}", host=host, port=port,
		)

	def make_client(self, config: LlamaServerConfig, **kwargs) -> LocalLLMClient:
		kwargs.setdefault("timeout", PROBE_TIMEOUT)
		return LocalLLMClient(base_url=config.base_url, **kwargs)


# ---------- rows 1-4, 9: decision logic, no real spawn ----------

class NoSpawnDecisionTests(LlamaServerTestCase):
	def test_row1_already_reachable_does_not_spawn_or_get_stopped(self):
		config = self.make_config(free_port())  # port need not even be bound
		client = FakeProbeClient(config.base_url, outcome=None)
		with mock.patch("llama_server.ManagedLlamaServer") as spy:
			result = ensure_llama_server(client, config=config)
		self.assertIsNone(result)
		spy.assert_not_called()

	def test_loading_503_is_treated_as_already_running_not_unreachable(self):
		config = self.make_config(free_port())
		client = FakeProbeClient(config.base_url, LocalLLMError("loading", kind="not_ready", retryable=True))
		with mock.patch("llama_server.ManagedLlamaServer") as spy:
			result = ensure_llama_server(client, config=config)
		self.assertIsNone(result)
		spy.assert_not_called()

	def test_row2_unreachable_no_model_configured_does_not_spawn(self):
		config = self.make_config(free_port(), model_path=None)
		client = FakeProbeClient(config.base_url, LocalLLMError("連線失敗", kind="connection", retryable=True))
		with mock.patch("llama_server.ManagedLlamaServer") as spy:
			result = ensure_llama_server(client, config=config)
		self.assertIsNone(result)
		spy.assert_not_called()

	def test_row3_unreachable_exe_missing_names_the_exe_path_and_does_not_spawn(self):
		missing_exe = self.tmp_path / "does-not-exist.exe"
		config = self.make_config(free_port(), server_exe=missing_exe)
		client = FakeProbeClient(config.base_url, LocalLLMError("連線失敗", kind="connection", retryable=True))
		with mock.patch("llama_server.ManagedLlamaServer") as spy:
			with self.assertRaises(LlamaServerError) as caught:
				ensure_llama_server(client, config=config)
		self.assertIn(str(missing_exe), str(caught.exception))
		spy.assert_not_called()

	def test_row4_unreachable_model_missing_names_the_model_path_and_does_not_spawn(self):
		missing_model = self.tmp_path / "does-not-exist.gguf"
		config = self.make_config(free_port(), server_exe=self.other_exe, model_path=missing_model)
		client = FakeProbeClient(config.base_url, LocalLLMError("逾時", kind="timeout", retryable=True))
		with mock.patch("llama_server.ManagedLlamaServer") as spy:
			with self.assertRaises(LlamaServerError) as caught:
				ensure_llama_server(client, config=config)
		self.assertIn(str(missing_model), str(caught.exception))
		spy.assert_not_called()

	def test_row9_non_loopback_base_url_is_refused_without_spawning(self):
		config = self.make_config(free_port(), host="198.51.100.5")  # TEST-NET-2, non-routable
		client = FakeProbeClient(config.base_url, LocalLLMError("連線失敗", kind="connection", retryable=True))
		with mock.patch("llama_server.ManagedLlamaServer") as spy:
			with self.assertRaisesRegex(LlamaServerError, "本機位址|loopback"):
				ensure_llama_server(client, config=config)
		spy.assert_not_called()

	def test_unexpected_error_kind_propagates_without_spawn_decision(self):
		config = self.make_config(free_port())
		client = FakeProbeClient(config.base_url, LocalLLMError("設定錯誤", kind="config"))
		with mock.patch("llama_server.ManagedLlamaServer") as spy:
			with self.assertRaises(LocalLLMError):
				ensure_llama_server(client, config=config)
		spy.assert_not_called()

	def test_h6_reachable_server_is_used_even_if_config_would_fail_to_resolve(self):
		"""Spec rule 2: a health probe success must win regardless of config validity.

		config=None (the production default) forces ensure_llama_server to resolve its own
		LlamaServerConfig -- but only if it decides the server is unreachable. A malformed
		LOCAL_LLM_SPAWN_WAIT must never be allowed to block use of an already-running server.
		"""
		client = FakeProbeClient("http://127.0.0.1:8766", outcome=None)  # already reachable
		with mock.patch.dict(os.environ, {"LOCAL_LLM_SPAWN_WAIT": "abc"}), \
				mock.patch("llama_server.LlamaServerConfig.resolve", side_effect=AssertionError("resolve must not run when reachable")) as resolve_spy, \
				mock.patch("llama_server.ManagedLlamaServer") as spawn_spy:
			result = ensure_llama_server(client, config=None)
		self.assertIsNone(result)
		resolve_spy.assert_not_called()
		spawn_spy.assert_not_called()


# ---------- rows 5-7: a real fake-server child process ----------

class RealSpawnTests(LlamaServerTestCase):
	def ensure(self, config, **kwargs):
		client = self.make_client(config, timeout=2)
		kwargs.setdefault("command_prefix", (sys.executable,))
		kwargs.setdefault("log_path", self.log_path)
		kwargs.setdefault("log", lambda *_: None)
		kwargs.setdefault("poll_interval", 0.1)
		server = ensure_llama_server(client, config=config, **kwargs)
		if isinstance(server, ManagedLlamaServer):
			self.addCleanup(server.stop)
		return server, client

	def test_row5_spawn_after_loading_becomes_ready_with_correct_argv(self):
		port = free_port()
		config = self.make_config(port, server_args=["--extra-flag", "value"], startup_wait=30)
		with mock.patch.dict(os.environ, {"FAKE_HEALTH_503_COUNT": "2"}):
			server, client = self.ensure(config)
		self.assertIsInstance(server, ManagedLlamaServer)
		client.ensure_ready(wait_seconds=0)  # already healthy; raises if not
		self.assertEqual(
			server.command,
			[sys.executable, str(FAKE_SERVER), "--extra-flag", "value", "--host", "127.0.0.1", "--port", str(port), "-m", str(self.model_path)],
		)

	def test_row6_child_exit_before_ready_fails_fast_with_exit_code_and_log_tail(self):
		port = free_port()
		config = self.make_config(port, startup_wait=120.0)  # a large budget the failure must beat
		with mock.patch.dict(os.environ, {"FAKE_EXIT_CODE": "7", "FAKE_EXIT_MESSAGE": "設定檔壞掉了"}):
			started = time.monotonic()
			with self.assertRaises(LlamaServerError) as caught:
				self.ensure(config)
			elapsed = time.monotonic() - started
		self.assertLess(elapsed, 10.0, "should fail long before the 120s budget")
		message = str(caught.exception)
		self.assertIn("exit code 7", message)
		self.assertIn("設定檔壞掉了", message)

	def test_row7_stop_terminates_child_and_is_idempotent(self):
		port = free_port()
		config = self.make_config(port, startup_wait=30)
		server, _client = self.ensure(config)
		self.assertIsInstance(server, ManagedLlamaServer)
		self.assertIsNone(server.process.poll())
		server.stop()
		self.assertIsNotNone(server.process.poll())
		server.stop()  # idempotent: no error, no hang

	def test_m1_wait_message_is_logged_once(self):
		port = free_port()
		config = self.make_config(port, startup_wait=30)
		logs = []
		with mock.patch.dict(os.environ, {"FAKE_HEALTH_503_COUNT": "3"}):
			server, _client = self.ensure(config, log=logs.append)
		wait_lines = [line for line in logs if "等待本機模型服務就緒" in line]
		self.assertEqual(len(wait_lines), 1, logs)
		self.assertIn("最多 30", wait_lines[0])


class SpawnFailureTests(LlamaServerTestCase):
	def test_h5_invalid_exe_is_wrapped_as_llamaservererror_naming_the_path_and_closes_the_log(self):
		bad_exe = self.tmp_path / "not-really-an-exe.exe"
		bad_exe.write_text("this is not a valid Win32 executable", encoding="utf-8")
		config = self.make_config(free_port(), server_exe=bad_exe)
		client = FakeProbeClient(config.base_url, LocalLLMError("連線失敗", kind="connection", retryable=True))

		created = []
		real_cls = ManagedLlamaServer

		def spy_factory(*args, **kwargs):
			instance = real_cls(*args, **kwargs)
			created.append(instance)
			return instance

		with mock.patch("llama_server.ManagedLlamaServer", side_effect=spy_factory):
			with self.assertRaises(LlamaServerError) as caught:
				ensure_llama_server(client, config=config, log_path=self.log_path, log=lambda *_: None)
		self.assertIn(str(bad_exe), str(caught.exception))
		self.assertEqual(len(created), 1)
		self.assertIsNone(created[0]._log_file, "log handle must be closed, not leaked, on a failed spawn")
		self.assertIsNone(created[0].process)

	def test_item7_stop_survives_a_wait_timeout_even_after_kill(self):
		config = self.make_config(free_port())
		server = ManagedLlamaServer(config, log=lambda *_: None)
		fake_process = mock.Mock()
		fake_process.poll.return_value = None  # still "running" every time it's checked
		fake_process.wait.side_effect = subprocess.TimeoutExpired(cmd="llama-server", timeout=5)
		server.process = fake_process
		server.stop()  # must not raise, even though kill()'s own wait() also times out
		fake_process.terminate.assert_called_once()
		fake_process.kill.assert_called_once()
		self.assertEqual(fake_process.wait.call_count, 2)


# ---------- row 8: config precedence, pure ----------

class ConfigResolutionTests(unittest.TestCase):
	def setUp(self) -> None:
		self.tmp = tempfile.TemporaryDirectory()
		self.addCleanup(self.tmp.cleanup)
		self.program_dir = Path(self.tmp.name)
		self.missing_config = self.program_dir / "missing.json"

	def test_row8_default_when_nothing_is_set_and_relative_paths_resolve_to_program_dir(self):
		config = LlamaServerConfig.resolve(environ={}, config_path=self.missing_config, program_dir=self.program_dir)
		self.assertEqual(config.server_exe, self.program_dir / DEFAULT_SERVER_EXE)
		self.assertIsNone(config.model_path)
		self.assertEqual(config.server_args, parse_server_args(DEFAULT_SERVER_ARGS))
		self.assertEqual(config.startup_wait, 300.0)
		self.assertEqual((config.host, config.port), ("127.0.0.1", 8766))

	def test_row8_json_wins_over_default(self):
		config_path = self.program_dir / "local_llm_server.json"
		config_path.write_text(json.dumps({
			"server_exe": "json-exe.exe", "model_path": "json-model.gguf",
			"server_args": ["--from-json"], "startup_wait": 111,
		}), encoding="utf-8")
		config = LlamaServerConfig.resolve(environ={}, config_path=config_path, program_dir=self.program_dir)
		self.assertEqual(config.server_exe, self.program_dir / "json-exe.exe")
		self.assertEqual(config.model_path, self.program_dir / "json-model.gguf")
		self.assertEqual(config.server_args, ["--from-json"])
		self.assertEqual(config.startup_wait, 111.0)

	def test_row8_env_wins_over_json(self):
		config_path = self.program_dir / "local_llm_server.json"
		config_path.write_text(json.dumps({
			"server_exe": "json-exe.exe", "model_path": "json-model.gguf",
			"server_args": "--from-json", "startup_wait": 111,
		}), encoding="utf-8")
		environ = {
			"LOCAL_LLM_SERVER_EXE": "env-exe.exe", "LOCAL_LLM_MODEL_PATH": "env-model.gguf",
			"LOCAL_LLM_SERVER_ARGS": "--from-env", "LOCAL_LLM_SPAWN_WAIT": "222",
		}
		config = LlamaServerConfig.resolve(environ=environ, config_path=config_path, program_dir=self.program_dir)
		self.assertEqual(config.server_exe, self.program_dir / "env-exe.exe")
		self.assertEqual(config.model_path, self.program_dir / "env-model.gguf")
		self.assertEqual(config.server_args, ["--from-env"])
		self.assertEqual(config.startup_wait, 222.0)

	def test_row8_absolute_override_is_kept_absolute(self):
		absolute = self.program_dir / "elsewhere" / "llama-server.exe"
		environ = {"LOCAL_LLM_SERVER_EXE": str(absolute)}
		config = LlamaServerConfig.resolve(environ=environ, config_path=self.missing_config, program_dir=self.program_dir)
		self.assertEqual(config.server_exe, absolute)

	def test_row8_base_url_env_overrides_default_and_derives_host_port(self):
		environ = {"LOCAL_LLM_BASE_URL": "http://127.0.0.1:9001"}
		config = LlamaServerConfig.resolve(environ=environ, config_path=self.missing_config, program_dir=self.program_dir)
		self.assertEqual((config.host, config.port, config.base_url), ("127.0.0.1", 9001, "http://127.0.0.1:9001"))

	def test_invalid_startup_wait_is_a_clear_config_error(self):
		with self.assertRaisesRegex(LlamaServerError, "startup_wait"):
			LlamaServerConfig.resolve(environ={"LOCAL_LLM_SPAWN_WAIT": "not-a-number"}, config_path=self.missing_config, program_dir=self.program_dir)


# ---------- row 10: server_args string vs list ----------

class ServerArgsParsingTests(unittest.TestCase):
	def test_row10_string_and_list_produce_the_same_argv(self):
		as_string = parse_server_args("-c 8192 -np 1 --jinja")
		as_list = parse_server_args(["-c", "8192", "-np", "1", "--jinja"])
		self.assertEqual(as_string, as_list)
		self.assertEqual(as_string, ["-c", "8192", "-np", "1", "--jinja"])

	def test_json_array_string_is_accepted_like_a_native_list(self):
		self.assertEqual(parse_server_args('["-c", "8192"]'), parse_server_args(["-c", "8192"]))

	def test_quoted_windows_path_with_spaces_survives_string_split(self):
		self.assertEqual(
			parse_server_args('--some-path "C:\\Program Files\\thing" --flag'),
			["--some-path", "C:\\Program Files\\thing", "--flag"],
		)

	def test_non_string_non_list_is_a_config_error(self):
		with self.assertRaises(LlamaServerError):
			parse_server_args(123)


if __name__ == "__main__":
	unittest.main()
