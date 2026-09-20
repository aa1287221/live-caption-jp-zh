import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from ontime_riva import OntimeRivaClient, RivaError


class RelayHandler(BaseHTTPRequestHandler):
	requests = []
	responder = None
	health_body = None

	def log_message(self, format, *args):
		pass

	def do_GET(self):
		body = type(self).health_body or {"service": "ontime-translator-relay", "nmt": {"loaded": True, "state": "ready", "error": None}}
		self._reply(200, body)

	def do_POST(self):
		length = int(self.headers.get("Content-Length", "0"))
		payload = json.loads(self.rfile.read(length).decode("utf-8"))
		type(self).requests.append(payload)
		status, body = type(self).responder(payload)
		if isinstance(body, bytes):
			self.send_response(status)
			self.end_headers()
			self.wfile.write(body)
		else:
			self._reply(status, body)

	def _reply(self, status, body):
		raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json")
		self.send_header("Content-Length", str(len(raw)))
		self.end_headers()
		self.wfile.write(raw)


def accepted(text):
	return {"text": text, "ontime": {"verdict": "ok", "reasons": [], "guard": {}}}


class RivaHttpTests(unittest.TestCase):
	def setUp(self):
		RelayHandler.requests = []
		RelayHandler.health_body = None
		RelayHandler.responder = lambda payload: (
			200,
			{"translations": [accepted("譯：" + text) for text in payload["text"]]},
		)
		self.server = ThreadingHTTPServer(("127.0.0.1", 0), RelayHandler)
		self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
		self.thread.start()
		self.addCleanup(self.server.server_close)
		self.addCleanup(self.server.shutdown)
		self.client = OntimeRivaClient(f"http://127.0.0.1:{self.server.server_port}", 2)

	def test_translate_uses_measured_contract(self):
		RelayHandler.responder = lambda payload: (200, {"translations": [accepted("你好。")]})
		self.assertEqual(self.client.translate("こんにちは。"), "你好。")
		self.assertEqual(RelayHandler.requests[0], {
			"text": ["こんにちは。"], "source_lang": "JA", "target_lang": "zh-TW",
		})

	def test_empty_input_bypasses_http_and_batch_groups_at_32(self):
		self.assertEqual(self.client.translate(" \n"), " \n")
		result = self.client.translate_batch([str(index) for index in range(33)])
		self.assertEqual(len(result), 33)
		self.assertEqual([len(call["text"]) for call in RelayHandler.requests], [32, 1])

	def test_response_protocol_failures_are_permanent(self):
		fixtures = [
			b"not-json",
			{},
			{"translations": []},
			{"translations": [{"text": "", "ontime": {"verdict": "ok", "reasons": [], "guard": {}}}]},
			{"translations": [{"text": "x", "ontime": {"verdict": "reject", "reasons": ["guard"], "guard": {}}}]},
			{"translations": [{"text": "x", "ontime": {"verdict": "warn", "reasons": ["input:truncated"], "guard": {}}}]},
		]
		for fixture in fixtures:
			with self.subTest(fixture=fixture):
				RelayHandler.responder = lambda payload, fixture=fixture: (200, fixture)
				with self.assertRaises(RivaError) as caught:
					self.client.translate("文")
				self.assertFalse(caught.exception.retryable)

	def test_warning_is_accepted_and_http_status_retryability_is_classified(self):
		RelayHandler.responder = lambda payload: (200, {"translations": [{
			"text": "警告譯文", "ontime": {"verdict": "warn", "reasons": ["style"], "guard": {}},
		}]})
		with self.assertLogs("ontime_riva", level="WARNING"):
			self.assertEqual(self.client.translate("文"), "警告譯文")
		for status, retryable in ((422, False), (429, True), (503, True)):
			with self.subTest(status=status):
				RelayHandler.responder = lambda payload, status=status: (status, {"message": "failed"})
				before = len(RelayHandler.requests)
				with self.assertRaises(RivaError) as caught:
					self.client.translate("文")
				self.assertEqual(caught.exception.retryable, retryable)
				self.assertEqual(len(RelayHandler.requests), before + 1)

	def test_long_input_splits_without_losing_order(self):
		text = "あ" * 700 + "。" + "い" * 700
		result = self.client.translate(text)
		self.assertEqual(result, "".join("譯：" + part for part in RelayHandler.requests[0]["text"]))
		self.assertTrue(all(len(part) <= 1200 for part in RelayHandler.requests[0]["text"]))

	def test_glossary_masks_longest_first_and_preserves_multiplicity(self):
		result = self.client.translate("東京都と東京、東京", {"東京都": "東京都", "東京": "東京"})
		self.assertEqual(result, "譯：東京都と東京、東京")
		masked = RelayHandler.requests[0]["text"][0]
		self.assertEqual(masked.count("__GLOSSARY_"), 3)

	def test_glossary_marker_collision_and_integrity_failures(self):
		literal = "__GLOSSARY_0__ と東京"
		self.assertEqual(self.client.translate(literal, {"東京": "Tokyo"}), "譯：__GLOSSARY_0__ とTokyo")
		for changed in ("", "__GLOSSARY_999__ __GLOSSARY_999__"):
			with self.subTest(changed=changed):
				RelayHandler.responder = lambda payload, changed=changed: (
					200, {"translations": [accepted(changed)]},
				)
				with self.assertRaisesRegex(RivaError, "術語"):
					self.client.translate("東京", {"東京": "Tokyo"})

	def test_skipped_punctuation_preserves_source_and_restore_is_one_pass(self):
		RelayHandler.responder = lambda payload: (200, {"translations": [{
			"text": "", "ontime": {"verdict": "skipped", "reasons": [], "guard": {}},
		}]})
		self.assertEqual(self.client.translate("！？。"), "！？。")
		self.assertEqual(
			self.client._restore("__GLOSSARY_0__", {"__GLOSSARY_0__": "__GLOSSARY_1__"}),
			"__GLOSSARY_1__",
		)

	def test_health_requires_expected_service_identity(self):
		RelayHandler.health_body = {"nmt": {"loaded": True, "state": "ready", "error": None}}
		self.assertEqual(self.client._probe_health(), "malformed")
		RelayHandler.health_body = {"service": "ontime-translator-relay", "nmt": {"loaded": True, "state": "ready", "error": None}}
		self.assertEqual(self.client._probe_health(), "ready")


class RivaLifecycleTests(unittest.TestCase):
	def setUp(self):
		self.environment = mock.patch.dict(os.environ, {}, clear=True)
		self.environment.start()
		self.addCleanup(self.environment.stop)

	def test_ready_service_is_reused_without_spawn(self):
		client = OntimeRivaClient()
		with mock.patch.object(client, "_probe_health", return_value="ready"), mock.patch("ontime_riva.subprocess.Popen") as popen:
			client.ensure_ready()
		popen.assert_not_called()

	def test_existing_starting_service_waits_without_spawn(self):
		client = OntimeRivaClient()
		with mock.patch.object(client, "_probe_health", side_effect=["starting", "ready"]), mock.patch("ontime_riva.time.sleep"), mock.patch("ontime_riva.subprocess.Popen") as popen:
			client.ensure_ready()
		popen.assert_not_called()

	def test_failed_or_malformed_service_does_not_spawn(self):
		for state in ("failed", "malformed"):
			client = OntimeRivaClient()
			with self.subTest(state=state), mock.patch.object(client, "_probe_health", return_value=state), mock.patch("ontime_riva.subprocess.Popen") as popen:
				with self.assertRaises(RivaError):
					client.ensure_ready()
				popen.assert_not_called()

	def test_absent_windows_service_uses_exact_direct_argv(self):
		with mock.patch.dict(os.environ, {"ONTIME_REPO_PATH": "/project/Ontime-Translator"}), mock.patch("ontime_riva.platform.system", return_value="Windows"):
			client = OntimeRivaClient()
			process = mock.Mock(poll=mock.Mock(return_value=None))
			with mock.patch.object(client, "_probe_health", side_effect=["absent", "ready"]), mock.patch("ontime_riva.Path.is_file", side_effect=AssertionError("Windows cannot inspect WSL paths")), mock.patch("ontime_riva.subprocess.Popen", return_value=process) as popen, mock.patch("ontime_riva.time.sleep"):
				client.ensure_ready()
		argv = popen.call_args.args[0]
		self.assertEqual(argv, ["wsl.exe", "-d", "Ubuntu-26.04", "-e", "bash", "/project/Ontime-Translator/start-relay.sh", "--no-preload", "--port", "8765"])
		self.assertIs(popen.call_args.kwargs["stdout"], popen.call_args.kwargs["stderr"])

	def test_linux_custom_port_uses_bash_and_remote_never_launches(self):
		with tempfile.TemporaryDirectory(prefix="riva repo ") as directory:
			launcher = Path(directory) / "start-relay.sh"
			launcher.touch()
			with mock.patch.dict(os.environ, {"ONTIME_REPO_PATH": directory}), mock.patch("ontime_riva.platform.system", return_value="Linux"):
				client = OntimeRivaClient("http://127.0.0.1:9123")
				with mock.patch.object(client, "_probe_health", side_effect=["absent", "ready"]), mock.patch("ontime_riva.subprocess.Popen", return_value=mock.Mock(poll=lambda: None)) as popen, mock.patch("ontime_riva.time.sleep"):
					client.ensure_ready()
			self.assertEqual(popen.call_args.args[0], ["bash", str(launcher), "--no-preload", "--port", "9123"])
		remote = OntimeRivaClient("http://example.test:8765")
		with mock.patch.object(remote, "_probe_health", return_value="absent"), mock.patch("ontime_riva.subprocess.Popen") as popen:
			with self.assertRaisesRegex(RivaError, "遠端"):
				remote.ensure_ready()
			popen.assert_not_called()

	def test_configuration_is_validated_at_construction(self):
		for url in ("ftp://host", "http://", "http://host/path", "http://host#x"):
			with self.subTest(url=url), self.assertRaises(RivaError):
				OntimeRivaClient(url)
		for timeout in (0, -1, float("inf"), "bad"):
			with self.subTest(timeout=timeout), self.assertRaises(RivaError):
				OntimeRivaClient(timeout=timeout)
		for path in ("relative/path", "", "C:\\Ontime-Translator"):
			with self.subTest(path=path), mock.patch.dict(os.environ, {"ONTIME_REPO_PATH": path}):
				with self.assertRaises(RivaError):
					OntimeRivaClient()

	def test_linux_missing_launcher_and_exited_process_surface_startup_error(self):
		with tempfile.TemporaryDirectory() as directory:
			client = OntimeRivaClient()
			client.repo_path = directory
			client.log_path = Path(directory) / "relay.log"
			with mock.patch.object(client, "_probe_health", return_value="absent"), mock.patch("ontime_riva.subprocess.Popen") as popen:
				with self.assertRaisesRegex(RivaError, "找不到啟動程式"):
					client.ensure_ready()
			popen.assert_not_called()
			launcher = Path(directory) / "start-relay.sh"
			launcher.touch()
			process = mock.Mock()
			process.poll.return_value = 7
			process.returncode = 7
			with mock.patch.object(client, "_probe_health", return_value="absent"), mock.patch("ontime_riva.subprocess.Popen", return_value=process):
				with self.assertRaisesRegex(RivaError, "提前結束，代碼 7"):
					client.ensure_ready()

	def test_startup_deadline_has_bounded_log_tail_when_service_stays_absent(self):
		with tempfile.TemporaryDirectory() as directory:
			launcher = Path(directory) / "start-relay.sh"
			launcher.touch()
			client = OntimeRivaClient()
			client.repo_path = directory
			client.log_path = Path(directory) / "relay.log"
			client.start_timeout = 1
			client.log_path.write_bytes(b"x" * 5000)
			process = mock.Mock()
			process.poll.return_value = None
			with mock.patch.object(client, "_probe_health", return_value="absent"), mock.patch("ontime_riva.subprocess.Popen", return_value=process), mock.patch("ontime_riva.time.monotonic", side_effect=[0, 0, 1]), mock.patch("ontime_riva.time.sleep"):
				with self.assertRaisesRegex(RivaError, "等待 NMT 就緒逾時") as caught:
					client.ensure_ready()
			self.assertIn(str(client.log_path), str(caught.exception))
			self.assertLess(len(str(caught.exception)), 2600)


if __name__ == "__main__":
	unittest.main()
