import json
import os
import re
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
		body = type(self).health_body
		if body is None:
			body = {"service": "ontime-translator-relay", "nmt": {"loaded": True, "state": "ready", "error": None}}
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

	def test_ok_whitespace_translation_is_rejected_for_substantive_input(self):
		RelayHandler.responder = lambda payload: (200, {"translations": [accepted("   ")]})
		with self.assertRaisesRegex(RivaError, "未回傳有效譯文") as caught:
			self.client.translate_batch(["日本語"])
		self.assertFalse(caught.exception.retryable)

	def test_reassembled_whitespace_translation_is_rejected(self):
		with mock.patch.object(self.client, "_request", return_value=[" \t "]):
			with self.assertRaisesRegex(RivaError, "未回傳有效譯文"):
				self.client.translate_batch(["日本語"])

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
		text = "あ" * 130 + "。" + "い" * 130
		result = self.client.translate(text)
		segments = [part for request in RelayHandler.requests for part in request["text"]]
		self.assertEqual("".join(segments), text)
		self.assertEqual(result, "".join("譯：" + part for part in segments))
		self.assertTrue(all(len(part) <= 120 for part in segments))

	def test_long_input_prefers_sentence_boundaries(self):
		text = "あ" * 80 + "。" + "い" * 80 + "。"
		self.client.translate(text)
		self.assertEqual(RelayHandler.requests[0]["text"], ["あ" * 80 + "。", "い" * 80 + "。"])

	def test_long_input_keeps_glossary_markers_atomic(self):
		text = "あ" * 110 + "東京" + "い" * 20
		self.client.translate(text, {"東京": "東京"})
		segments = RelayHandler.requests[0]["text"]
		self.assertEqual(segments, ["あ" * 110, "__GLOSSARY_0__" + "い" * 20])
		self.assertTrue(all(len(part) <= 120 for part in segments))

	def test_each_segment_rejects_swapped_glossary_marker_ids(self):
		swaps = {"__GLOSSARY_0__": "__GLOSSARY_1__", "__GLOSSARY_1__": "__GLOSSARY_0__"}
		RelayHandler.responder = lambda payload: (
			200,
			{"translations": [accepted(re.sub(
				r"__GLOSSARY_[01]__",
				lambda match: swaps[match.group(0)],
				part,
			)) for part in payload["text"]]},
		)
		text = "あ" * 105 + "東京。" + "い" * 105 + "大阪"
		with self.assertRaisesRegex(RivaError, "術語標記完整性失敗"):
			self.client.translate(text, {"東京": "Tokyo", "大阪": "Osaka"})
		self.assertEqual(len(RelayHandler.requests), 1)
		self.assertEqual(len(RelayHandler.requests[0]["text"]), 2)

	def test_segments_still_batch_at_32_requests(self):
		text = "あ" * (120 * 33)
		result = self.client.translate(text)
		segments = [part for request in RelayHandler.requests for part in request["text"]]
		self.assertEqual([len(request["text"]) for request in RelayHandler.requests], [32, 1])
		self.assertEqual("".join(segments), text)
		self.assertEqual(result, "".join("譯：" + part for part in segments))

	def test_long_input_rejects_an_empty_segment_before_reassembly(self):
		RelayHandler.responder = lambda payload: (
			200,
			{"translations": [accepted("")] + [accepted("譯文") for _ in payload["text"][1:]]},
		)
		with self.assertRaisesRegex(RivaError, "未回傳有效譯文"):
			self.client.translate("あ" * 1201)

	def test_response_requires_guard_object(self):
		for ontime in (
			{"verdict": "ok", "reasons": []},
			{"verdict": "ok", "reasons": [], "guard": []},
		):
			with self.subTest(ontime=ontime):
				RelayHandler.responder = lambda payload, ontime=ontime: (
					200,
					{"translations": [{"text": "譯文", "ontime": ontime}]},
				)
				with self.assertRaisesRegex(RivaError, "判定資訊無效"):
					self.client.translate("文")

	def test_glossary_masks_longest_first_and_preserves_multiplicity(self):
		result = self.client.translate("東京都と東京、東京", {"東京都": "東京都", "東京": "東京"})
		self.assertEqual(result, "譯：東京都と東京、東京")
		masked = RelayHandler.requests[0]["text"][0]
		self.assertEqual(masked.count("__GLOSSARY_"), 3)

	def test_glossary_marker_collision_and_integrity_failures(self):
		literal = "__GLOSSARY_0__ と東京"
		self.assertEqual(self.client.translate(literal, {"東京": "Tokyo"}), "譯：__GLOSSARY_0__ とTokyo")
		for changed, message in (
			("", "未回傳有效譯文"),
			("__GLOSSARY_999__ __GLOSSARY_999__", "術語"),
		):
			with self.subTest(changed=changed):
				RelayHandler.responder = lambda payload, changed=changed: (
					200, {"translations": [accepted(changed)]},
				)
				with self.assertRaisesRegex(RivaError, message):
					self.client.translate("東京", {"東京": "Tokyo"})

	def test_adjacent_forged_markers_cannot_satisfy_integrity_by_substring(self):
		RelayHandler.responder = lambda payload: (
			200,
			{"translations": [accepted("__GLOSSARY_1__GLOSSARY_0__GLOSSARY_2__")]},
		)
		with self.assertRaisesRegex(RivaError, "術語標記完整性失敗"):
			self.client.translate("東京大阪京都", {"東京": "東京", "大阪": "大阪", "京都": "京都"})

	def test_very_long_literal_marker_uses_a_short_atomic_escape(self):
		literal = "__GLOSSARY_" + "9" * 5000 + "__"
		text = "前" + literal + "後"
		RelayHandler.responder = lambda payload: (
			200,
			{"translations": [accepted(part) for part in payload["text"]]},
		)
		self.assertEqual(self.client.translate(text), text)
		segments = [part for request in RelayHandler.requests for part in request["text"]]
		self.assertTrue(all(len(part) <= 120 for part in segments))
		self.assertNotIn(literal, "".join(segments))
		self.assertEqual("".join(segments).count("__GLOSSARY_0__"), 1)

	def test_literal_marker_is_not_recursively_matched_by_glossary_terms(self):
		literal = "__GLOSSARY_" + "8" * 5000 + "__"
		text = "前" + literal + "中" + literal + "後"
		RelayHandler.responder = lambda payload: (
			200,
			{"translations": [accepted(part) for part in payload["text"]]},
		)
		self.assertEqual(self.client.translate(text, {"GLOSSARY": "詞彙", "__": "雙底線"}), text)
		segments = [part for request in RelayHandler.requests for part in request["text"]]
		short_markers = re.findall(r"__GLOSSARY_\d+__", "".join(segments))
		self.assertEqual(short_markers, ["__GLOSSARY_0__", "__GLOSSARY_1__"])
		self.assertEqual(len(short_markers), len(set(short_markers)))
		self.assertTrue(all(len(part) <= 120 for part in segments))
		self.assertTrue(all(sum(marker in part for part in segments) == 1 for marker in short_markers))

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

	def test_non_object_health_is_malformed_and_never_starts_relay(self):
		for body in ([], ["ready"], "ready", 42):
			with self.subTest(body=body):
				RelayHandler.health_body = body
				self.assertEqual(self.client._probe_health(), "malformed")
				with mock.patch("ontime_riva.subprocess.Popen") as popen:
					with self.assertRaisesRegex(RivaError, "健康檢查"):
						self.client.ensure_ready()
				popen.assert_not_called()


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
