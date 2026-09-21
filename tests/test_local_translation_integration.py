import importlib
import json
import queue
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from ontime_riva import RivaError


class FakeSegment:
	def __init__(self, index):
		self.start = index + 0.25
		self.end = index + 0.75
		self.text = f" 日文{index + 1} "


def import_application():
	numpy = types.ModuleType("numpy")
	numpy.ndarray = object
	numpy.float32 = float
	torch = types.ModuleType("torch")
	torch.cuda = types.SimpleNamespace(is_available=lambda: False)
	scipy = types.ModuleType("scipy")
	scipy_signal = types.ModuleType("scipy.signal")
	scipy_signal.resample_poly = mock.Mock(name="resample_poly")
	pil = types.ModuleType("PIL")
	pil.Image = mock.Mock(name="Image")
	pil.ImageTk = mock.Mock(name="ImageTk")
	faster_whisper = types.ModuleType("faster_whisper")
	faster_whisper.WhisperModel = mock.Mock(name="WhisperModel")
	silero_vad = types.ModuleType("silero_vad")
	silero_vad.load_silero_vad = mock.Mock(name="load_silero_vad")
	silero_vad.VADIterator = mock.Mock(name="VADIterator")
	modules = {
		"numpy": numpy,
		"torch": torch,
		"scipy": scipy,
		"scipy.signal": scipy_signal,
		"cv2": types.ModuleType("cv2"),
		"PIL": pil,
		"faster_whisper": faster_whisper,
		"silero_vad": silero_vad,
		"pyaudiowpatch": types.ModuleType("pyaudiowpatch"),
		"sounddevice": types.ModuleType("sounddevice"),
	}
	sys.modules.update(modules)
	sys.modules.pop("live_caption_gemini", None)
	return importlib.import_module("live_caption_gemini")


class LocalTranslationIntegrationTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.app = import_application()

	def make_translator(self):
		with mock.patch.object(self.app, "OntimeRivaClient") as client_class:
			client_class.return_value.base_url = "http://127.0.0.1:8765"
			translator = self.app.Translator()
		return translator, client_class

	def main_startup_mocks(self, stack, *, cue_mode=False):
		"""Keep the real main() control flow while replacing platform boundaries."""
		device = {"name": "CABLE Output", "defaultSampleRate": 48000}
		window = types.SimpleNamespace(title="來源視窗", visible=True, width=800, height=600, _hWnd=7)
		probe = mock.Mock()
		probe.get_loopback_device_info_generator.return_value = [device]
		settings = types.ModuleType("startup_gui")
		settings.load_last_settings = mock.Mock(return_value={})
		settings.save_last_settings = mock.Mock()
		settings.index_of_name = mock.Mock(return_value=0)
		settings.run_startup_dialog = mock.Mock(return_value={
			"window_index": 0, "input_index": 1, "output_index": 0,
			"episode_title": "測試", "delay": 2.0,
		})
		windows = types.ModuleType("pygetwindow")
		windows.getAllWindows = mock.Mock(return_value=[window])
		process = types.ModuleType("win32process")
		process.GetWindowThreadProcessId = mock.Mock(return_value=(1, 42))
		psutil = types.ModuleType("psutil")
		psutil.Process = mock.Mock(return_value=types.SimpleNamespace(name=lambda: "browser.exe"))
		stack.enter_context(mock.patch.dict(sys.modules, {
			"startup_gui": settings, "pygetwindow": windows,
			"win32process": process, "psutil": psutil,
		}))
		stack.enter_context(mock.patch.object(self.app.tk, "Tk"))
		stack.enter_context(mock.patch("tkinter.messagebox.askyesno", return_value=cue_mode))
		stack.enter_context(mock.patch.object(self.app.pyaudio, "PyAudio", return_value=probe, create=True))
		stack.enter_context(mock.patch.object(self.app.sd, "query_hostapis", return_value=[], create=True))
		stack.enter_context(mock.patch.object(self.app.sd, "query_devices", return_value=[], create=True))
		stack.enter_context(mock.patch.object(self.app, "SOUND_VOLUME_VIEW_PATH", mock.Mock(exists=lambda: True)))
		stack.enter_context(mock.patch.object(self.app, "get_app_output_device", return_value="喇叭"))
		return stack.enter_context(mock.patch.object(self.app, "set_app_output_device", return_value=True))

	def test_constructor_checks_readiness_once_and_context_sends_only_current_sentence(self):
		translator, client_class = self.make_translator()
		translator.client.translate.return_value = "  請多指教。  "

		result = translator.translate_with_context(
			"田中です。", "よろしく。", {"田中": "田中先生"}
		)

		client_class.assert_called_once_with()
		translator.client.ensure_ready.assert_called_once_with()
		translator.client.translate.assert_called_once_with("よろしく。", {"田中": "田中先生"})
		self.assertEqual(result, "請多指教。")

	def test_worker_checks_translation_before_loading_whisper(self):
		events = []
		translator = mock.Mock()

		def make_translator():
			events.append("translator_ready")
			return translator

		def make_whisper(*_args, **_kwargs):
			events.append("whisper_loaded")
			return mock.Mock()

		with tempfile.TemporaryDirectory() as temp_dir, \
			 mock.patch.object(self.app, "Translator", side_effect=make_translator), \
			 mock.patch.object(self.app, "WhisperModel", side_effect=make_whisper), \
			 mock.patch.object(self.app, "load_glossary", return_value={}), \
			 mock.patch.object(self.app, "TRANSCRIPT_DIR", Path(temp_dir)):
			worker = self.app.Worker(queue.Queue(), queue.Queue(), queue.Queue())

		self.assertEqual(events, ["translator_ready", "whisper_loaded"])
		self.assertIs(worker.translator, translator)

	def test_worker_reuses_injected_translator_without_second_readiness_check(self):
		translator = mock.Mock()
		with tempfile.TemporaryDirectory() as temp_dir, \
			 mock.patch.object(self.app, "Translator") as translator_class, \
			 mock.patch.object(self.app, "WhisperModel"), \
			 mock.patch.object(self.app, "load_glossary", return_value={}), \
			 mock.patch.object(self.app, "TRANSCRIPT_DIR", Path(temp_dir)):
			worker = self.app.Worker(
				queue.Queue(), queue.Queue(), queue.Queue(), translator=translator
			)

		self.assertIs(worker.translator, translator)
		translator_class.assert_not_called()

	def test_main_riva_failure_cannot_reroute_source_audio(self):
		with ExitStack() as stack:
			reroute = self.main_startup_mocks(stack)
			stack.enter_context(mock.patch.object(self.app, "Translator", side_effect=RivaError("未就緒")))
			for name in ("FrameBuffer", "AudioRingBuffer", "PauseState", "ScreenCapture", "AudioCapture"):
				stack.enter_context(mock.patch.object(self.app, name))
			with self.assertRaisesRegex(RivaError, "未就緒"):
				self.app.main()

		reroute.assert_not_called()

	def test_main_preflight_translator_is_reused_after_audio_reroute(self):
		events = []
		translator = mock.Mock()
		with ExitStack() as stack:
			reroute = self.main_startup_mocks(stack)
			reroute.side_effect = lambda *_args: events.append("rerouted") or True
			stack.enter_context(mock.patch.object(
				self.app, "Translator", side_effect=lambda: events.append("ready") or translator
			))
			for name in ("FrameBuffer", "AudioRingBuffer", "PauseState", "ScreenCapture", "AudioCapture"):
				stack.enter_context(mock.patch.object(self.app, name))
			worker = stack.enter_context(mock.patch.object(self.app, "Worker"))
			worker.side_effect = lambda *_args, **_kwargs: events.append("worker") or mock.Mock()
			stack.enter_context(mock.patch.object(self.app, "DelayedAudioPlayer", side_effect=RuntimeError("stop")))
			with self.assertRaisesRegex(RuntimeError, "stop"):
				self.app.main()

		self.assertEqual(events, ["ready", "rerouted", "worker"])
		self.assertIs(worker.call_args.kwargs["translator"], translator)

	def test_main_cue_mode_does_not_preflight_translator(self):
		with ExitStack() as stack:
			self.main_startup_mocks(stack, cue_mode=True)
			stack.enter_context(mock.patch("tkinter.filedialog.askopenfilename", return_value="cues.json"))
			stack.enter_context(mock.patch.object(
				self.app, "load_cue_file", return_value={"cues": [{"start": 0, "ja": "日文", "zh": "中文"}]}
			))
			translator_class = stack.enter_context(mock.patch.object(self.app, "Translator"))
			stack.enter_context(mock.patch.object(self.app, "FrameBuffer", side_effect=RuntimeError("stop")))
			with self.assertRaisesRegex(RuntimeError, "stop"):
				self.app.main()

		translator_class.assert_not_called()

	def test_empty_inputs_make_no_translation_calls(self):
		translator, _ = self.make_translator()

		self.assertEqual(translator.translate("  "), "")
		self.assertEqual(translator.translate_with_context("前句", "\t"), "")
		self.assertEqual(translator.translate_batch([], {"田中": "田中先生"}), [])
		translator.client.translate.assert_not_called()
		translator.client.translate_batch.assert_not_called()

	def test_live_failure_preserves_source_without_retry_or_sleep(self):
		translator, _ = self.make_translator()
		translator.client.translate.side_effect = RivaError("連線失敗", retryable=True)

		with mock.patch.object(self.app.time, "sleep") as sleep:
			result = translator.translate("失敗原文")

		self.assertEqual(result, "（翻譯失敗，保留日文原文）失敗原文")
		translator.client.translate.assert_called_once_with("失敗原文", None)
		sleep.assert_not_called()

	def test_batch_translation_isolates_failed_chunk_and_retries_only_twice(self):
		translator, _ = self.make_translator()
		texts = [f"日文{index + 1}" for index in range(65)]
		first = [f"中文{index + 1}" for index in range(32)]
		translator.client.translate_batch.side_effect = [
			first,
			RivaError("暫時錯誤", retryable=True),
			RivaError("暫時錯誤", retryable=True),
			RivaError("暫時錯誤", retryable=True),
			["中文65"],
		]

		with mock.patch.object(self.app.time, "sleep") as sleep:
			result = translator.translate_batch(texts, {"日文": "日文"})

		self.assertEqual(result[:32], first)
		self.assertEqual(
			result[32:64],
			[f"（翻譯失敗，保留日文原文）{text}" for text in texts[32:64]],
		)
		self.assertEqual(result[64:], ["中文65"])
		self.assertEqual(translator.client.translate_batch.call_count, 5)
		for call in translator.client.translate_batch.call_args_list:
			self.assertLessEqual(len(call.args[0]), 32)
			self.assertEqual(call.args[1], {"日文": "日文"})
		self.assertEqual(sleep.call_args_list, [mock.call(20), mock.call(20)])

	def test_permanent_batch_failure_falls_back_immediately(self):
		translator, _ = self.make_translator()
		translator.client.translate_batch.side_effect = RivaError("數字守門拒絕")

		with mock.patch.object(self.app.time, "sleep") as sleep:
			result = translator.translate_batch(["三億円", "後續句"])

		self.assertEqual(result, [
			"（翻譯失敗，保留日文原文）三億円",
			"（翻譯失敗，保留日文原文）後續句",
		])
		translator.client.translate_batch.assert_called_once_with(["三億円", "後續句"], None)
		sleep.assert_not_called()

	def test_translator_constructs_riva_client_without_provider_arguments(self):
		with mock.patch.object(self.app, "OntimeRivaClient") as client_class:
			client_class.return_value.base_url = "mock://relay"
			translator = self.app.Translator()

		client_class.assert_called_once_with()
		translator.client.ensure_ready.assert_called_once_with()

	def test_offline_batches_preserve_cues_and_pair_each_translation(self):
		segments = [FakeSegment(index) for index in range(33)]
		fake_asr = mock.Mock()
		fake_asr.transcribe.return_value = (segments, None)
		translator = mock.Mock()
		batches = []

		def translate_batch(texts, glossary):
			batches.append((list(texts), glossary))
			return [text.replace("日文", "中文") for text in texts]

		translator.translate_batch.side_effect = translate_batch
		sys.modules.pop("transcribe_audio_file", None)
		offline = importlib.import_module("transcribe_audio_file")
		with tempfile.TemporaryDirectory() as temp_dir:
			audio_path = Path(temp_dir) / "episode.wav"
			audio_path.write_bytes(b"audio")
			with mock.patch.object(offline, "WhisperModel", return_value=fake_asr), \
				 mock.patch.object(offline, "Translator", return_value=translator), \
				 mock.patch.object(offline, "load_glossary", return_value={"田中": "田中先生"}):
				paths = offline.transcribe_audio_file(audio_path, "節目")
			data = json.loads(paths[0].read_text(encoding="utf-8"))

		self.assertEqual([len(batch[0]) for batch in batches], [32, 1])
		self.assertEqual([batch[1] for batch in batches], [{"田中": "田中先生"}] * 2)
		self.assertEqual(len(data["cues"]), 33)
		for index, cue in enumerate(data["cues"]):
			self.assertEqual(
				(cue["start"], cue["end"], cue["ja"], cue["zh"]),
				(index + 0.25, index + 0.75, f"日文{index + 1}", f"中文{index + 1}"),
			)

	def test_offline_checks_translation_before_constructing_or_running_whisper(self):
		events = []
		segment = FakeSegment(0)
		fake_asr = mock.Mock()
		fake_asr.transcribe.side_effect = lambda *_args, **_kwargs: (
			events.append("whisper_transcribed") or ([segment], None)
		)
		translator = mock.Mock()
		translator.translate_batch.return_value = ["中文1"]
		sys.modules.pop("transcribe_audio_file", None)
		offline = importlib.import_module("transcribe_audio_file")

		def make_translator():
			events.append("translator_ready")
			return translator

		def make_whisper(*_args, **_kwargs):
			events.append("whisper_loaded")
			return fake_asr

		with tempfile.TemporaryDirectory() as temp_dir:
			audio_path = Path(temp_dir) / "episode.wav"
			audio_path.write_bytes(b"audio")
			with mock.patch.object(offline, "Translator", side_effect=make_translator), \
				 mock.patch.object(offline, "WhisperModel", side_effect=make_whisper), \
				 mock.patch.object(offline, "load_glossary", return_value={}):
				offline.transcribe_audio_file(audio_path)

		self.assertEqual(events, ["translator_ready", "whisper_loaded", "whisper_transcribed"])

	def test_full_audio_reconstruction_batches_raw_lines_and_preserves_pairs(self):
		segments = [FakeSegment(index) for index in range(33)]
		asr = mock.Mock()
		asr.transcribe.return_value = (segments, None)
		translator = mock.Mock()
		translator.translate_batch.side_effect = [
			[f"中文{index + 1}" for index in range(32)],
			["中文33"],
		]
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "show_audio.wav"
			wav_path.write_bytes(b"audio")
			with mock.patch.object(Path, "unlink") as unlink:
				output = self.app.rebuild_transcript_from_full_audio(
					wav_path, asr, translator, {"田中": "田中先生"}, "節目"
				)
				markdown = output.read_text(encoding="utf-8")

		self.assertEqual(
			[len(call.args[0]) for call in translator.translate_batch.call_args_list],
			[32, 1],
		)
		for call in translator.translate_batch.call_args_list:
			self.assertEqual(call.args[1], {"田中": "田中先生"})
		self.assertIn("日文1\n中文1", markdown)
		self.assertIn("日文33\n中文33", markdown)
		self.assertNotIn("潤稿", markdown)
		unlink.assert_called_once_with()

	def test_full_audio_fallback_remains_visible_and_cleanup_is_preserved(self):
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(0)], None)
		translator = mock.Mock()
		translator.translate_batch.return_value = ["（翻譯失敗，保留日文原文）日文1"]
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "fallback_audio.wav"
			wav_path.write_bytes(b"audio")
			with mock.patch.object(Path, "unlink") as unlink:
				output = self.app.rebuild_transcript_from_full_audio(
					wav_path, asr, translator, None
				)
				markdown = output.read_text(encoding="utf-8")

		self.assertIn("日文1\n（翻譯失敗，保留日文原文）日文1", markdown)
		unlink.assert_called_once_with()

	def test_pretranslated_cues_do_not_initialize_translation(self):
		caption_queue = queue.Queue()
		transcript_queue = queue.Queue()
		cues = [{"start": 1.25, "end": 2.5, "ja": "日本語", "zh": "中文"}]
		with mock.patch.object(self.app, "Translator") as translator_class:
			self.app.push_cues_to_queues(cues, caption_queue, transcript_queue, 100.0)

		expected = (101.25 + self.app.DISPLAY_DELAY_SEC, "日本語", "中文")
		self.assertEqual(caption_queue.get_nowait(), expected)
		self.assertEqual(transcript_queue.get_nowait(), expected)
		translator_class.assert_not_called()


if __name__ == "__main__":
	unittest.main()
