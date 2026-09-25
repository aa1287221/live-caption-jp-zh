import importlib
import json
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from local_llm import LocalLLMError


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
	sys.modules.update({
		"numpy": numpy, "torch": torch, "scipy": scipy,
		"scipy.signal": scipy_signal, "cv2": types.ModuleType("cv2"),
		"PIL": pil, "faster_whisper": faster_whisper,
		"silero_vad": silero_vad, "pyaudiowpatch": types.ModuleType("pyaudiowpatch"),
		"sounddevice": types.ModuleType("sounddevice"),
	})
	sys.modules.pop("live_caption_gemini", None)
	return importlib.import_module("live_caption_gemini")


class LocalTranslationIntegrationTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.app = import_application()

	def make_translator(self):
		with mock.patch.object(self.app, "LocalLLMClient") as client_class:
			client_class.return_value.base_url = "http://127.0.0.1:8766"
			translator = self.app.Translator()
		return translator, client_class

	def test_constructor_uses_local_client_without_google_or_ontime_provider(self):
		translator, client_class = self.make_translator()
		client_class.assert_called_once_with()
		self.assertEqual(translator.client.base_url, "http://127.0.0.1:8766")

	def test_context_translation_preserves_prompt_boundary_and_glossary(self):
		translator, _ = self.make_translator()
		translator.client.chat.return_value = "  請多指教。  "

		result = translator.translate_with_context("田中です。", "よろしく。", {"田中": "田中先生"})

		self.assertEqual(result, "請多指教。")
		system_prompt, user_prompt = translator.client.chat.call_args.args
		self.assertIn("田中", system_prompt)
		self.assertIn("前一句話", user_prompt)
		self.assertIn("よろしく。", user_prompt)

	def test_live_failure_preserves_source_without_retry_or_sleep(self):
		translator, _ = self.make_translator()
		translator.client.chat.side_effect = LocalLLMError("格式錯誤")

		with mock.patch.object(self.app.time, "sleep") as sleep:
			result = translator.translate("失敗原文")

		self.assertEqual(result, "（翻譯失敗，保留日文原文）失敗原文")
		translator.client.chat.assert_called_once()
		sleep.assert_not_called()

	def test_empty_inputs_make_no_translation_calls(self):
		translator, _ = self.make_translator()

		self.assertEqual(translator.translate("  "), "")
		self.assertEqual(translator.translate_with_context("前句", "\t"), "")
		self.assertEqual(translator.translate_batch([], {"田中": "田中先生"}), [])
		translator.client.chat.assert_not_called()

	def test_worker_reuses_injected_translator_without_constructing_another(self):
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

	def test_batch_retries_only_retryable_local_errors(self):
		translator, _ = self.make_translator()
		translator.client.chat.side_effect = [
			LocalLLMError("暫時錯誤", retryable=True),
			LocalLLMError("暫時錯誤", retryable=True),
			"成功翻譯",
		]

		with mock.patch.object(self.app.time, "sleep") as sleep:
			result = translator.translate_batch(["日文"], None)

		self.assertEqual(result, ["成功翻譯"])
		self.assertEqual(translator.client.chat.call_count, 3)
		self.assertEqual(sleep.call_args_list, [mock.call(20), mock.call(20)])

	def test_batch_permanent_failure_falls_back_immediately(self):
		translator, _ = self.make_translator()
		translator.client.chat.side_effect = LocalLLMError("輸出格式錯誤")

		with mock.patch.object(self.app.time, "sleep") as sleep:
			result = translator.translate_batch(["三億円", "後續句"])

		self.assertEqual(result, [
			"（翻譯失敗，保留日文原文）三億円",
			"（翻譯失敗，保留日文原文）後續句",
		])
		self.assertEqual(translator.client.chat.call_count, 2)
		sleep.assert_not_called()

	def test_offline_batches_preserve_cues_and_pair_each_translation(self):
		segments = [FakeSegment(index) for index in range(33)]
		fake_asr = mock.Mock()
		fake_asr.transcribe.return_value = (segments, None)
		translator = mock.Mock()
		translator.translate_batch.side_effect = lambda texts, glossary: [text.replace("日文", "中文") for text in texts]
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

		self.assertEqual([len(call.args[0]) for call in translator.translate_batch.call_args_list], [32, 1])
		self.assertEqual(len(data["cues"]), 33)
		self.assertEqual(data["cues"][0]["zh"], "中文1")

	def test_full_audio_reconstruction_keeps_translation_pairs_and_cleanup(self):
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(0)], None)
		translator = mock.Mock()
		translator.translate_batch.return_value = ["中文1"]
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "show_audio.wav"
			wav_path.write_bytes(b"audio")
			with mock.patch.object(Path, "unlink") as unlink:
				output = self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, None)
				markdown = output.read_text(encoding="utf-8")

		self.assertIn("日文1\n中文1", markdown)
		unlink.assert_called_once_with()

	def test_full_audio_reconstruction_batches_33_cues_in_order(self):
		segments = [FakeSegment(index) for index in range(33)]
		asr = mock.Mock()
		asr.transcribe.return_value = (segments, None)
		translator = mock.Mock()
		translator.translate_batch.side_effect = [
			[f"中文{index + 1}" for index in range(32)], ["中文33"]
		]
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "show_audio.wav"
			wav_path.write_bytes(b"audio")
			with mock.patch.object(Path, "unlink"):
				output = self.app.rebuild_transcript_from_full_audio(
					wav_path, asr, translator, {"田中": "田中先生"}, "節目"
				)
				markdown = output.read_text(encoding="utf-8")

		self.assertEqual([len(call.args[0]) for call in translator.translate_batch.call_args_list], [32, 1])
		self.assertIn("日文1\n中文1", markdown)
		self.assertIn("日文33\n中文33", markdown)
		self.assertNotIn("潤稿", markdown)

	def test_full_audio_fallback_remains_visible_and_cleanup_is_preserved(self):
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(0)], None)
		translator = mock.Mock()
		translator.translate_batch.return_value = ["（翻譯失敗，保留日文原文）日文1"]
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "fallback_audio.wav"
			wav_path.write_bytes(b"audio")
			with mock.patch.object(Path, "unlink") as unlink:
				output = self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, None)
				markdown = output.read_text(encoding="utf-8")

		self.assertIn("日文1\n（翻譯失敗，保留日文原文）日文1", markdown)
		unlink.assert_called_once_with()

	def test_pretranslated_cues_preserve_queue_timing_and_tuple_shape(self):
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
