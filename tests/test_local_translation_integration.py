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


if __name__ == "__main__":
	unittest.main()
