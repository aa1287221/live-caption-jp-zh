import importlib
import json
import os
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
		with mock.patch.object(self.app, "LocalChatClient") as client_class:
			client_class.return_value.model = "test-model"
			client_class.return_value.base_url = "http://127.0.0.1:8080/v1"
			translator = self.app.Translator()
		return translator

	def test_context_translation_uses_local_client_and_preserves_prompts(self):
		translator = self.make_translator()
		translator.client.chat.return_value = "  請多指教。  "

		result = translator.translate_with_context(
			"田中です。", "よろしく。", {"田中": "田中先生"}
		)

		self.assertEqual(result, "請多指教。")
		system, user = translator.client.chat.call_args.args
		self.assertIn(self.app.glossary_hint({"田中": "田中先生"}), system)
		self.assertIn("田中です。", user)
		self.assertIn("よろしく。", user)
		self.assertIn("不用翻譯", user)

	def test_empty_inputs_make_no_http_calls(self):
		translator = self.make_translator()

		self.assertEqual(translator.translate("  "), "")
		self.assertEqual(translator.translate_with_context("前句", "\t"), "")
		translator.client.chat.assert_not_called()

	def test_chat_collapses_duplicates_and_handles_local_error(self):
		translator = self.make_translator()
		translator.client.chat.return_value = "完成。完成。完成。完成。完成。"
		self.assertEqual(translator.translate("完了"), "完成。完成。")

		translator.client.chat.side_effect = LocalLLMError("連線失敗")
		self.assertEqual(translator.translate("失敗"), "")

	def test_constructs_without_google_module_or_key(self):
		with mock.patch.dict(os.environ, {}, clear=True):
			translator = self.make_translator()
		self.assertEqual(translator.client.model, "test-model")

	def test_offline_numbered_batches_preserve_cues_and_missing_number(self):
		segments = [FakeSegment(index) for index in range(41)]
		first = "\n".join(
			f"{number}. 中文{number}" for number in range(1, 41) if number != 17
		)
		responses = [first, "1. 中文41"]
		chat_calls = []

		def chat(_self, system_prompt, user_prompt):
			chat_calls.append((system_prompt, user_prompt))
			return responses.pop(0)

		fake_asr = mock.Mock()
		fake_asr.transcribe.return_value = (segments, None)
		with tempfile.TemporaryDirectory() as temp_dir:
			audio_path = Path(temp_dir) / "episode.wav"
			audio_path.write_bytes(b"audio")
			with mock.patch("faster_whisper.WhisperModel", return_value=fake_asr), \
				 mock.patch.object(self.app.Translator, "_chat", autospec=True, side_effect=chat), \
				 mock.patch.object(self.app, "load_glossary", return_value={"田中": "田中先生"}), \
				 mock.patch("transcribe_audio_file.WhisperModel", return_value=fake_asr), \
				 mock.patch("transcribe_audio_file.Translator", self.app.Translator), \
				 mock.patch("transcribe_audio_file.load_glossary", return_value={"田中": "田中先生"}), \
				 mock.patch("transcribe_audio_file.time.sleep"):
				offline = importlib.import_module("transcribe_audio_file")
				paths = offline.transcribe_audio_file(audio_path, "節目")
			data = json.loads(paths[0].read_text(encoding="utf-8"))

		self.assertEqual(len(data["cues"]), 41)
		for index, cue in enumerate(data["cues"]):
			self.assertEqual((cue["start"], cue["end"], cue["ja"]),
				(index + 0.25, index + 0.75, f"日文{index + 1}"))
		self.assertEqual(data["cues"][16]["zh"], "")
		self.assertEqual(data["cues"][17]["zh"], "中文18")
		self.assertEqual(data["cues"][40]["zh"], "中文41")
		self.assertEqual(len(chat_calls), 2)
		self.assertIn("田中", chat_calls[1][0])
		self.assertIn("日文39", chat_calls[1][1])
		self.assertIn("日文40", chat_calls[1][1])

	def test_full_audio_reconstruction_batches_and_preserves_bilingual_output(self):
		segments = [FakeSegment(index) for index in range(61)]
		asr = mock.Mock()
		asr.transcribe.return_value = (segments, None)
		translator = self.make_translator()
		translator.client.chat.side_effect = ["日文一\n中文一", "日文六十一\n中文六十一"]
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "show_audio.wav"
			wav_path.write_bytes(b"audio")
			with mock.patch.object(Path, "unlink") as unlink, mock.patch.object(self.app.time, "sleep"):
				output = self.app.rebuild_transcript_from_full_audio(
					wav_path, asr, translator, {"田中": "田中先生"}, "節目"
				)
				markdown = output.read_text(encoding="utf-8")

		self.assertIn("日文一\n中文一", markdown)
		self.assertIn("日文六十一\n中文六十一", markdown)
		self.assertEqual(translator.client.chat.call_count, 2)
		first_system, _ = translator.client.chat.call_args_list[0].args
		_, second_user = translator.client.chat.call_args_list[1].args
		self.assertIn("田中", first_system)
		self.assertIn("編輯", first_system)
		self.assertIn("日文59", second_user)
		self.assertIn("日文60", second_user)
		unlink.assert_called_once_with()

	def test_reconstruction_retries_and_all_failure_keeps_audio(self):
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(0)], None)
		translator = self.make_translator()
		translator.client.chat.side_effect = [LocalLLMError("暫時失敗"), "日文一\n中文一"]
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "retry_audio.wav"
			wav_path.write_bytes(b"audio")
			with mock.patch.object(self.app.time, "sleep") as sleep:
				self.assertIsNotNone(self.app.rebuild_transcript_from_full_audio(
					wav_path, asr, translator, None
				))
			self.assertEqual(translator.client.chat.call_count, 2)
			sleep.assert_called_once_with(20)

			failed_path = Path(temp_dir) / "failed_audio.wav"
			failed_path.write_bytes(b"audio")
			failed = self.make_translator()
			failed.client.chat.side_effect = LocalLLMError("持續失敗")
			with mock.patch.object(self.app.time, "sleep"), mock.patch.object(Path, "unlink") as unlink:
				self.assertIsNone(self.app.rebuild_transcript_from_full_audio(
					failed_path, asr, failed, None
				))
			self.assertTrue(failed_path.exists())
			self.assertEqual(failed.client.chat.call_count, 3)
			unlink.assert_not_called()

	def test_push_cues_preserves_queue_tuple_contract(self):
		caption_queue = queue.Queue()
		transcript_queue = queue.Queue()
		cues = [{"start": 1.25, "end": 2.5, "ja": "日本語", "zh": "中文"}]
		self.app.push_cues_to_queues(cues, caption_queue, transcript_queue, 100.0)
		expected = (101.25 + self.app.DISPLAY_DELAY_SEC, "日本語", "中文")
		self.assertEqual(caption_queue.get_nowait(), expected)
		self.assertEqual(transcript_queue.get_nowait(), expected)


if __name__ == "__main__":
	unittest.main()
