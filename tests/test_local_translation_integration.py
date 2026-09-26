"""Integration tests for the translation backends inside the real application modules.

Windows, audio, GPU and GUI modules are replaced in ``sys.modules`` before import; the
translation code under test (prompts, batching, retries, output files) is the real code.
"""

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

from local_llm import MODE_INSTRUCT, MODE_RIVA, LocalLLMError
from local_translation import LocalTranslationEngine


class FakeSegment:
	def __init__(self, index, text=None):
		self.start = index + 0.25
		self.end = index + 0.75
		self.text = f" {text if text is not None else f'日文{index + 1}です。'} "


class ScriptedClient:
	def __init__(self, handler, mode=MODE_INSTRUCT):
		self.handler = handler
		self.mode = mode
		self.server_info = {}
		self.model = None
		self.base_url = "http://127.0.0.1:8766"
		self.timeout = 120.0
		self.max_tokens = 256
		self.calls = []

	def ensure_ready(self, *_, **__):
		pass

	def chat(self, system_prompt, user_prompt, *, max_tokens=None, timeout=None, temperature=None):
		self.calls.append((system_prompt, user_prompt))
		reply = self.handler(system_prompt, user_prompt)
		if isinstance(reply, BaseException):
			raise reply
		return reply


def install_fake_gemini(reply):
	"""Install a google.genai stand-in; returns the list of generate_content calls."""
	calls = []

	class GenerateContentConfig:
		def __init__(self, **kwargs):
			self.kwargs = kwargs

	class Models:
		def generate_content(self, model, contents, config):
			calls.append({"model": model, "contents": contents, **config.kwargs})
			result = reply(contents, len(calls))
			if isinstance(result, BaseException):
				raise result
			return types.SimpleNamespace(text=result)

		def list(self):
			return []

	class Client:
		def __init__(self, api_key):
			self.api_key = api_key
			self.models = Models()

	google = types.ModuleType("google")
	genai = types.ModuleType("google.genai")
	genai_types = types.ModuleType("google.genai.types")
	genai_errors = types.ModuleType("google.genai.errors")
	genai.Client = Client
	genai_types.GenerateContentConfig = GenerateContentConfig
	genai.types, genai.errors, google.genai = genai_types, genai_errors, genai
	sys.modules.update({"google": google, "google.genai": genai, "google.genai.types": genai_types, "google.genai.errors": genai_errors})
	return calls


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
	tkinter = mock.MagicMock(name="tkinter")
	sys.modules.update({
		"numpy": numpy, "torch": torch, "scipy": scipy,
		"scipy.signal": scipy_signal, "cv2": types.ModuleType("cv2"),
		"PIL": pil, "faster_whisper": faster_whisper,
		"silero_vad": silero_vad, "pyaudiowpatch": types.ModuleType("pyaudiowpatch"),
		"sounddevice": types.ModuleType("sounddevice"),
		"tkinter": tkinter, "tkinter.scrolledtext": tkinter.scrolledtext,
	})
	for name in ("live_caption_gemini", "transcribe_audio_file"):
		sys.modules.pop(name, None)
	return importlib.import_module("live_caption_gemini")


MASTER_LIVE_SYSTEM_PROMPT = (
	"你是專業的日文-繁體中文口譯，正在幫忙即時翻譯一個日文廣播/網路節目的口語對話。"
	"內容常有語助詞、停頓、話講到一半重講、省略主詞這類真人講話的狀況，"
	"請翻成自然通順、口語化的繁體中文，像是真人在說話，不要翻得死板生硬、也不要逐字直譯。"
	"只要輸出翻譯結果本身，不要加任何說明、引號、備註、拼音或原文。"
)


class TranslationBackendTestCase(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.app = import_application()
		cls.offline = importlib.import_module("transcribe_audio_file")

	def setUp(self):
		self.environment = mock.patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}, clear=True)
		self.environment.start()
		self.addCleanup(self.environment.stop)
		self.app.genai = self.app.genai_types = self.app.genai_errors = None
		self.sleeps = []
		sleeper = mock.patch.object(self.app.time, "sleep", side_effect=self.sleeps.append)
		sleeper.start()
		self.addCleanup(sleeper.stop)
		printer = mock.patch("builtins.print")
		self.printed = printer.start()
		self.addCleanup(printer.stop)

	def printed_text(self):
		return "\n".join(" ".join(str(arg) for arg in call.args) for call in self.printed.call_args_list)

	def gemini_translator(self, reply):
		calls = install_fake_gemini(reply)
		return self.app.Translator("gemini"), calls

	def local_translator(self, handler, mode=MODE_INSTRUCT):
		warmups = ("請翻譯這一句：\nこんにちは。", "Translate this into Traditional Chinese: こんにちは。")
		client = ScriptedClient(lambda system, user: "你好。" if user in warmups else handler(system, user), mode=mode)

		def factory(**kwargs):
			return LocalTranslationEngine(
				client, live_timeout=15, batch_lines=20, startup_wait=0,
				sleep=self.sleeps.append, log=lambda *_: None, **kwargs,
			)

		with mock.patch.object(self.app, "LocalTranslationEngine", factory):
			translator = self.app.Translator("local")
		client.calls.clear()
		return translator, client

	def run_reconstruction(self, translator, lines):
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(i, text) for i, text in enumerate(lines)], None)
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		wav_path = Path(temp_dir.name) / "transcript_20260926_010203_audio.wav"
		wav_path.write_bytes(b"RIFF")
		out_path = self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, {"田中": "田中"}, "節目")
		return out_path, wav_path

	def run_offline(self, translator, lines):
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(i, text) for i, text in enumerate(lines)], None)
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		audio_path = Path(temp_dir.name) / "episode.wav"
		audio_path.write_bytes(b"RIFF")
		with mock.patch.object(self.offline, "WhisperModel", return_value=asr), \
				mock.patch.object(self.offline, "load_glossary", return_value={"田中": "田中"}), \
				mock.patch.object(self.offline.time, "sleep", side_effect=self.sleeps.append):
			cues_path, transcript_path = self.offline.transcribe_audio_file(audio_path, "節目", translator=translator)
		return json.loads(cues_path.read_text(encoding="utf-8"))["cues"], transcript_path.read_text(encoding="utf-8")


class GeminiPathTests(TranslationBackendTestCase):
	def test_default_backend_is_gemini_with_master_prompts_and_config(self):
		calls = install_fake_gemini(lambda contents, n: "  請多指教。  ")
		translator = self.app.Translator()
		self.assertEqual(translator.backend, "gemini")
		self.assertEqual(translator.translate_with_context("田中です。", "よろしく。", {"田中": "田中"}), "請多指教。")
		self.assertEqual(calls, [{
			"model": self.app.GEMINI_MODEL,
			"contents": "前一句話（僅供參考上下文，不用翻譯）：\n田中です。\n\n請翻譯這一句：\nよろしく。",
			"system_instruction": MASTER_LIVE_SYSTEM_PROMPT + "\n" + self.app.glossary_hint({"田中": "田中"}),
			"temperature": 0.3,
		}])

	def test_gemini_rate_limit_skips_sentence_without_sleeping(self):
		translator, _ = self.gemini_translator(lambda contents, n: RuntimeError("429 RESOURCE_EXHAUSTED"))
		self.assertEqual(translator.translate_with_context("", "よろしく。"), "")
		self.assertEqual(self.sleeps, [])
		self.assertIn("免費額度", self.printed_text())

	def test_missing_gemini_sdk_explains_how_to_fix_it(self):
		with mock.patch.dict(sys.modules, {"google": None, "google.genai": None}):
			with self.assertRaisesRegex(RuntimeError, "pip install google-genai"):
				self.app.Translator("gemini")

	def test_missing_gemini_key_still_fails_clearly(self):
		install_fake_gemini(lambda contents, n: "")
		os.environ.pop("GEMINI_API_KEY")
		with mock.patch.object(self.app, "GEMINI_API_KEY_FILE", Path("/nonexistent/gemini_api_key.txt")):
			with self.assertRaisesRegex(RuntimeError, "Gemini API 金鑰"):
				self.app.Translator("gemini")

	def test_gemini_reconstruction_keeps_master_batches_retries_and_pacing(self):
		lines = [f"日文{i}です。" for i in range(61)]
		translator, calls = self.gemini_translator(lambda contents, n: "" if n == 1 else f"批次{n}")
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual([len(call["contents"].split("請處理這一批句子：\n", 1)[1].splitlines()) for call in calls], [60, 60, 1])
		self.assertEqual(calls[0]["contents"], "請處理這一批句子：\n" + "\n".join(lines[:60]))
		self.assertTrue(calls[2]["contents"].startswith("（前面幾句當上下文參考，不用重複翻譯：\n日文58です。\n日文59です。\n\n"))
		self.assertEqual(self.sleeps, [20, 4.5])
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("翻譯模型看過完整上下文潤過的版本", text)
		self.assertTrue(text.endswith("批次2\n\n批次3\n"))
		self.assertFalse(wav_path.exists())

	def test_gemini_offline_numbered_batches_are_unchanged(self):
		lines = [f"日文{i}です。" for i in range(41)]

		def reply(contents, n):
			body = contents.split("句）：\n", 1)[1]
			return "\n".join(f"{line.split('. ', 1)[0]}. 中文{line.split('. ', 1)[1]}" for line in body.splitlines())

		translator, calls = self.gemini_translator(reply)
		cues, transcript = self.run_offline(translator, lines)
		self.assertEqual(len(calls), 2)
		self.assertIn("請翻譯這一批句子（共 40 句）：\n1. 日文0です。", calls[0]["contents"])
		self.assertTrue(calls[1]["contents"].startswith("（前面幾句當上下文參考，不用重複翻譯：\n日文38です。\n日文39です。\n\n請翻譯這一批句子（共 1 句）：\n1. 日文40です。"))
		self.assertEqual([cue["zh"] for cue in cues], [f"中文日文{i}です。" for i in range(41)])
		self.assertEqual(self.sleeps, [4.5])
		self.assertIn("**[00:00:40]**\n日文40です。\n中文日文40です。", transcript)


class PromptParityTests(TranslationBackendTestCase):
	"""The local backend must send exactly what the Gemini backend sends."""

	def test_live_caption_prompts_are_identical(self):
		translator, calls = self.gemini_translator(lambda contents, n: "請多指教。")
		local, client = self.local_translator(lambda system, user: "請多指教。")
		glossary = {"羊宮妃那": "羊宮妃那", "ゆみやひな": "羊宮妃那"}
		for prev_ja in ("", "田中です。"):
			translator.translate_with_context(prev_ja, "よろしくお願いします。", glossary)
			local.translate_with_context(prev_ja, "よろしくお願いします。", glossary)
		self.assertEqual([(call["system_instruction"], call["contents"]) for call in calls], client.calls)

	def test_reconstruction_prompts_are_identical(self):
		lines = [f"日文{i}です。" for i in range(12)]
		reply = "\n\n".join(f"{line}\n中文{i}。" for i, line in enumerate(lines))
		translator, calls = self.gemini_translator(lambda contents, n: reply)
		self.run_reconstruction(translator, lines)
		local, client = self.local_translator(lambda system, user: reply)
		self.run_reconstruction(local, lines)
		self.assertEqual([(call["system_instruction"], call["contents"]) for call in calls], client.calls)

	def test_offline_numbered_prompts_are_identical(self):
		lines = [f"日文{i}です。" for i in range(12)]
		reply = "\n".join(f"{i + 1}. 中文{i}。" for i in range(12))
		translator, calls = self.gemini_translator(lambda contents, n: reply)
		gemini_cues, gemini_transcript = self.run_offline(translator, lines)
		local, client = self.local_translator(lambda system, user: reply)
		local_cues, local_transcript = self.run_offline(local, lines)
		self.assertEqual([(call["system_instruction"], call["contents"]) for call in calls], client.calls)
		self.assertEqual(gemini_cues, local_cues)
		self.assertEqual(gemini_transcript.split("---", 1)[1], local_transcript.split("---", 1)[1])


class LocalBackendTests(TranslationBackendTestCase):
	def test_local_backend_does_not_import_the_google_sdk(self):
		with mock.patch.object(self.app, "_import_gemini_sdk", side_effect=AssertionError("Gemini SDK imported")):
			translator, _ = self.local_translator(lambda system, user: "你好。")
		self.assertEqual(translator.backend, "local")
		self.assertIsNone(self.app.genai)

	def test_environment_selects_backend_and_rejects_unknown_names(self):
		os.environ["TRANSLATION_BACKEND"] = " LOCAL "
		self.assertEqual(self.app.resolve_translation_backend(os.environ["TRANSLATION_BACKEND"]), "local")
		with self.assertRaisesRegex(RuntimeError, "gemini 或 local"):
			self.app.resolve_translation_backend("ollama")
		self.assertEqual(self.app.resolve_translation_backend(None), "gemini")

	def test_unreachable_local_service_fails_with_guidance(self):
		client = ScriptedClient(lambda s, u: "")
		client.ensure_ready = mock.Mock(side_effect=LocalLLMError("本機模型連線失敗", retryable=True, kind="connection"))
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		with mock.patch.object(self.app, "LocalTranslationEngine", factory):
			with self.assertRaisesRegex(RuntimeError, "llama-server[\\s\\S]*Gemini API"):
				self.app.Translator("local")

	def test_local_live_caption_repairs_bad_reply(self):
		replies = iter(["今日はいい天気ですね。", "今天天氣真好呢。"])
		translator, client = self.local_translator(lambda system, user: next(replies))
		self.assertEqual(translator.translate_with_context("田中です。", "今日はいい天気ですね。"), "今天天氣真好呢。")
		self.assertEqual(client.calls[1][1], "請翻譯：\n今日はいい天気ですね。")

	def test_local_reconstruction_splits_truncated_batches_and_keeps_every_line(self):
		lines = [f"日文{i}です。" for i in range(24)]

		def handler(system, user):
			batch = user.split("請處理這一批句子：\n", 1)[1].splitlines()
			if len(batch) > 10:
				return LocalLLMError("截斷", kind="truncated")
			return "\n\n".join(f"{line}\n中文{line[2:-3]}。" for line in batch)

		translator, client = self.local_translator(handler)
		out_path, wav_path = self.run_reconstruction(translator, lines)
		text = out_path.read_text(encoding="utf-8")
		for i in range(24):
			self.assertIn(f"日文{i}です。\n中文{i}。", text)
		self.assertIn("翻譯模型看過完整上下文潤過的版本", text)
		self.assertFalse(wav_path.exists())
		self.assertEqual(self.sleeps, [])

	def test_local_reconstruction_keeps_recording_when_server_dies(self):
		translator, _ = self.local_translator(lambda system, user: LocalLLMError("連線失敗", retryable=True, kind="connection"))
		out_path, wav_path = self.run_reconstruction(translator, ["日文です。"])
		self.assertIsNone(out_path)
		self.assertTrue(wav_path.exists())
		self.assertIn("已保留完整錄音", self.printed_text())

	def test_riva_reconstruction_is_labelled_as_sentence_by_sentence(self):
		translator, client = self.local_translator(lambda system, user: "中文。", mode=MODE_RIVA)
		out_path, _ = self.run_reconstruction(translator, ["日文です。", "田中です。"])
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("本機翻譯專用模型逐句翻譯", text)
		self.assertIn("日文です。\n中文。\n\n田中です。\n中文。", text)
		self.assertEqual(client.calls[1], ("", "Translate this into Traditional Chinese: __GLOSSARY_0__です。"))

	def test_local_offline_fills_missing_numbers(self):
		lines = [f"日文{i}です。" for i in range(5)]

		def handler(system, user):
			if "請翻譯這一批句子" not in user:
				return "補翻的一句。"
			body = user.split("句）：\n", 1)[1].splitlines()
			return "\n".join(f"{line.split('. ', 1)[0]}. 中文{line.split('. ', 1)[1]}" for line in body if not line.startswith("3. ") or len(body) == 1)

		translator, client = self.local_translator(handler)
		cues, _ = self.run_offline(translator, lines)
		self.assertTrue(all(cue["zh"] for cue in cues))
		self.assertEqual(cues[2]["zh"], "中文日文2です。")
		self.assertEqual(self.sleeps, [])


class WorkflowWiringTests(TranslationBackendTestCase):
	def test_worker_reuses_injected_translator_without_constructing_another(self):
		injected = mock.Mock()
		with mock.patch.object(self.app, "Translator") as translator_class, \
				mock.patch.object(self.app, "load_glossary", return_value={}), \
				tempfile.TemporaryDirectory() as temp_dir, \
				mock.patch.object(self.app, "TRANSCRIPT_DIR", Path(temp_dir)):
			worker = self.app.Worker(queue.Queue(), queue.Queue(), queue.Queue(), translator=injected)
		translator_class.assert_not_called()
		self.assertIs(worker.translator, injected)

	def test_pretranslated_cues_preserve_queue_timing_and_tuple_shape(self):
		caption_queue, transcript_queue = queue.Queue(), queue.Queue()
		self.app.push_cues_to_queues([{"start": 1.5, "ja": "日", "zh": "中"}, {"start": 3.0, "ja": "文"}], caption_queue, transcript_queue, 100.0)
		expected = [(100.0 + 1.5 + self.app.DISPLAY_DELAY_SEC, "日", "中"), (100.0 + 3.0 + self.app.DISPLAY_DELAY_SEC, "文", "")]
		self.assertEqual([caption_queue.get_nowait(), caption_queue.get_nowait()], expected)
		self.assertEqual([transcript_queue.get_nowait(), transcript_queue.get_nowait()], expected)


if __name__ == "__main__":
	unittest.main()
