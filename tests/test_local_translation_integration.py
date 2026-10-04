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
	sys.modules.pop("live_caption_gemini", None)
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

		# Hermetic regardless of what's on the developer's real machine (LOCAL_LLM_* env,
		# a real local_llm_server.json): never let auto-start actually run in this helper.
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=None):
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

	def test_gemini_echoing_japanese_is_retried_once_without_context(self):
		source = "スタッフさんがペコリとしてくださっております"
		translator, calls = self.gemini_translator(lambda contents, n: source if n == 1 else "工作人員向我們鞠躬了")
		self.assertEqual(translator.translate_with_context("こんばんは", source), "工作人員向我們鞠躬了")
		self.assertEqual(len(calls), 2)
		self.assertEqual(calls[1]["contents"], "請把這一句翻成繁體中文：\n" + source)

	def test_gemini_keeps_first_reply_when_retry_is_still_japanese(self):
		source = "スタッフさんがペコリとしてくださっております"
		translator, calls = self.gemini_translator(lambda contents, n: source)
		self.assertEqual(translator.translate_with_context("", source), source)
		self.assertEqual(len(calls), 2)

	def test_gemini_chinese_reply_is_not_retried(self):
		translator, calls = self.gemini_translator(lambda contents, n: "大家晚安")
		self.assertEqual(translator.translate_with_context("", "こんばんは"), "大家晚安")
		self.assertEqual(len(calls), 1)

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

	def test_gemini_reconstruction_keeps_wav_when_a_batch_fails_all_retries(self):
		lines = [f"日文{i}です。" for i in range(61)]
		translator, calls = self.gemini_translator(lambda contents, n: "" if n <= 3 else f"批次{n}")
		out_path, wav_path = self.run_reconstruction(translator, lines)
		self.assertEqual(len(calls), 4)
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("有 1 批整理失敗", text)
		self.assertTrue(text.endswith("批次4\n"))
		self.assertTrue(wav_path.exists())
		self.assertIn("已保留完整錄音", self.printed_text())



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
		# Patched so this test's outcome never depends on a developer's own untracked
		# local_llm_server.json / LOCAL_LLM_* env vars actually present on this machine.
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=None):
			with self.assertRaisesRegex(RuntimeError, "llama-server[\\s\\S]*Gemini API"):
				self.app.Translator("local")

	def test_unreachable_local_service_without_model_path_gets_autostart_hint(self):
		client = ScriptedClient(lambda s, u: "")
		client.ensure_ready = mock.Mock(side_effect=LocalLLMError("本機模型連線失敗", retryable=True, kind="connection"))
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=None) as spy:
			with self.assertRaisesRegex(RuntimeError, "llama-server[\\s\\S]*Gemini API[\\s\\S]*LOCAL_LLM_MODEL_PATH"):
				self.app.Translator("local")
		spy.assert_called_once_with(client)

	def test_configured_but_broken_local_service_reports_the_specific_reason_without_generic_wrapper(self):
		from llama_server import LlamaServerError

		client = ScriptedClient(lambda s, u: "")
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", side_effect=LlamaServerError("找不到模型檔：C:\\models\\missing.gguf")):
			with self.assertRaises(RuntimeError) as caught:
				self.app.Translator("local")
		message = str(caught.exception)
		self.assertIn("找不到模型檔", message)
		self.assertNotIn("請先啟動 llama-server 並載入模型", message)

	def test_m3_spawned_server_gets_a_different_message_when_prepare_still_fails(self):
		# The app itself just auto-started the server (ensure_llama_server returned a
		# handle); telling the user to "start it yourself" / "configure auto-start" would
		# be nonsensical when auto-start already ran and the failure is something else.
		client = ScriptedClient(lambda s, u: LocalLLMError("暖機失敗", kind="empty"))
		factory = lambda **kwargs: LocalTranslationEngine(client, startup_wait=0, log=lambda *_: None, **kwargs)
		with mock.patch.object(self.app, "LocalTranslationEngine", factory), \
				mock.patch.object(self.app, "ensure_llama_server", return_value=mock.Mock()):
			with self.assertRaises(RuntimeError) as caught:
				self.app.Translator("local")
		message = str(caught.exception)
		self.assertIn("已自動啟動 llama-server", message)
		self.assertIn("暖機失敗", message)
		self.assertNotIn("請先啟動 llama-server 並載入模型", message)
		self.assertNotIn("LOCAL_LLM_MODEL_PATH", message)

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



class WhisperModelConfigTests(TranslationBackendTestCase):
	def test_cuda_default_is_large_v3(self):
		self.assertEqual(self.app.resolve_whisper_model_size(True, environ={}), "large-v3")

	def test_cpu_default_is_small(self):
		self.assertEqual(self.app.resolve_whisper_model_size(False, environ={}), "small")

	def test_whisper_model_env_overrides_both_cuda_and_cpu_defaults(self):
		self.assertEqual(self.app.resolve_whisper_model_size(True, environ={"WHISPER_MODEL": "medium"}), "medium")
		self.assertEqual(self.app.resolve_whisper_model_size(False, environ={"WHISPER_MODEL": " large-v3-turbo "}), "large-v3-turbo")

	def test_blank_env_value_falls_back_to_the_cuda_cpu_default(self):
		self.assertEqual(self.app.resolve_whisper_model_size(True, environ={"WHISPER_MODEL": "   "}), "large-v3")


class GlossaryFileTests(TranslationBackendTestCase):
	def test_load_glossary_recreates_the_default_file_when_missing(self):
		# glossary.json is now gitignored (a user's own list should not conflict with git
		# pull); a fresh clone must still work by falling back to _DEFAULT_GLOSSARY.
		with tempfile.TemporaryDirectory() as temp_dir:
			missing_path = Path(temp_dir) / "glossary.json"
			with mock.patch.object(self.app, "GLOSSARY_PATH", missing_path):
				glossary = self.app.load_glossary()
			self.assertEqual(glossary, self.app._DEFAULT_GLOSSARY)
			self.assertTrue(missing_path.exists())
			self.assertEqual(json.loads(missing_path.read_text(encoding="utf-8")), self.app._DEFAULT_GLOSSARY)

	def test_glossary_hint_groups_misheard_spellings_under_the_correct_name(self):
		hint = self.app.glossary_hint({"羊宮妃那": "羊宮妃那", "ゆみやひな": "羊宮妃那", "陽宮ひな": "羊宮妃那", "HOOOOPE": "HOOOOPE"})
		self.assertIn("- 羊宮妃那（聽寫可能寫成：ゆみやひな、陽宮ひな）", hint)
		self.assertIn("- HOOOOPE\n", hint + "\n")
		self.assertNotIn("完全保留原文", hint)

	def test_glossary_is_not_used_as_a_whisper_prompt(self):
		translator, _ = self.gemini_translator(lambda contents, n: "中文")
		asr = mock.Mock()
		asr.transcribe.return_value = ([FakeSegment(0, "田中です。")], None)
		with tempfile.TemporaryDirectory() as temp_dir:
			wav_path = Path(temp_dir) / "transcript_20260926_010203_audio.wav"
			wav_path.write_bytes(b"RIFF")
			self.app.rebuild_transcript_from_full_audio(wav_path, asr, translator, {"田中": "田中"}, "節目")
		self.assertIsNone(asr.transcribe.call_args.kwargs.get("initial_prompt"))


SOUND_ROWS = [
	{"Name": "喇叭", "Type": "Device", "Direction": "Render", "Default": "Render",
	 "Item ID": "{0.0.0.00000000}.{aaa}", "Command-Line Friendly ID": "FxSound Audio Enhancer\\Device\\喇叭\\Render"},
	{"Name": "Realtek HD Audio 2nd output", "Type": "Device", "Direction": "Render", "Default": "",
	 "Item ID": "{0.0.0.00000000}.{bbb}", "Command-Line Friendly ID": "Realtek(R) Audio\\Device\\Realtek HD Audio 2nd output\\Render"},
	{"Name": "CABLE Input", "Type": "Device", "Direction": "Render", "Default": "",
	 "Item ID": "{0.0.0.00000000}.{ccc}", "Command-Line Friendly ID": "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render"},
	# 程式工作階段那一列的 Name 是顯示名稱（Brave），不是 brave.exe
	{"Name": "Brave", "Type": "Application", "Direction": "Render", "Default": "",
	 "Item ID": "{0.0.0.00000000}.{bbb}|\\Device\\HarddiskVolume3\\Program Files\\Brave\\brave.exe%b{0}|1%b42",
	 "Process Path": "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"},
]


class AudioRoutingTests(TranslationBackendTestCase):
	def test_app_device_is_found_by_process_path_not_display_name(self):
		self.assertEqual(self.app.get_app_output_device("brave.exe", SOUND_ROWS),
			"Realtek(R) Audio\\Device\\Realtek HD Audio 2nd output\\Render")

	def test_app_without_a_session_returns_none(self):
		self.assertIsNone(self.app.get_app_output_device("chrome.exe", SOUND_ROWS))

	def test_default_and_named_devices(self):
		self.assertEqual(self.app.get_default_output_device(SOUND_ROWS), "FxSound Audio Enhancer\\Device\\喇叭\\Render")
		self.assertEqual(self.app.find_output_device("CABLE Input", SOUND_ROWS), "VB-Audio Virtual Cable\\Device\\CABLE Input\\Render")
		self.assertIsNone(self.app.find_output_device("不存在的裝置", SOUND_ROWS))


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

	def test_worker_keeps_confident_speech_even_with_high_no_speech_prob(self):
		# large-v3 often gives clear speech over background music a no_speech_prob of 0.6~0.9;
		# faster-whisper's own rule (no_speech_prob AND low avg_logprob) already filters silence,
		# so the worker must not drop a segment just because no_speech_prob is high.
		import time
		segment = FakeSegment(0, "していく番組です")
		segment.no_speech_prob = 0.85
		segment.avg_logprob = -0.16
		translator = mock.Mock()
		translator.translate_with_context.return_value = "這是會這樣做下去的節目。"
		utterances, captions = queue.Queue(), queue.Queue()
		with mock.patch.object(self.app, "load_glossary", return_value={}), \
				tempfile.TemporaryDirectory() as temp_dir, \
				mock.patch.object(self.app, "TRANSCRIPT_DIR", Path(temp_dir)):
			worker = self.app.Worker(utterances, captions, queue.Queue(), translator=translator)
			worker.asr = mock.Mock()
			worker.asr.transcribe.return_value = ([segment], None)
			worker.start()
			utterances.put(([0.0] * 16000, time.time()))  # numpy is stubbed here; asr is mocked anyway
			try:
				_, ja, zh = captions.get(timeout=5)
			finally:
				worker.stop()
				worker.join(timeout=5)
		self.assertEqual(ja, "していく番組です")
		self.assertEqual(zh, "這是會這樣做下去的節目。")



class KeptRecordingCommandTests(TranslationBackendTestCase):
	"""Every message that keeps the recording must say how to re-process it; the console gives the exact command."""

	def assert_rebuild_command(self, text, wav_path):
		import re

		match = re.search(r'python "([^"]+)" --rebuild "([^"]+)"', text)
		self.assertIsNotNone(match, text)
		script, wav = match.groups()
		# Absolute paths, so the command works whichever folder it is pasted into.
		self.assertTrue(os.path.isabs(script) and os.path.isabs(wav), match.group(0))
		self.assertTrue(os.path.samefile(script, self.app.__file__), script)
		self.assertTrue(os.path.samefile(wav, wav_path), wav)

	def test_rebuild_command_makes_a_relative_recording_path_absolute(self):
		relative = Path("transcripts/transcript_20260926_010203_audio.wav")
		command = self.app._rebuild_command(relative)
		self.assertTrue(command.endswith(f' --rebuild "{os.path.abspath(relative)}"'), command)

	def test_partial_failure_transcript_gives_the_rebuild_command_without_local_paths(self):
		translator, _ = self.gemini_translator(lambda contents, n: "" if n <= 3 else f"批次{n}")
		out_path, wav_path = self.run_reconstruction(translator, [f"日文{i}です。" for i in range(61)])
		text = out_path.read_text(encoding="utf-8")
		self.assertIn("--rebuild", text)
		self.assertIn(wav_path.name, text)
		# The transcript is meant to be read and shared; absolute paths would expose the user's folder.
		self.assertNotIn(str(wav_path.parent), text)
		self.assertNotIn(os.path.abspath(self.app.__file__), text)

	def test_partial_failure_console_message_gives_the_rebuild_command(self):
		translator, _ = self.gemini_translator(lambda contents, n: "" if n <= 3 else f"批次{n}")
		_, wav_path = self.run_reconstruction(translator, [f"日文{i}です。" for i in range(61)])
		self.assert_rebuild_command(self.printed_text(), wav_path)

	def test_empty_translation_console_message_gives_the_rebuild_command(self):
		translator, _ = self.gemini_translator(lambda contents, n: "")
		out_path, wav_path = self.run_reconstruction(translator, ["日文です。"])
		self.assertIsNone(out_path)
		self.assert_rebuild_command(self.printed_text(), wav_path)

	def test_local_service_died_console_message_gives_the_rebuild_command(self):
		translator, _ = self.local_translator(lambda system, user: LocalLLMError("連線失敗", retryable=True, kind="connection"))
		out_path, wav_path = self.run_reconstruction(translator, ["日文です。"])
		self.assertIsNone(out_path)
		self.assert_rebuild_command(self.printed_text(), wav_path)



class SavedRecordingRebuildTests(TranslationBackendTestCase):
	"""rebuild_saved_recording re-processes a kept recording with the backend main() would pick."""

	def saved_recording(self, name="transcript_20260926_010203_audio.wav", title="節目", settings=None):
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		folder = Path(temp_dir.name)
		wav_path = folder / name
		wav_path.write_bytes(b"RIFF")
		if title is not None:
			(folder / "transcript_20260926_010203.txt").write_text(f"#TITLE: {title}\n\n", encoding="utf-8")
		settings_path = folder / "last_settings.json"
		if settings is not None:
			settings_path.write_text(json.dumps(settings), encoding="utf-8")
		patcher = mock.patch.object(self.app, "SETTINGS_PATH", settings_path)
		patcher.start()
		self.addCleanup(patcher.stop)
		return wav_path

	def rebuild_saved(self, wav_path, translator_error=None, whisper_error=None, rebuild_error=None):
		"""Run rebuild_saved_recording with the models and the rebuild step itself mocked out."""
		with mock.patch.object(self.app, "Translator", side_effect=translator_error) as translator_class, \
				mock.patch.object(self.app, "WhisperModel", side_effect=whisper_error) as whisper_class, \
				mock.patch.object(self.app, "load_glossary", return_value={"田中": "田中"}), \
				mock.patch.object(
					self.app, "rebuild_transcript_from_full_audio", return_value=Path("out.md"), side_effect=rebuild_error
				) as rebuild:
			result = self.app.rebuild_saved_recording(wav_path)
		return result, translator_class, whisper_class, rebuild

	def test_rebuild_uses_the_backend_main_would_pick_the_glossary_and_the_episode_title(self):
		cases = (
			({"TRANSLATION_BACKEND": "local"}, {"translation_backend": "gemini"}, "local"),
			({}, {"translation_backend": "local"}, "local"),
			({}, None, "gemini"),
			({"TRANSLATION_BACKEND": "ollama"}, {"translation_backend": "local"}, "gemini"),
		)
		for env, settings, expected in cases:
			with self.subTest(env=env, settings=settings), mock.patch.dict(os.environ, env):
				wav_path = self.saved_recording(settings=settings)
				result, translator_class, whisper_class, rebuild = self.rebuild_saved(wav_path)
				self.assertEqual(result, Path("out.md"))
				translator_class.assert_called_once_with(expected)
				whisper_class.assert_called_once_with(
					self.app.WHISPER_MODEL_SIZE, device=self.app.WHISPER_DEVICE, compute_type=self.app.WHISPER_COMPUTE_TYPE
				)
				rebuild.assert_called_once_with(
					wav_path, whisper_class.return_value, translator_class.return_value, {"田中": "田中"}, "節目"
				)
		self.assertIn("改用 Gemini API", self.printed_text())

	def test_rebuild_without_the_raw_transcript_has_no_episode_title(self):
		_, _, _, rebuild = self.rebuild_saved(self.saved_recording(title=None))
		self.assertEqual(rebuild.call_args.args[4], "")

	def test_unreadable_raw_transcript_still_rebuilds_without_an_episode_title(self):
		wav_path = self.saved_recording(title=None)
		# Re-saved as Big5 (Notepad's ANSI on a Taiwanese Windows), so it is not valid UTF-8.
		wav_path.with_name("transcript_20260926_010203.txt").write_bytes("#TITLE: 節目\n\n".encode("big5"))
		result, translator_class, whisper_class, rebuild = self.rebuild_saved(wav_path)
		self.assertEqual(result, Path("out.md"))
		rebuild.assert_called_once_with(
			wav_path, whisper_class.return_value, translator_class.return_value, {"田中": "田中"}, ""
		)
		self.assertRegex(self.printed_text(), r"讀取 transcript_20260926_010203\.txt 的節目名稱失敗，改用預設標題：.*can't decode")
		self.assertNotIn("用完整錄音重新整理逐字稿時發生錯誤", self.printed_text())

	def test_success_prints_where_the_new_transcript_is(self):
		result, _, _, _ = self.rebuild_saved(self.saved_recording())
		self.assertEqual(result, Path("out.md"))
		self.assertIn("已產生用完整錄音重新辨識、準確度更好的逐字稿：out.md", self.printed_text())

	def test_rebuild_error_returns_none_and_prints_the_command_to_try_again(self):
		wav_path = self.saved_recording()
		error = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
		result, _, _, _ = self.rebuild_saved(wav_path, rebuild_error=error)
		self.assertIsNone(result)
		self.assertIn("用完整錄音重新整理逐字稿時發生錯誤：Library cublas64_12.dll is not found", self.printed_text())
		self.assertIn(f"之後可以用這行指令重新處理：{self.app._rebuild_command(wav_path)}", self.printed_text())
		self.assertTrue(wav_path.exists())

	def test_whisper_load_error_returns_none_and_prints_the_command_to_try_again(self):
		wav_path = self.saved_recording()
		error = RuntimeError("CUDA failed with error no CUDA-capable device is detected")
		result, translator_class, whisper_class, rebuild = self.rebuild_saved(wav_path, whisper_error=error)
		self.assertIsNone(result)
		# Built before the slow Whisper load, so a bad API key is reported without waiting for the model.
		translator_class.assert_called_once()
		whisper_class.assert_called_once()
		rebuild.assert_not_called()
		self.assertIn("用完整錄音重新整理逐字稿時發生錯誤：CUDA failed with error no CUDA-capable device", self.printed_text())
		self.assertIn(f"之後可以用這行指令重新處理：{self.app._rebuild_command(wav_path)}", self.printed_text())
		self.assertTrue(wav_path.exists())

	def test_rebuild_error_after_the_recording_was_deleted_does_not_print_the_command(self):
		wav_path = self.saved_recording()

		def delete_then_fail(wav, *_):
			wav.unlink()
			raise RuntimeError("整理到一半發生錯誤")

		result, _, _, _ = self.rebuild_saved(wav_path, rebuild_error=delete_then_fail)
		self.assertIsNone(result)
		self.assertFalse(wav_path.exists())
		self.assertIn("用完整錄音重新整理逐字稿時發生錯誤：整理到一半發生錯誤", self.printed_text())
		# The recording is gone, so the command could only answer 找不到錄音檔.
		self.assertNotIn("之後可以用這行指令重新處理", self.printed_text())

	def test_missing_recording_returns_none_before_loading_any_model(self):
		temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(temp_dir.cleanup)
		missing = Path(temp_dir.name) / "transcript_20260926_010203_audio.wav"
		result, translator_class, whisper_class, rebuild = self.rebuild_saved(missing)
		self.assertIsNone(result)
		translator_class.assert_not_called()
		whisper_class.assert_not_called()
		rebuild.assert_not_called()
		self.assertIn(str(missing), self.printed_text())

	def test_audio_the_app_did_not_keep_is_refused_before_loading_any_model(self):
		# A successful rebuild deletes its input, so it must only ever touch the app's own recordings.
		for name in ("podcast.wav", "podcast_audio.wav", "transcript_20260926_010203_notebooklm_style.md"):
			with self.subTest(name=name):
				other = self.saved_recording(name=name)
				result, translator_class, whisper_class, rebuild = self.rebuild_saved(other)
				self.assertIsNone(result)
				translator_class.assert_not_called()
				whisper_class.assert_not_called()
				rebuild.assert_not_called()
				self.assertEqual(other.read_bytes(), b"RIFF")
				self.assertIn(str(other), self.printed_text())

	def test_translator_failure_returns_none_before_loading_whisper(self):
		error = RuntimeError("找不到 Gemini API 金鑰")
		result, _, whisper_class, rebuild = self.rebuild_saved(self.saved_recording(), translator_error=error)
		self.assertIsNone(result)
		whisper_class.assert_not_called()
		rebuild.assert_not_called()
		self.assertIn("翻譯引擎初始化失敗：找不到 Gemini API 金鑰", self.printed_text())



class RebuildCommandLineTests(TranslationBackendTestCase):
	def test_no_arguments_open_the_app_as_before(self):
		with mock.patch.object(self.app, "main") as main, \
				mock.patch.object(self.app, "rebuild_saved_recording") as rebuild:
			self.assertEqual(self.app._run_cli([]), 0)
		main.assert_called_once_with()
		rebuild.assert_not_called()

	def test_rebuild_exit_code_says_whether_a_transcript_was_produced(self):
		wav = "transcript_20260926_010203_audio.wav"
		for produced, expected in ((Path("transcript_20260926_010203_notebooklm_style.md"), 0), (None, 1)):
			with self.subTest(produced=produced):
				with mock.patch.object(self.app, "main") as main, \
						mock.patch.object(self.app, "rebuild_saved_recording", return_value=produced) as rebuild:
					self.assertEqual(self.app._run_cli(["--rebuild", wav]), expected)
				rebuild.assert_called_once_with(Path(wav))
				main.assert_not_called()

	def test_other_arguments_print_usage_and_exit_2(self):
		for argv in (["--help"], ["--rebuild"], ["transcript_20260926_010203_audio.wav"], ["--rebuild", "a.wav", "b.wav"]):
			with self.subTest(argv=argv):
				with mock.patch.object(self.app, "main") as main, \
						mock.patch.object(self.app, "rebuild_saved_recording") as rebuild:
					self.assertEqual(self.app._run_cli(argv), 2)
				main.assert_not_called()
				rebuild.assert_not_called()
		self.assertIn("--rebuild", self.printed_text())



if __name__ == "__main__":
	unittest.main()
