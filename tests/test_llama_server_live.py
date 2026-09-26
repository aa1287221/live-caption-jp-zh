"""Contract checks against a real, already running llama-server (skipped by default).

Run them against the model you actually use:

    LLAMA_SERVER_URL=http://127.0.0.1:8766 python -m unittest discover -s tests -p test_llama_server_live.py -v

PowerShell: $env:LLAMA_SERVER_URL = "http://127.0.0.1:8766" before the same command.
The checks are model-agnostic: they verify the HTTP contract the application relies on
(readiness, mode detection, a usable Chinese reply, truncation and context-overflow
signals), not translation quality — use compare_translation_backends.py for quality.
"""

import os
import unittest

from local_llm import MODE_INSTRUCT, MODE_RIVA, RIVA_TRANSLATION_INSTRUCTION, LocalLLMClient, LocalLLMError
from local_translation import translation_problem


SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "").strip()
SOURCE = "今日はいい天気ですね。"


@unittest.skipUnless(SERVER_URL, "set LLAMA_SERVER_URL to run checks against a real llama-server")
class LlamaServerContractTests(unittest.TestCase):
	@classmethod
	def setUpClass(cls):
		cls.client = LocalLLMClient(base_url=SERVER_URL, timeout=float(os.environ.get("LOCAL_LLM_TIMEOUT", "120")))
		cls.client.ensure_ready(wait_seconds=60)
		cls.mode = cls.client.mode

	def translate(self, **kwargs):
		if self.mode == MODE_RIVA:
			return self.client.chat("", f"{RIVA_TRANSLATION_INSTRUCTION} {SOURCE}", **kwargs)
		return self.client.chat("你是日翻中口譯，只輸出繁體中文翻譯。", f"請翻譯這一句：\n{SOURCE}", **kwargs)

	def test_mode_is_detected(self):
		self.assertIn(self.mode, (MODE_INSTRUCT, MODE_RIVA))

	def test_translation_reply_is_usable_chinese(self):
		reply = self.translate(max_tokens=256)
		self.assertIsNone(translation_problem(reply, SOURCE), reply)

	def test_output_limit_is_reported_as_truncation(self):
		with self.assertRaises(LocalLLMError) as caught:
			self.translate(max_tokens=1)
		self.assertEqual(caught.exception.kind, "truncated")

	def test_oversized_prompt_is_reported_as_context_overflow(self):
		n_ctx = self.client.server_info.get("n_ctx")
		if not isinstance(n_ctx, int) or n_ctx <= 0 or n_ctx > 32768:
			self.skipTest("server did not report a usable n_ctx")
		with self.assertRaises(LocalLLMError) as caught:
			self.client.chat("", "あ" * (n_ctx * 2), max_tokens=16)
		self.assertTrue(caught.exception.too_large, caught.exception.kind)


if __name__ == "__main__":
	unittest.main()
