import json
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import compare_translation_backends as compare


class FakeTranslator:
	def __init__(self, backend, mapping, verdict_for=None):
		self.backend = backend
		self.mapping = mapping
		self.verdict_for = verdict_for
		self.calls = []
		self.judge_prompts = []

	def translate_with_context(self, prev_ja, current_ja, glossary=None):
		self.calls.append((prev_ja, current_ja))
		return self.mapping.get(current_ja, "")

	def _chat(self, system_prompt, user_prompt):
		self.judge_prompts.append(user_prompt)
		return self.verdict_for(user_prompt)


def prefer_candidate_containing(marker):
	"""Judge that picks whichever candidate contains ``marker`` (order independent)."""
	def verdict(prompt):
		a = prompt.split("譯文 A：\n", 1)[1].split("\n\n", 1)[0]
		b = prompt.split("譯文 B：\n", 1)[1].split("\n\n", 1)[0]
		if (marker in a) == (marker in b):
			return "TIE"
		return "A" if marker in a else "B"
	return verdict


class HelperTests(unittest.TestCase):
	def test_load_samples_accepts_plain_text_and_transcript_logs(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			plain = Path(temp_dir) / "plain.txt"
			plain.write_text("# comment\nおはよう。\n\n  こんにちは。 \n", encoding="utf-8")
			self.assertEqual(compare.load_samples(plain), ["おはよう。", "こんにちは。"])
			log = Path(temp_dir) / "transcript_20260926_010203.txt"
			log.write_text("#TITLE: 節目\n\n[21:00:01] JP: おはよう。\n[21:00:01] ZH: 早安。\n\n[21:00:05] JP: またね。\n[21:00:05] ZH: 再見。\n", encoding="utf-8")
			self.assertEqual(compare.load_samples(log), ["おはよう。", "またね。"])
			self.assertEqual(compare.load_samples(log, limit=1), ["おはよう。"])

	def test_chrf_is_a_bounded_similarity(self):
		self.assertEqual(compare.chrf("今天天氣很好", "今天天氣很好"), 100.0)
		self.assertEqual(compare.chrf("甲乙丙", "丁戊己"), 0.0)
		self.assertTrue(0 < compare.chrf("今天天氣不錯", "今天天氣很好") < 100)

	def test_glossary_expectations_use_display_text_and_longest_terms(self):
		glossary = {"羊宮妃那": "羊宮妃那", "羊宮妃那のHOOOOPE!": "羊宮妃那のHOOOOPE!", "ゆみやひな": "羊宮妃那"}
		self.assertEqual(compare.glossary_expectations("羊宮妃那のHOOOOPE!です", glossary), ["羊宮妃那のHOOOOPE!"])
		self.assertEqual(compare.glossary_expectations("ゆみやひなです", glossary), ["羊宮妃那"])

	def test_wilson_lower_bound_is_conservative(self):
		self.assertLess(compare.wilson_lower_bound(0.6, 10), 0.6)
		self.assertGreater(compare.wilson_lower_bound(0.6, 1000), 0.55)
		self.assertEqual(compare.wilson_lower_bound(0.5, 0), 0.0)

	def test_judge_randomizes_order_and_maps_verdicts_to_local_view(self):
		prefer = prefer_candidate_containing("好")
		with mock.patch("builtins.print"):
			verdicts = compare.judge_pairs(
				lambda system_prompt, user_prompt: prefer(user_prompt), ["a", "b", "c", "d"], ["很好", "普通", "好", ""], ["普通", "很好", "好", "譯文"],
				rng=random.Random(7),
			)
		self.assertEqual(verdicts, ["loss", "win", "tie", "win"])
		stats = compare.judge_stats(verdicts + ["skip"])
		self.assertEqual((stats["judged"], stats["skipped"], stats["score"]), (4, 1, 0.625))


class MainTests(unittest.TestCase):
	def setUp(self):
		self.temp_dir = tempfile.TemporaryDirectory()
		self.addCleanup(self.temp_dir.cleanup)
		self.samples = Path(self.temp_dir.name) / "samples.txt"
		self.samples.write_text("ゆみやひなです。\n今日はいい天気ですね。\nまたね。\n", encoding="utf-8")
		self.glossary = {"ゆみやひな": "羊宮妃那"}
		printer = mock.patch("builtins.print")
		printer.start()
		self.addCleanup(printer.stop)

	def run_main(self, translators, *extra):
		sleeps = []
		out = Path(self.temp_dir.name) / "report.md"
		json_path = Path(self.temp_dir.name) / "result.json"
		code = compare.main(
			[str(self.samples), "--out", str(out), "--json", str(json_path), *extra],
			translator_factory=lambda name: translators[name],
			glossary_loader=lambda: self.glossary,
			sleep=sleeps.append,
		)
		return code, out.read_text(encoding="utf-8"), json.loads(json_path.read_text(encoding="utf-8")), sleeps

	def test_equal_quality_local_model_passes_gate(self):
		gemini = FakeTranslator("gemini", {"ゆみやひなです。": "我是羊宮妃那。", "今日はいい天気ですね。": "今天天氣真好。", "またね。": "再見。"}, prefer_candidate_containing("好"))
		local = FakeTranslator("local", {"ゆみやひなです。": "我是羊宮妃那。", "今日はいい天気ですね。": "今天天氣真好呢。", "またね。": "下次見。"})
		code, report, result, sleeps = self.run_main({"gemini": gemini, "local": local}, "--judge")
		self.assertEqual(code, 0)
		self.assertIn("PASS", report)
		self.assertEqual(result["judge"]["judged"], 3)
		self.assertEqual(result["summaries"]["local"]["glossary_rate"], 1.0)
		self.assertEqual(gemini.calls[1], ("ゆみやひなです。", "今日はいい天気ですね。"))
		self.assertEqual(sleeps, [4.5, 4.5, 4.5, 4.5])
		self.assertIn("| 1 | ゆみやひなです。 | 我是羊宮妃那。 | 我是羊宮妃那。 | 平手 |", report)

	def test_untranslated_local_output_fails_gate(self):
		gemini = FakeTranslator("gemini", {"ゆみやひなです。": "我是羊宮妃那。", "今日はいい天気ですね。": "今天天氣真好。", "またね。": "再見。"})
		local = FakeTranslator("local", {"ゆみやひなです。": "我是羊宮妃那。", "今日はいい天気ですね。": "今日はいい天気ですね。", "またね。": "再見。"})
		code, report, result, _ = self.run_main({"gemini": gemini, "local": local}, "--gemini-interval", "0")
		self.assertEqual(code, 1)
		self.assertIn("FAIL", report)
		self.assertEqual(result["summaries"]["local"]["problems"], {"untranslated": 1})

	def test_glossary_miss_fails_gate(self):
		gemini = FakeTranslator("gemini", {"ゆみやひなです。": "我是羊宮妃那。", "今日はいい天気ですね。": "今天天氣真好。", "またね。": "再見。"})
		local = FakeTranslator("local", {"ゆみやひなです。": "我是弓矢雛。", "今日はいい天気ですね。": "今天天氣真好。", "またね。": "再見。"})
		code, _, result, _ = self.run_main({"gemini": gemini, "local": local}, "--gemini-interval", "0")
		self.assertEqual(code, 1)
		self.assertFalse(result["gate"]["passed"])

	def test_local_only_smoke_run_needs_no_gemini(self):
		local = FakeTranslator("local", {"ゆみやひなです。": "我是羊宮妃那。", "今日はいい天気ですね。": "今天天氣真好。", "またね。": "再見。"})
		code, report, result, sleeps = self.run_main({"local": local}, "--backends", "local")
		self.assertEqual(code, 0)
		self.assertIsNone(result["gate"])
		self.assertEqual(sleeps, [])
		self.assertNotIn("結論", report)

	def test_judge_requires_both_backends(self):
		self.assertEqual(compare.main([str(self.samples), "--backends", "local", "--judge"], translator_factory=lambda name: None, glossary_loader=dict), 2)

	def test_offline_mode_uses_the_offline_batch_functions(self):
		offline = types.SimpleNamespace(
			_OFFLINE_SYSTEM_PROMPT="SYSTEM", TRANSLATION_BACKEND_LOCAL="local",
			glossary_hint=lambda glossary: "HINT",
			_translate_cues_local=mock.Mock(side_effect=lambda translator, cues, glossary, system: [cue.update(zh="本機" + cue["ja"]) for cue in cues]),
			_translate_cues_gemini=mock.Mock(side_effect=lambda translator, cues, system: [cue.update(zh="雲端" + cue["ja"]) for cue in cues]),
		)
		translators = {"gemini": FakeTranslator("gemini", {}), "local": FakeTranslator("local", {})}
		with mock.patch("builtins.print"):
			code = compare.main(
				[str(self.samples), "--mode", "offline", "--out", str(Path(self.temp_dir.name) / "o.md"), "--max-problem-gap", "1"],
				translator_factory=lambda name: translators[name], glossary_loader=lambda: self.glossary, offline_module=offline,
			)
		self.assertEqual(code, 0)
		self.assertEqual(offline._translate_cues_local.call_args.args[3], "SYSTEM\nHINT")
		self.assertEqual([cue["ja"] for cue in offline._translate_cues_gemini.call_args.args[1]], ["ゆみやひなです。", "今日はいい天気ですね。", "またね。"])


if __name__ == "__main__":
	unittest.main()
