import unittest

from local_llm import MODE_INSTRUCT, MODE_RIVA, LocalLLMError
from local_translation import (
	TRANSLATION_FAILED_TEXT,
	LocalTranslationEngine,
	apply_glossary_targets,
	bilingual_pairs_acceptable,
	mask_glossary,
	parse_bilingual_pairs,
	restore_glossary,
	split_for_translation,
	translation_problem,
)


class ScriptedClient:
	"""Stand-in for LocalLLMClient whose replies come from a handler function."""

	def __init__(self, handler, mode=MODE_INSTRUCT, server_info=None):
		self.handler = handler
		self.mode = mode
		self.server_info = server_info or {}
		self.model = None
		self.base_url = "http://127.0.0.1:8766"
		self.timeout = 120.0
		self.max_tokens = 256
		self.calls = []
		self.ready_calls = 0

	def ensure_ready(self, wait_seconds=0.0, **_):
		self.ready_calls += 1

	def chat(self, system_prompt, user_prompt, *, max_tokens=None, timeout=None, temperature=None):
		self.calls.append({"system": system_prompt, "user": user_prompt, "max_tokens": max_tokens, "timeout": timeout})
		reply = self.handler(system_prompt, user_prompt, len(self.calls))
		if isinstance(reply, BaseException):
			raise reply
		return reply


class Clock:
	def __init__(self):
		self.now = 0.0
		self.sleeps = []
	def __call__(self):
		return self.now
	def sleep(self, seconds):
		self.sleeps.append(seconds)
		self.now += seconds


def make_engine(handler, mode=MODE_INSTRUCT, **kwargs):
	clock = Clock()
	logs = []
	client = ScriptedClient(handler, mode=mode, server_info=kwargs.pop("server_info", None))
	engine = LocalTranslationEngine(
		client, live_timeout=kwargs.pop("live_timeout", 15), batch_lines=kwargs.pop("batch_lines", 4),
		batch_max_tokens=4096, startup_wait=0, segment_chars=kwargs.pop("segment_chars", 120),
		clock=clock, sleep=clock.sleep, log=logs.append, **kwargs,
	)
	return engine, client, clock, logs


def connection_error():
	return LocalLLMError("本機模型連線失敗", retryable=True, kind="connection")


def numbered_reply(user_prompt, translate=lambda text: f"譯{text}"):
	body = user_prompt.split("句）：\n", 1)[1]
	lines = []
	for line in body.splitlines():
		number, text = line.split(". ", 1)
		lines.append(f"{number}. {translate(text)}")
	return "\n".join(lines)


def numbered_prompt(batch, context_tail):
	numbered = "\n".join(f"{n + 1}. {text}" for n, text in enumerate(batch))
	return ((f"（前面幾句當上下文參考，不用重複翻譯：\n{context_tail}\n\n") if context_tail else "") + f"請翻譯這一批句子（共 {len(batch)} 句）：\n{numbered}"


def parse_numbered(text, expected):
	import re
	result = [""] * expected
	for line in text.splitlines():
		match = re.match(r"\s*(\d+)[.\、\)]\s*(.+)", line)
		if match and 0 <= int(match.group(1)) - 1 < expected:
			result[int(match.group(1)) - 1] = match.group(2).strip()
	return result


def rebuild_prompt(batch, context_tail):
	return ((f"（前面幾句當上下文參考，不用重複翻譯：\n{context_tail}\n\n") if context_tail else "") + "請處理這一批句子：\n" + "\n".join(batch)


def context_prompts(prev_ja, current_ja):
	user = f"前一句話（僅供參考上下文，不用翻譯）：\n{prev_ja}\n\n請翻譯這一句：\n{current_ja}" if prev_ja else f"請翻譯這一句：\n{current_ja}"
	return "SYSTEM", user


class TextCheckTests(unittest.TestCase):
	def test_translation_problem_flags_unusable_replies(self):
		self.assertIsNone(translation_problem("我們今天也一起加油吧。", "今日も一緒に頑張ろうね。"))
		self.assertEqual(translation_problem("  ", "テスト"), "empty")
		self.assertEqual(translation_problem("前一句話：你好\n請翻譯這一句：謝謝", "ありがとう"), "echo")
		self.assertEqual(translation_problem("今日はいい天気ですね。", "今日はいい天気ですね。"), "untranslated")
		self.assertEqual(translation_problem("__GLOSSARY_0__來了", "ゆみやひなが来た"), "marker")
		self.assertEqual(translation_problem("好" * 200, "はい"), "verbose")

	def test_glossary_kana_names_are_not_mistaken_for_untranslated_text(self):
		glossary = {"ゆみやひな": "ゆみやひな"}
		self.assertIsNone(translation_problem("ゆみやひな好可愛", "ゆみやひなは可愛い。", glossary))
		self.assertEqual(translation_problem("ゆみやひな好可愛", "ゆみやひなは可愛い。"), "untranslated")
		self.assertIsNone(translation_problem("ゆみやひな今天也很可愛。", "ゆみやひなは今日も可愛い。"))

	def test_parse_bilingual_pairs_handles_labels_and_missing_blank_lines(self):
		self.assertEqual(
			parse_bilingual_pairs("日文：おはよう。\n中文：早安。\n\nありがとう。\n謝謝。"),
			[("おはよう。", "早安。"), ("ありがとう。", "謝謝。")],
		)
		self.assertEqual(
			parse_bilingual_pairs("おはよう。\n早安。\nありがとう。\n謝謝。"),
			[("おはよう。", "早安。"), ("ありがとう。", "謝謝。")],
		)
		self.assertEqual(parse_bilingual_pairs("おはよう。"), [("おはよう。", "")])

	def test_bilingual_acceptance_tolerates_merged_lines_but_not_gaps(self):
		batch = [f"文{i}です。" for i in range(8)]
		pairs = [(f"文{i}です。", f"句子{i}。") for i in range(6)]
		self.assertTrue(bilingual_pairs_acceptable(pairs, batch, None))
		self.assertFalse(bilingual_pairs_acceptable(pairs[:5], batch, None))
		broken = pairs[:5] + [("文5です。", "")]
		self.assertFalse(bilingual_pairs_acceptable(broken, batch[:6], None))


class GlossaryTests(unittest.TestCase):
	def test_mask_and_restore_use_configured_targets_longest_first(self):
		glossary = {"羊宮妃那": "羊宮妃那", "羊宮妃那のHOOOOPE!": "羊宮妃那のHOOOOPE!", "ゆみやひな": "羊宮妃那"}
		masked = mask_glossary("羊宮妃那のHOOOOPE!でゆみやひなとゆみやひな", glossary)
		self.assertEqual(masked.text, "__GLOSSARY_0__で__GLOSSARY_1__と__GLOSSARY_1__")
		self.assertEqual(restore_glossary("在__GLOSSARY_0__裡__GLOSSARY_1__和__GLOSSARY_1__", masked), "在羊宮妃那のHOOOOPE!裡羊宮妃那和羊宮妃那")
		self.assertIsNone(restore_glossary("在__GLOSSARY_0__裡__GLOSSARY_1__", masked))
		self.assertIsNone(restore_glossary("__GLOSSARY_0____GLOSSARY_1____GLOSSARY_1____GLOSSARY_7__", masked))

	def test_literal_marker_text_disables_masking(self):
		masked = mask_glossary("__GLOSSARY_0__ と ゆみやひな", {"ゆみやひな": "羊宮妃那"})
		self.assertEqual(masked.text, "__GLOSSARY_0__ と ゆみやひな")
		self.assertEqual(restore_glossary("x", masked), "x")

	def test_apply_targets_maps_only_differing_values(self):
		self.assertEqual(apply_glossary_targets("ゆみやひな和羊宮妃那", {"ゆみやひな": "羊宮妃那", "羊宮妃那": "羊宮妃那"}), "羊宮妃那和羊宮妃那")

	def test_split_prefers_sentences_and_keeps_terms_whole(self):
		text = "今日はいい天気ですね。" * 20
		pieces = split_for_translation(text, 60)
		self.assertEqual("".join(pieces), text)
		self.assertTrue(all(len(piece) <= 60 and piece.endswith("。") for piece in pieces))
		long_run = "あ" * 55 + "羊宮妃那のHOOOOPE!" + "い" * 30
		pieces = split_for_translation(long_run, 60, {"羊宮妃那のHOOOOPE!": "羊宮妃那のHOOOOPE!"})
		self.assertEqual("".join(pieces), long_run)
		self.assertTrue(any("羊宮妃那のHOOOOPE!" in piece for piece in pieces))


class LiveTranslationTests(unittest.TestCase):
	def test_valid_reply_is_returned_with_live_timeout(self):
		engine, client, _, _ = make_engine(lambda s, u, n: " 請多指教。 ")
		self.assertEqual(engine.translate_live("SYS", "USER", "よろしく。", None, "FALLBACK"), "請多指教。")
		self.assertEqual(client.calls[0]["timeout"], 15)
		self.assertEqual(len(client.calls), 1)

	def test_echoed_or_untranslated_reply_is_repaired_without_context(self):
		replies = iter(["前一句話：田中です。請翻譯這一句：よろしく", "請多指教。"])
		engine, client, _, _ = make_engine(lambda s, u, n: next(replies))
		self.assertEqual(engine.translate_live("SYS", "USER", "よろしく。", None, "FALLBACK"), "請多指教。")
		self.assertEqual([call["user"] for call in client.calls], ["USER", "FALLBACK"])

	def test_reply_that_retranslates_the_context_sentence_is_repaired(self):
		replies = iter(["我是田中。", "我是田中。請多指教。", "請多指教。"])
		engine, client, _, _ = make_engine(lambda s, u, n: next(replies))
		self.assertEqual(engine.translate_live("SYS", "USER1", "田中です。", None, "FALLBACK1"), "我是田中。")
		self.assertEqual(engine.translate_live("SYS", "USER2", "よろしく。", None, "FALLBACK2"), "請多指教。")
		self.assertEqual([call["user"] for call in client.calls], ["USER1", "USER2", "FALLBACK2"])

	def test_repeated_sentence_is_not_mistaken_for_context_leak(self):
		engine, client, _, _ = make_engine(lambda s, u, n: "謝謝大家。")
		for _ in range(2):
			self.assertEqual(engine.translate_live("SYS", "USER", "ありがとう。", None, "FALLBACK"), "謝謝大家。")
		self.assertEqual(len(client.calls), 2)

	def test_unreachable_server_is_not_retried_during_live_captions(self):
		engine, client, clock, _ = make_engine(lambda s, u, n: connection_error())
		self.assertEqual(engine.translate_live("SYS", "USER", "よろしく。", None, "FALLBACK"), "")
		self.assertEqual(len(client.calls), 1)
		self.assertEqual(clock.sleeps, [])

	def test_circuit_breaker_pauses_live_calls_then_recovers(self):
		state = {"down": True}
		engine, client, clock, logs = make_engine(lambda s, u, n: connection_error() if state["down"] else "好。")
		for _ in range(3):
			engine.translate_live("SYS", "USER", "はい。", None, "FALLBACK")
		self.assertEqual(len(client.calls), 3)
		engine.translate_live("SYS", "USER", "はい。", None, "FALLBACK")
		self.assertEqual(len(client.calls), 3)
		self.assertTrue(any("暫停翻譯" in line for line in logs))
		state["down"] = False
		clock.now += 31
		self.assertEqual(engine.translate_live("SYS", "USER", "はい。", None, "FALLBACK"), "好。")

	def test_riva_profile_uses_plain_prompt_and_restores_glossary(self):
		seen = []
		def handler(system, user, n):
			seen.append((system, user))
			return "__GLOSSARY_0__今天也很可愛。"
		engine, _, _, _ = make_engine(handler, mode=MODE_RIVA)
		result = engine.translate_live("SYS", "USER", "ゆみやひなは今日も可愛い。", {"ゆみやひな": "羊宮妃那"}, "FALLBACK")
		self.assertEqual(result, "羊宮妃那今天也很可愛。")
		self.assertEqual(seen, [("", "Translate this into Traditional Chinese: __GLOSSARY_0__は今日も可愛い。")])

	def test_riva_marker_loss_falls_back_to_unmasked_translation(self):
		replies = iter(["今天也很可愛。", "ゆみやひな今天也很可愛。"])
		engine, client, _, _ = make_engine(lambda s, u, n: next(replies), mode=MODE_RIVA)
		self.assertEqual(engine.translate_live("", "", "ゆみやひなは今日も可愛い。", {"ゆみやひな": "羊宮妃那"}), "羊宮妃那今天也很可愛。")
		self.assertEqual(client.calls[1]["user"], "Translate this into Traditional Chinese: ゆみやひなは今日も可愛い。")

	def test_riva_long_input_is_segmented(self):
		engine, client, _, _ = make_engine(lambda s, u, n: f"段{n}。", mode=MODE_RIVA, segment_chars=22)
		result = engine.translate_live("", "", "今日はいい天気ですね。" * 4, None)
		self.assertEqual(result, "段1。段2。")
		self.assertEqual([call["user"] for call in client.calls], ["Translate this into Traditional Chinese: 今日はいい天気ですね。今日はいい天気ですね。"] * 2)


class BatchTests(unittest.TestCase):
	def test_batch_chat_retries_briefly_then_opens_breaker(self):
		engine, client, clock, _ = make_engine(lambda s, u, n: connection_error())
		from local_translation import LocalServiceUnavailable
		with self.assertRaises(LocalServiceUnavailable):
			engine.batch_chat("S", "U")
		self.assertEqual(len(client.calls), 3)
		self.assertEqual(clock.sleeps, [2.0, 5.0])

	def test_numbered_batches_fill_gaps_instead_of_leaving_blanks(self):
		texts = [f"文{i}です。" for i in range(6)]
		def handler(system, user, n):
			if user.startswith("前一句話") or user.startswith("請翻譯這一句"):
				return "單句補翻。"
			reply = numbered_reply(user)
			if n == 1:
				reply = "\n".join(line for line in reply.splitlines() if not line.startswith("2. "))
			return reply
		engine, client, _, _ = make_engine(handler, batch_lines=4)
		outcome = engine.translate_numbered(
			texts, glossary=None, system_prompt="SYS", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		self.assertEqual(outcome.results, [f"譯文{i}です。" for i in range(6)])
		self.assertEqual(outcome.failed, 0)
		self.assertIn("請翻譯這一批句子（共 1 句）：\n1. 文1です。", client.calls[1]["user"])
		self.assertIn("文0です。", client.calls[1]["user"].split("請翻譯")[0])

	def test_numbered_batch_splits_when_truncated(self):
		def handler(system, user, n):
			if "（共 4 句）" in user:
				return LocalLLMError("截斷", kind="truncated")
			return numbered_reply(user)
		engine, client, _, _ = make_engine(handler, batch_lines=4)
		outcome = engine.translate_numbered(
			[f"文{i}" for i in range(4)], glossary=None, system_prompt="SYS", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		self.assertEqual(outcome.results, [f"譯文{i}" for i in range(4)])
		self.assertEqual(len(client.calls), 3)

	def test_slow_model_timeouts_split_batches_instead_of_aborting(self):
		def handler(system, user, n):
			if "（共 4 句）" in user:
				return LocalLLMError("逾時", retryable=True, kind="timeout")
			return numbered_reply(user)
		engine, client, clock, _ = make_engine(handler, batch_lines=4)
		outcome = engine.translate_numbered(
			[f"文{i}" for i in range(8)], glossary=None, system_prompt="SYS", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		self.assertFalse(outcome.aborted)
		self.assertEqual(outcome.results, [f"譯文{i}" for i in range(8)])
		self.assertEqual(clock.sleeps, [])

	def test_hung_server_aborts_after_consecutive_timeouts(self):
		engine, client, _, _ = make_engine(lambda s, u, n: LocalLLMError("逾時", retryable=True, kind="timeout"), batch_lines=8)
		outcome = engine.translate_numbered(
			[f"文{i}" for i in range(8)], glossary=None, system_prompt="SYS", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		self.assertTrue(outcome.aborted)
		self.assertEqual(len(client.calls), 3)

	def test_later_batches_carry_more_preceding_lines_than_gemini(self):
		engine, client, _, _ = make_engine(lambda s, u, n: numbered_reply(u), batch_lines=10)
		self.assertEqual(engine.context_lines, 8)
		engine.translate_numbered(
			[f"文{i}" for i in range(20)], glossary=None, system_prompt="SYS", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		tail = client.calls[1]["user"].split("\n\n", 1)[0]
		self.assertEqual(tail, "（前面幾句當上下文參考，不用重複翻譯：\n" + "\n".join(f"文{i}" for i in range(2, 10)))

	def test_numbered_batch_reports_abort_when_server_dies(self):
		engine, _, _, _ = make_engine(lambda s, u, n: connection_error(), batch_lines=2)
		outcome = engine.translate_numbered(
			["一", "二", "三"], glossary=None, system_prompt="SYS", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		self.assertTrue(outcome.aborted)
		self.assertEqual(outcome.failed, 3)

	def test_bilingual_polish_splits_malformed_batches_and_never_drops_lines(self):
		lines = [f"文{i}です。" for i in range(4)]
		def handler(system, user, n):
			batch = user.split("請處理這一批句子：\n", 1)[1].splitlines() if "請處理這一批句子" in user else []
			if len(batch) == 4:
				return "好的，以下是翻譯："
			if user.startswith("前一句話"):
				return "第三句。"
			# Any batch containing line 3 drops it, forcing a split down to the single-line fallback.
			return "\n\n".join(f"{line}\n句{line[1]}。" if line != "文3です。" else line for line in batch)
		engine, client, _, _ = make_engine(handler, batch_lines=4)
		parts = engine.polish_bilingual(lines, glossary=None, system_prompt="SYS", build_user_prompt=rebuild_prompt, context_prompts=context_prompts)
		self.assertEqual(parts, ["文0です。\n句0。\n\n文1です。\n句1。\n\n文2です。\n句2。\n\n文3です。\n第三句。"])

	def test_bilingual_polish_returns_none_when_server_dies(self):
		engine, _, _, logs = make_engine(lambda s, u, n: connection_error())
		self.assertIsNone(engine.polish_bilingual(["文"], glossary=None, system_prompt="S", build_user_prompt=rebuild_prompt, context_prompts=context_prompts))
		self.assertTrue(any("停止整理" in line for line in logs))

	def test_permanent_errors_keep_source_visible(self):
		engine, _, _, _ = make_engine(lambda s, u, n: LocalLLMError("HTTP 400", kind="http", status=400))
		parts = engine.polish_bilingual(["文です。"], glossary=None, system_prompt="S", build_user_prompt=rebuild_prompt, context_prompts=context_prompts)
		self.assertEqual(parts, [f"文です。\n{TRANSLATION_FAILED_TEXT}"])

	def test_batch_budget_respects_server_context(self):
		engine, client, _, _ = make_engine(lambda s, u, n: numbered_reply(u, lambda text: "中文譯文"), batch_lines=4, server_info={"n_ctx": 1024})
		engine.translate_numbered(
			["あ" * 300] * 4, glossary=None, system_prompt="S", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		self.assertEqual(client.calls[0]["max_tokens"], 512)

	def test_riva_profile_translates_batches_line_by_line(self):
		engine, client, _, _ = make_engine(lambda s, u, n: f"譯{n}", mode=MODE_RIVA, batch_lines=2)
		outcome = engine.translate_numbered(
			["一です", "二です", "三です"], glossary=None, system_prompt="S", build_user_prompt=numbered_prompt,
			parse_numbered=parse_numbered, context_prompts=context_prompts,
		)
		self.assertEqual(outcome.results, ["譯1", "譯2", "譯3"])
		self.assertTrue(all(call["user"].startswith("Translate this into Traditional Chinese: ") for call in client.calls))


class PrepareTests(unittest.TestCase):
	def test_prepare_waits_detects_and_warms_up(self):
		engine, client, _, logs = make_engine(lambda s, u, n: "你好。", server_info={"model_alias": "qwen2.5-14b", "n_ctx": 2048})
		engine.prepare("SYS", "請翻譯這一句：\nこんにちは。", "こんにちは。")
		self.assertEqual(client.ready_calls, 1)
		self.assertEqual(client.calls[0]["user"], "請翻譯這一句：\nこんにちは。")
		self.assertTrue(any("instruct" in line for line in logs))
		self.assertTrue(any("context 只有 2048" in line for line in logs))
		self.assertTrue(any("こんにちは。 → 你好。" in line for line in logs))

	def test_prepare_surfaces_request_errors(self):
		engine, _, _, _ = make_engine(lambda s, u, n: LocalLLMError("HTTP 400（model is required）", kind="http", status=400))
		with self.assertRaisesRegex(LocalLLMError, "model is required"):
			engine.prepare("SYS", "U", "こんにちは。")

	def test_prepare_warns_about_translation_only_models(self):
		engine, client, _, logs = make_engine(lambda s, u, n: "你好。", mode=MODE_RIVA)
		engine.prepare("SYS", "U", "こんにちは。")
		self.assertEqual(client.calls[0]["system"], "")
		self.assertTrue(any("翻譯專用模型不會參考前一句" in line for line in logs))

	def test_invalid_environment_settings_fail_clearly(self):
		import os
		from unittest import mock
		with mock.patch.dict(os.environ, {"LOCAL_LLM_LIVE_TIMEOUT": "soon"}):
			with self.assertRaisesRegex(LocalLLMError, "LOCAL_LLM_LIVE_TIMEOUT"):
				LocalTranslationEngine(ScriptedClient(lambda *a: ""))
		with mock.patch.dict(os.environ, {"LOCAL_LLM_BATCH_LINES": "0"}):
			with self.assertRaisesRegex(LocalLLMError, "LOCAL_LLM_BATCH_LINES"):
				LocalTranslationEngine(ScriptedClient(lambda *a: ""))


if __name__ == "__main__":
	unittest.main()
