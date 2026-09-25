"""
compare_translation_backends.py
用同一份日文樣本分別跑 Gemini API 與本機語言模型，產生並排比較報告，並用門檻
判斷本機模型「有沒有比 Gemini 差」——換模型、調參數、改提示詞之後拿來做回歸驗證。

用法：
    python compare_translation_backends.py 樣本.txt
    python compare_translation_backends.py transcripts/transcript_20260926_210000.txt --judge
    python compare_translation_backends.py 樣本.txt --mode offline --out 報告.md
    python compare_translation_backends.py 樣本.txt --backends local     （只測本機模型，不需要金鑰）

樣本檔：一行一句日文（空白行、# 開頭的行會略過）；也可以直接丟即時字幕存下來的
transcripts/transcript_*.txt，會自動取出裡面的 JP 句子。

兩個引擎用的是跟主程式完全相同的 Translator 與提示詞：
    --mode live     逐句、帶前一句上下文（即時字幕的路徑，預設）
    --mode offline  編號批次（transcribe_audio_file.py 預先轉錄的路徑）

報告內容：每個引擎的失敗/問題率（空白、照抄提示、沒翻成中文、過長）、術語命中率、
延遲，兩邊譯文的 chrF 相似度；加 --judge 會請 Gemini 盲評每一句（A/B 隨機排序），
算出本機模型的勝率分數。Gemini 評審如果偏好自己的譯文，只會讓本機模型更難過關，
所以通過門檻的結果是保守的。

門檻（都可以用參數調整），全部通過才回傳 exit code 0：
    本機問題率 ≤ Gemini 問題率 + 2%
    本機術語命中率 ≥ Gemini 術語命中率 - 2%（樣本裡有術語時才檢查）
    有 --judge 時：本機分數（勝 + 平手/2）≥ 0.45
"""

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from local_translation import translation_problem


_LOG_JP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] JP: (.*)$")
_VERDICT_RE = re.compile(r"\b(TIE|A|B)\b", re.IGNORECASE)

JUDGE_SYSTEM_PROMPT = (
	"你是嚴格的日文→台灣繁體中文字幕翻譯評審。使用者會給你前一句日文（上下文）、"
	"要翻譯的日文句子，以及兩個候選譯文 A、B。請依序比較：意思是否正確完整（最重要）、"
	"有沒有漏譯或多譯、專有名詞是否正確、中文是否自然口語、適合當字幕。"
	"只輸出一個詞：A、B 或 TIE（兩者品質相當時），不要輸出任何其他文字。"
)


# ---------- samples and scoring ----------

def load_samples(path: Path, limit: int | None = None) -> list[str]:
	"""Read one sentence per line, or the JP lines of a live-caption transcript log."""
	lines = path.read_text(encoding="utf-8").splitlines()
	log_lines = [match.group(1).strip() for match in map(_LOG_JP_RE.match, lines) if match]
	if log_lines:
		samples = log_lines
	else:
		samples = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
	samples = [sample for sample in samples if sample]
	return samples[:limit] if limit else samples


def _char_ngrams(text: str, n: int) -> Counter:
	text = re.sub(r"\s+", "", text)
	return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf(hypothesis: str, reference: str, max_n: int = 6, beta: float = 2.0) -> float:
	"""Character n-gram F-score (0-100); a similarity measure, not a quality score."""
	precisions, recalls = [], []
	for n in range(1, max_n + 1):
		hyp, ref = _char_ngrams(hypothesis, n), _char_ngrams(reference, n)
		if not hyp or not ref:
			continue
		overlap = sum((hyp & ref).values())
		precisions.append(overlap / sum(hyp.values()))
		recalls.append(overlap / sum(ref.values()))
	if not precisions:
		return 100.0 if hypothesis.strip() == reference.strip() else 0.0
	precision, recall = statistics.mean(precisions), statistics.mean(recalls)
	if precision == 0 and recall == 0:
		return 0.0
	return 100.0 * (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)


def glossary_expectations(source: str, glossary: dict | None) -> list[str]:
	"""Display text expected in the translation for glossary terms present in ``source``."""
	expected = []
	for term in sorted((key for key in (glossary or {}) if isinstance(key, str) and key), key=len, reverse=True):
		if term in source and not any(term in longer for longer in expected):
			target = glossary.get(term)
			expected.append(target if isinstance(target, str) and target.strip() else term)
	return expected


def percentile(values: list[float], fraction: float) -> float:
	if not values:
		return 0.0
	ordered = sorted(values)
	index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
	return ordered[index]


def wilson_lower_bound(score: float, count: int, z: float = 1.96) -> float:
	if count <= 0:
		return 0.0
	centre = score + z * z / (2 * count)
	margin = z * math.sqrt(score * (1 - score) / count + z * z / (4 * count * count))
	return (centre - margin) / (1 + z * z / count)


# ---------- running the backends ----------

@dataclass
class BackendRun:
	name: str
	outputs: list = field(default_factory=list)
	latencies: list = field(default_factory=list)


def run_live(name, translator, samples, glossary, *, pace_seconds=0.0, retry_empty=0, retry_wait=20.0, sleep=time.sleep, clock=time.perf_counter) -> BackendRun:
	"""Translate like the live Worker: each sentence with the previous one as context."""
	run = BackendRun(name)
	for index, sentence in enumerate(samples):
		previous = samples[index - 1] if index > 0 else ""
		started = clock()
		output = translator.translate_with_context(previous, sentence, glossary)
		for _ in range(retry_empty):
			if output:
				break
			sleep(retry_wait)
			output = translator.translate_with_context(previous, sentence, glossary)
		run.latencies.append(clock() - started)
		run.outputs.append(output or "")
		print(f"  [{name}] {index + 1}/{len(samples)}")
		if pace_seconds and index + 1 < len(samples):
			sleep(pace_seconds)
	return run


def run_offline(name, translator, samples, glossary, offline_module=None, clock=time.perf_counter) -> BackendRun:
	"""Translate like transcribe_audio_file.py: numbered batches with the same prompts."""
	if offline_module is None:
		import transcribe_audio_file as offline_module
	cues = [{"start": float(index), "end": index + 0.5, "ja": sentence} for index, sentence in enumerate(samples)]
	system_prompt = offline_module._OFFLINE_SYSTEM_PROMPT
	hint = offline_module.glossary_hint(glossary) if glossary else ""
	if hint:
		system_prompt += "\n" + hint
	started = clock()
	if translator.backend == offline_module.TRANSLATION_BACKEND_LOCAL:
		offline_module._translate_cues_local(translator, cues, glossary, system_prompt)
	else:
		offline_module._translate_cues_gemini(translator, cues, system_prompt)
	per_line = (clock() - started) / max(1, len(samples))
	return BackendRun(name, [cue.get("zh", "") for cue in cues], [per_line] * len(samples))


def summarize(run: BackendRun, samples: list[str], glossary: dict | None) -> dict:
	problems = Counter()
	glossary_total = glossary_hits = 0
	for sentence, output in zip(samples, run.outputs):
		problem = translation_problem(output, sentence, glossary)
		if problem:
			problems[problem] += 1
		for expected in glossary_expectations(sentence, glossary):
			glossary_total += 1
			glossary_hits += expected in output
	count = max(1, len(samples))
	return {
		"name": run.name,
		"count": len(samples),
		"problem_rate": sum(problems.values()) / count,
		"problems": dict(problems),
		"glossary_total": glossary_total,
		"glossary_rate": glossary_hits / glossary_total if glossary_total else None,
		"latency_p50": percentile(run.latencies, 0.5),
		"latency_p95": percentile(run.latencies, 0.95),
	}


# ---------- blind judging ----------

def judge_pairs(judge, samples, gemini_outputs, local_outputs, *, rng, pace_seconds=0.0, retry_empty=2, retry_wait=20.0, sleep=time.sleep) -> list[str]:
	"""Return "win"/"loss"/"tie"/"skip" per sentence from the local model's point of view."""
	verdicts = []
	for index, (sentence, gemini_text, local_text) in enumerate(zip(samples, gemini_outputs, local_outputs)):
		if not gemini_text and not local_text:
			verdicts.append("skip")
			continue
		if not local_text or not gemini_text:
			verdicts.append("loss" if not local_text else "win")
			continue
		local_is_a = rng.random() < 0.5
		candidate_a, candidate_b = (local_text, gemini_text) if local_is_a else (gemini_text, local_text)
		previous = samples[index - 1] if index > 0 else "（無）"
		user_prompt = (
			f"前一句日文（上下文）：\n{previous}\n\n要翻譯的日文：\n{sentence}\n\n"
			f"譯文 A：\n{candidate_a}\n\n譯文 B：\n{candidate_b}\n\n哪一個比較好？只回答 A、B 或 TIE。"
		)
		reply = judge(JUDGE_SYSTEM_PROMPT, user_prompt)
		for _ in range(retry_empty):
			if reply:
				break
			sleep(retry_wait)
			reply = judge(JUDGE_SYSTEM_PROMPT, user_prompt)
		match = _VERDICT_RE.search(reply or "")
		if not match:
			verdicts.append("skip")
		else:
			choice = match.group(1).upper()
			if choice == "TIE":
				verdicts.append("tie")
			else:
				verdicts.append("win" if (choice == "A") == local_is_a else "loss")
		print(f"  [judge] {index + 1}/{len(samples)}")
		if pace_seconds and index + 1 < len(samples):
			sleep(pace_seconds)
	return verdicts


def judge_stats(verdicts: list[str]) -> dict:
	counted = [verdict for verdict in verdicts if verdict != "skip"]
	tally = Counter(counted)
	score = (tally["win"] + 0.5 * tally["tie"]) / len(counted) if counted else 0.0
	return {
		"judged": len(counted),
		"skipped": len(verdicts) - len(counted),
		"win": tally["win"],
		"tie": tally["tie"],
		"loss": tally["loss"],
		"score": score,
		"score_lower_95": wilson_lower_bound(score, len(counted)),
	}


# ---------- gate and report ----------

def evaluate_gate(gemini: dict, local: dict, judged: dict | None, *, max_problem_gap: float, max_glossary_gap: float, min_judge_score: float) -> tuple[bool, list[str]]:
	checks = []
	passed = local["problem_rate"] <= gemini["problem_rate"] + max_problem_gap
	checks.append((passed, f"問題率：本機 {local['problem_rate']:.1%}，Gemini {gemini['problem_rate']:.1%}（容許差 {max_problem_gap:.0%}）"))
	if gemini["glossary_rate"] is not None and local["glossary_rate"] is not None:
		passed = local["glossary_rate"] >= gemini["glossary_rate"] - max_glossary_gap
		checks.append((passed, f"術語命中率：本機 {local['glossary_rate']:.1%}，Gemini {gemini['glossary_rate']:.1%}（容許差 {max_glossary_gap:.0%}）"))
	if judged is not None:
		passed = judged["judged"] > 0 and judged["score"] >= min_judge_score
		checks.append((passed, f"盲評分數：本機 {judged['score']:.2f}（勝 {judged['win']}／平 {judged['tie']}／負 {judged['loss']}，95% 下界 {judged['score_lower_95']:.2f}，門檻 {min_judge_score:.2f}）"))
	return all(ok for ok, _ in checks), [("✅ " if ok else "❌ ") + text for ok, text in checks]


def _cell(text: str) -> str:
	return (text or "（空白）").replace("|", "｜").replace("\n", " ")


def render_report(samples, runs, summaries, verdicts, judged, gate, mode) -> str:
	lines = [
		"# 翻譯引擎比較報告",
		"",
		f"- 產生時間：{time.strftime('%Y-%m-%d %H:%M:%S')}",
		f"- 模式：{'即時字幕（逐句 + 前一句上下文）' if mode == 'live' else '預先轉錄（編號批次）'}",
		f"- 樣本數：{len(samples)}",
		"",
		"## 摘要",
		"",
		"| 引擎 | 問題率 | 問題類型 | 術語命中率 | 延遲 p50 | 延遲 p95 |",
		"| --- | --- | --- | --- | --- | --- |",
	]
	for summary in summaries.values():
		glossary = "—" if summary["glossary_rate"] is None else f"{summary['glossary_rate']:.1%}（{summary['glossary_total']} 處）"
		problems = "、".join(f"{kind} {count}" for kind, count in sorted(summary["problems"].items())) or "無"
		lines.append(f"| {summary['name']} | {summary['problem_rate']:.1%} | {problems} | {glossary} | {summary['latency_p50']:.2f}s | {summary['latency_p95']:.2f}s |")
	if "gemini" in runs and "local" in runs:
		similarity = statistics.mean(chrf(local, gemini) for local, gemini in zip(runs["local"].outputs, runs["gemini"].outputs)) if samples else 0.0
		lines += ["", f"本機譯文與 Gemini 譯文的平均 chrF 相似度：{similarity:.1f}（只代表相似程度，不代表好壞）"]
	if gate is not None:
		passed, reasons = gate
		lines += ["", f"## 結論：{'PASS — 本機模型不低於 Gemini' if passed else 'FAIL — 本機模型低於 Gemini'}", ""]
		lines += [f"- {reason}" for reason in reasons]
	if judged is not None:
		lines += ["", "盲評由 Gemini 擔任，A/B 順序隨機；若評審偏好自己的譯文，只會讓本機模型更難通過。"]
	lines += ["", "## 逐句比較", ""]
	names = list(runs)
	header = "| # | 日文 | " + " | ".join(names) + (" | 盲評 |" if verdicts else " |")
	lines += [header, "| --- | --- | " + " | ".join("---" for _ in names) + (" | --- |" if verdicts else " |")]
	labels = {"win": "本機勝", "loss": "Gemini 勝", "tie": "平手", "skip": "略過"}
	for index, sentence in enumerate(samples):
		cells = [_cell(runs[name].outputs[index]) for name in names]
		row = f"| {index + 1} | {_cell(sentence)} | " + " | ".join(cells)
		lines.append(row + (f" | {labels[verdicts[index]]} |" if verdicts else " |"))
	return "\n".join(lines) + "\n"


# ---------- command line ----------

def _default_translator_factory(backend: str):
	from live_caption_gemini import Translator
	return Translator(backend)


def _default_glossary():
	from live_caption_gemini import load_glossary
	return load_glossary()


def main(argv=None, *, translator_factory=None, glossary_loader=None, offline_module=None, sleep=time.sleep) -> int:
	parser = argparse.ArgumentParser(description="比較 Gemini 與本機語言模型的翻譯結果")
	parser.add_argument("samples", help="日文樣本檔（一行一句，或 transcripts/transcript_*.txt）")
	parser.add_argument("--mode", choices=("live", "offline"), default="live")
	parser.add_argument("--backends", default="gemini,local", help="要比較的引擎，逗號分隔（預設 gemini,local）")
	parser.add_argument("--glossary", help="術語表 JSON（預設讀 glossary.json）")
	parser.add_argument("--limit", type=int, default=None, help="只取前 N 句")
	parser.add_argument("--judge", action="store_true", help="請 Gemini 盲評每一句（需要 gemini 與 local 都有跑）")
	parser.add_argument("--gemini-interval", type=float, default=4.5, help="Gemini 每次呼叫之間等待的秒數（避免撞到每分鐘額度）")
	parser.add_argument("--seed", type=int, default=1234, help="盲評 A/B 排序的亂數種子")
	parser.add_argument("--max-problem-gap", type=float, default=0.02)
	parser.add_argument("--max-glossary-gap", type=float, default=0.02)
	parser.add_argument("--min-judge-score", type=float, default=0.45)
	parser.add_argument("--out", help="報告輸出路徑（預設：樣本檔名_backend_report.md）")
	parser.add_argument("--json", dest="json_path", help="另外輸出 JSON 結果")
	args = parser.parse_args(argv)

	samples_path = Path(args.samples)
	samples = load_samples(samples_path, args.limit)
	if not samples:
		print(f"{samples_path} 裡沒有可用的日文句子。")
		return 2
	backends = [name.strip().lower() for name in args.backends.split(",") if name.strip()]
	if not backends or any(name not in ("gemini", "local") for name in backends):
		print("--backends 只能是 gemini、local 或 gemini,local")
		return 2
	if args.judge and set(backends) != {"gemini", "local"}:
		print("--judge 需要同時比較 gemini 與 local。")
		return 2

	if args.glossary:
		glossary = json.loads(Path(args.glossary).read_text(encoding="utf-8"))
	else:
		glossary = (glossary_loader or _default_glossary)()
	translator_factory = translator_factory or _default_translator_factory

	runs, translators = {}, {}
	for name in backends:
		print(f"=== {name}：{len(samples)} 句（{args.mode}）===")
		translators[name] = translator_factory(name)
		pace = args.gemini_interval if name == "gemini" else 0.0
		if args.mode == "live":
			runs[name] = run_live(name, translators[name], samples, glossary, pace_seconds=pace, retry_empty=2 if name == "gemini" else 0, sleep=sleep)
		else:
			runs[name] = run_offline(name, translators[name], samples, glossary, offline_module=offline_module)

	summaries = {name: summarize(run, samples, glossary) for name, run in runs.items()}
	verdicts, judged = [], None
	if args.judge:
		print("=== Gemini 盲評 ===")
		verdicts = judge_pairs(
			translators["gemini"]._chat, samples, runs["gemini"].outputs, runs["local"].outputs,
			rng=random.Random(args.seed), pace_seconds=args.gemini_interval, sleep=sleep,
		)
		judged = judge_stats(verdicts)
	gate = None
	if "gemini" in summaries and "local" in summaries:
		gate = evaluate_gate(
			summaries["gemini"], summaries["local"], judged,
			max_problem_gap=args.max_problem_gap, max_glossary_gap=args.max_glossary_gap, min_judge_score=args.min_judge_score,
		)

	report = render_report(samples, runs, summaries, verdicts, judged, gate, args.mode)
	out_path = Path(args.out) if args.out else samples_path.with_name(samples_path.stem + "_backend_report.md")
	out_path.write_text(report, encoding="utf-8")
	print(f"報告：{out_path}")
	if args.json_path:
		payload = {
			"mode": args.mode, "samples": samples, "outputs": {name: run.outputs for name, run in runs.items()},
			"summaries": summaries, "verdicts": verdicts, "judge": judged,
			"gate": None if gate is None else {"passed": gate[0], "checks": gate[1]},
		}
		Path(args.json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
	if gate is not None:
		print("PASS：本機模型不低於 Gemini" if gate[0] else "FAIL：本機模型低於 Gemini")
		for reason in gate[1]:
			print(f"  {reason}")
		return 0 if gate[0] else 1
	for summary in summaries.values():
		print(f"  {summary['name']}：問題率 {summary['problem_rate']:.1%}，延遲 p50 {summary['latency_p50']:.2f}s")
	return 0 if all(summary["problem_rate"] == 0 for summary in summaries.values()) else 1


if __name__ == "__main__":
	sys.exit(main())
