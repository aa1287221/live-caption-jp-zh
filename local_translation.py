"""Local-model translation that produces the same artifacts as the Gemini backend.

The Gemini backend sends a system prompt and a user prompt and trusts the reply. The local
backend sends the very same prompts (built by the same functions in the caller), then adds
the guard rails a smaller model needs to deliver the same results:

- live captions: one fast attempt with a short timeout, a validity check (empty reply, echoed
  prompt, untranslated Japanese, runaway length) and one repair attempt without context;
- batch workflows: smaller batches that split in half when a reply is truncated, too large
  for the server context, or malformed; numbered gaps are re-requested instead of left blank;
- a circuit breaker so a stopped server fails fast instead of stalling captions or jobs.

Translation-only models (Riva-Translate) cannot follow those instructions, so the ``riva``
profile uses Ontime's plain per-sentence prompt, masks glossary terms and splits long input.
Prompts and model output are never logged.
"""

import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

from local_llm import MODE_RIVA, RIVA_TRANSLATION_INSTRUCTION, LocalLLMClient, LocalLLMError


TRANSLATION_FAILED_TEXT = "（翻譯失敗）"

_KANA_RE = re.compile(r"[぀-ゟ゠-ヿㇰ-ㇿｦ-ﾟ]")
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
_PROMPT_ECHO_MARKERS = (
	"前一句話", "僅供參考上下文", "請翻譯這一句", "請處理這一批句子", "請翻譯這一批句子", "不用重複翻譯",
)
_LINE_LABEL_RE = re.compile(
	r"^\s*(?:[-*]\s*)?(?:日文原句|日文|中文翻譯|中文|原文|譯文|翻譯|日本語|JA|JP|ZH)\s*[：:]\s*",
	re.IGNORECASE,
)
_GLOSSARY_MARKER_RE = re.compile(r"__GLOSSARY_\d+__")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[。！？!?…\n])")
_CLAUSE_BREAK_RE = re.compile(r"(?<=[、，,])")


class LocalServiceUnavailable(RuntimeError):
	"""Raised inside batch workflows when the local server stops answering."""


# ---------- configuration helpers ----------

def _env_number(value, env_name: str, default, parse, *, allow_zero: bool = False):
	raw = value if value is not None else os.environ.get(env_name)
	if raw is None or (isinstance(raw, str) and not raw.strip()):
		return default
	try:
		if isinstance(raw, bool):
			raise ValueError
		number = parse(raw.strip() if isinstance(raw, str) else raw)
	except (TypeError, ValueError, OverflowError) as error:
		raise LocalLLMError(f"{env_name} 設定無效：請指定正數。", kind="config") from error
	if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
		raise LocalLLMError(f"{env_name} 設定無效：請指定正數。", kind="config")
	return number


# ---------- text checks ----------

def _glossary_terms(glossary: dict | None) -> list[str]:
	terms = set()
	for key, value in (glossary or {}).items():
		for term in (key, value):
			if isinstance(term, str) and term.strip():
				terms.add(term)
	return sorted(terms, key=len, reverse=True)


def translation_problem(text: str, source: str, glossary: dict | None = None) -> str | None:
	"""Return why ``text`` is not a usable Chinese rendering of ``source``, or None."""
	if not isinstance(text, str) or not text.strip():
		return "empty"
	if any(marker in text for marker in _PROMPT_ECHO_MARKERS):
		return "echo"
	if _GLOSSARY_MARKER_RE.search(text):
		return "marker"
	body = text
	for term in _glossary_terms(glossary):
		body = body.replace(term, "")
	kana = len(_KANA_RE.findall(body))
	cjk = len(_CJK_RE.findall(body))
	if kana >= 3 and kana > cjk:
		return "untranslated"
	if len(text) > 4 * len(source.strip()) + 40:
		return "verbose"
	return None


def parse_bilingual_pairs(text: str) -> list[tuple[str, str]]:
	"""Parse "日文\\n中文" blocks separated by blank lines into (ja, zh) pairs.

	Blocks that lost their blank-line separators are re-split on lines containing kana.
	A block without a Chinese line yields a pair with an empty translation.
	"""
	pairs = []
	for block in re.split(r"\n\s*\n", text.strip()):
		lines = []
		for line in block.splitlines():
			line = _LINE_LABEL_RE.sub("", line).strip()
			if line and line != "---" and not line.startswith("#"):
				lines.append(line)
		if not lines:
			continue
		if len(lines) == 2:
			pairs.append((lines[0], lines[1]))
			continue
		regrouped = []
		for line in lines:
			if _KANA_RE.search(line) or not regrouped:
				regrouped.append([line, []])
			else:
				regrouped[-1][1].append(line)
		if len(regrouped) > 1 and all(zh for _, zh in regrouped):
			pairs.extend((ja, "".join(zh)) for ja, zh in regrouped)
		else:
			pairs.append((lines[0], "".join(lines[1:])))
	return pairs


def bilingual_pairs_acceptable(pairs: list[tuple[str, str]], batch: list[str], glossary: dict | None) -> bool:
	"""Accept a reply that covers the batch; ASR correction may merge or split a few lines."""
	count = len(batch)
	if not pairs or len(pairs) < max(1, math.ceil(count * 0.75)) or len(pairs) > 2 * count + 2:
		return False
	bad = sum(1 for ja, zh in pairs if not ja.strip() or translation_problem(zh, ja, glossary) is not None)
	return bad <= count // 10


# ---------- glossary protection for translation-only models ----------

@dataclass
class MaskedText:
	text: str
	targets: dict
	expected: Counter


def mask_glossary(text: str, glossary: dict | None) -> MaskedText:
	"""Replace glossary source terms with atomic markers, longest term first."""
	terms = sorted((term for term in (glossary or {}) if isinstance(term, str) and term), key=len, reverse=True)
	if not terms or "__GLOSSARY_" in text:
		return MaskedText(text, {}, Counter())
	pattern = re.compile("|".join(re.escape(term) for term in terms))
	markers = {}
	targets = {}

	def replace(match):
		term = match.group(0)
		marker = markers.setdefault(term, f"__GLOSSARY_{len(markers)}__")
		target = glossary.get(term)
		targets[marker] = target if isinstance(target, str) and target.strip() else term
		return marker

	masked = pattern.sub(replace, text)
	return MaskedText(masked, targets, Counter(_GLOSSARY_MARKER_RE.findall(masked)))


def restore_glossary(translated: str, masked: MaskedText) -> str | None:
	"""Put configured targets back; None when a marker was lost, duplicated or invented."""
	if not masked.targets:
		return translated
	if Counter(_GLOSSARY_MARKER_RE.findall(translated)) != masked.expected:
		return None
	return _GLOSSARY_MARKER_RE.sub(lambda match: masked.targets[match.group(0)], translated)


def apply_glossary_targets(text: str, glossary: dict | None) -> str:
	"""Map source terms the model kept verbatim to their configured display text."""
	for term in sorted((key for key in (glossary or {}) if isinstance(key, str) and key), key=len, reverse=True):
		target = glossary.get(term)
		if isinstance(target, str) and target.strip() and target != term:
			text = text.replace(term, target)
	return text


def _protected_spans(text: str, glossary: dict | None) -> list[tuple[int, int]]:
	spans = []
	for term in _glossary_terms({key: key for key in (glossary or {})}):
		start = text.find(term)
		while start >= 0:
			spans.append((start, start + len(term)))
			start = text.find(term, start + 1)
	return spans


def _safe_cut(text: str, limit: int, glossary: dict | None) -> int:
	"""Largest cut position <= limit that does not split a glossary term."""
	cut = limit
	for start, end in _protected_spans(text, glossary):
		if start < cut < end:
			cut = start if start > 0 else end
	return max(1, cut)


def split_for_translation(text: str, max_chars: int, glossary: dict | None = None) -> list[str]:
	"""Split long input at sentence, clause, then size boundaries, keeping glossary terms whole."""
	text = text.strip()
	if len(text) <= max_chars:
		return [text] if text else []
	pieces = []
	for sentence in (part for part in _SENTENCE_BREAK_RE.split(text) if part.strip()):
		if len(sentence) <= max_chars:
			pieces.append(sentence)
			continue
		for clause in (part for part in _CLAUSE_BREAK_RE.split(sentence) if part.strip()):
			while len(clause) > max_chars:
				cut = _safe_cut(clause, max_chars, glossary)
				pieces.append(clause[:cut])
				clause = clause[cut:]
			if clause.strip():
				pieces.append(clause)
	merged = []
	for piece in pieces:
		if merged and len(merged[-1]) + len(piece) <= max_chars:
			merged[-1] += piece
		else:
			merged.append(piece)
	return [piece.strip() for piece in merged if piece.strip()]


# ---------- engine ----------

@dataclass
class BatchOutcome:
	results: list
	failed: int = 0
	aborted: bool = False
	reason: str = ""


class LocalTranslationEngine:
	"""Local-model calls with the validation and pacing the translation workflows rely on."""

	RETRY_DELAYS = (2.0, 5.0)

	def __init__(
		self,
		client: LocalLLMClient | None = None,
		*,
		live_timeout=None,
		batch_lines=None,
		batch_max_tokens=None,
		startup_wait=None,
		segment_chars=None,
		context_lines=None,
		max_consecutive_failures: int = 3,
		cooldown_seconds: float = 30.0,
		postprocess=None,
		clock=time.monotonic,
		sleep=time.sleep,
		log=print,
	) -> None:
		self.client = client if client is not None else LocalLLMClient()
		self.live_timeout = float(_env_number(live_timeout, "LOCAL_LLM_LIVE_TIMEOUT", 15.0, float))
		self.batch_lines = int(_env_number(batch_lines, "LOCAL_LLM_BATCH_LINES", 20, int))
		self.batch_max_tokens = int(_env_number(batch_max_tokens, "LOCAL_LLM_BATCH_MAX_TOKENS", 4096, int))
		self.startup_wait = float(_env_number(startup_wait, "LOCAL_LLM_STARTUP_WAIT", 120.0, float, allow_zero=True))
		self.segment_chars = int(_env_number(segment_chars, "LOCAL_LLM_SEGMENT_CHARS", 120, int))
		# Local batches are smaller than Gemini's, so each one carries more preceding lines.
		self.context_lines = int(_env_number(context_lines, "LOCAL_LLM_CONTEXT_LINES", 8, int, allow_zero=True))
		self.max_consecutive_failures = max_consecutive_failures
		self.cooldown_seconds = cooldown_seconds
		self.postprocess = postprocess or (lambda text: text)
		self.clock = clock
		self.sleep = sleep
		self.log = log
		self.last_error: LocalLLMError | None = None
		self._live_failures = 0
		self._live_paused_until = 0.0
		self._batch_failures = 0

	@property
	def profile(self) -> str:
		return self.client.mode

	@property
	def is_translation_only(self) -> bool:
		return self.profile == MODE_RIVA

	# ---------- startup ----------

	def prepare(self, warmup_system_prompt: str, warmup_user_prompt: str, warmup_source: str) -> None:
		"""Wait for the server, pick the request style and prove one translation works."""
		self.log(f"連線本機語言模型服務：{self.client.base_url}")
		notices = []

		def on_wait(error):
			if not notices:
				self.log(f"等待本機模型服務就緒（最多 {self.startup_wait:g} 秒）：{error}")
			notices.append(error)

		self.client.ensure_ready(self.startup_wait, poll_interval=2.0, on_wait=on_wait, clock=self.clock, sleep=self.sleep)
		# Resolving the mode reads /props, which also fills in server_info.
		self.client.mode
		info = self.client.server_info
		model_name = self.client.model or info.get("model_alias") or info.get("model_path") or "伺服器目前載入的模型"
		if self.is_translation_only:
			self.log(f"本機模型模式：riva（翻譯專用模型，逐句翻譯）｜模型：{model_name}")
			self.log(
				"提醒：翻譯專用模型不會參考前一句，也不會校正辨識錯字或潤稿；"
				"要跟 Gemini 同樣的上下文翻譯與潤稿效果，請載入通用指令模型（見 README）。"
			)
		else:
			self.log(f"本機模型模式：instruct（與 Gemini 相同的提示詞與流程）｜模型：{model_name}")
			n_ctx = info.get("n_ctx")
			if isinstance(n_ctx, int) and 0 < n_ctx < 4096:
				self.log(f"提醒：本機模型 context 只有 {n_ctx}，事後整理會自動改用較小批次；建議用 -c 8192 以上啟動。")
		started = self.clock()
		if self.is_translation_only:
			sample = self._plain_translate(warmup_source, None, timeout=self.client.timeout, batch=False)
		else:
			sample = self.postprocess(self.client.chat(warmup_system_prompt, warmup_user_prompt, timeout=self.client.timeout))
		self.log(f"本機模型已就緒（暖機 {self.clock() - started:.1f} 秒）：{warmup_source} → {sample}")

	# ---------- live captions ----------

	def live_chat(self, system_prompt: str, user_prompt: str) -> str:
		"""One live attempt: never sleeps or retries, returns "" on failure."""
		if self.clock() < self._live_paused_until:
			return ""
		try:
			text = self.client.chat(system_prompt, user_prompt, timeout=self.live_timeout)
		except LocalLLMError as error:
			self._live_failed(error)
			return ""
		self._live_succeeded()
		return self.postprocess(text).strip()

	def translate_live(
		self,
		system_prompt: str,
		user_prompt: str,
		source: str,
		glossary: dict | None,
		fallback_user_prompt: str | None = None,
	) -> str:
		"""Translate one caption; repair a bad reply once without the context sentence."""
		if self.is_translation_only:
			if self.clock() < self._live_paused_until:
				return ""
			try:
				text = self._plain_translate(source, glossary, timeout=self.live_timeout, batch=False)
			except LocalLLMError as error:
				self._live_failed(error)
				return ""
			self._live_succeeded()
			return text if translation_problem(text, source, glossary) is None else ""
		first = self.live_chat(system_prompt, user_prompt)
		problem = translation_problem(first, source, glossary)
		if problem is None:
			return first
		if not first and (self.last_error is None or self.last_error.unreachable):
			# The server is slow or gone; a second request would only delay later captions.
			return ""
		if fallback_user_prompt is None:
			return first if problem not in ("echo", "marker") else ""
		second = self.live_chat(system_prompt, fallback_user_prompt)
		if translation_problem(second, source, glossary) is None:
			return second
		# Prefer a non-empty first reply over a blank caption unless it echoed the prompt.
		return first if first and problem not in ("echo", "marker") else ""

	def _live_failed(self, error: LocalLLMError) -> None:
		self.last_error = error
		self.log(f"本機模型翻譯失敗：{error}")
		if not error.unreachable:
			return
		self._live_failures += 1
		if self._live_failures >= self.max_consecutive_failures:
			self._live_failures = 0
			self._live_paused_until = self.clock() + self.cooldown_seconds
			self.log(f"本機模型連續 {self.max_consecutive_failures} 次沒有回應，先暫停翻譯 {self.cooldown_seconds:g} 秒，之後自動重試。")

	def _live_succeeded(self) -> None:
		self.last_error = None
		self._live_failures = 0

	# ---------- batch requests ----------

	def batch_chat(
		self,
		system_prompt: str,
		user_prompt: str,
		*,
		max_tokens: int | None = None,
		splittable: bool = False,
	) -> str:
		"""Offline request: brief retries for transient errors, circuit breaker for a dead server.

		With ``splittable`` a timeout is raised at once so the caller can resend a smaller
		batch; a slow model is not the same as a dead server. Consecutive failures of any
		retryable kind still open the circuit breaker.
		"""
		attempt = 0
		while True:
			try:
				text = self.client.chat(system_prompt, user_prompt, max_tokens=max_tokens, timeout=self.client.timeout)
			except LocalLLMError as error:
				self.last_error = error
				if error.retryable:
					self._batch_failures += 1
					if self._batch_failures >= self.max_consecutive_failures:
						raise LocalServiceUnavailable(f"本機模型服務持續沒有回應：{error}") from error
					if error.kind == "timeout" and splittable:
						raise
					if attempt < len(self.RETRY_DELAYS):
						delay = self.RETRY_DELAYS[attempt]
						self.log(f"  本機模型暫時沒有回應，{delay:g} 秒後重試：{error}")
						self.sleep(delay)
						attempt += 1
						continue
				raise
			self._batch_failures = 0
			self.last_error = None
			return self.postprocess(text).strip()

	@staticmethod
	def _should_split(error: LocalLLMError, indices: list[int]) -> bool:
		"""Errors a smaller request can fix; anything else will not improve by splitting."""
		return error.too_large or error.kind in ("empty", "protocol") or (error.kind == "timeout" and len(indices) > 1)

	def _budget(self, batch: list[str], factor: float) -> int:
		chars = sum(len(text) for text in batch)
		budget = min(self.batch_max_tokens, max(256, int(chars * factor) + 24 * len(batch)))
		n_ctx = self.client.server_info.get("n_ctx")
		if isinstance(n_ctx, int) and n_ctx > 0:
			budget = min(budget, max(256, n_ctx // 2))
		return budget

	def _line_with_context(self, index: int, lines: list[str], glossary: dict | None, context_prompts) -> str:
		"""Translate one line with the live-caption prompts; "" when the reply is unusable."""
		previous = lines[index - 1] if index > 0 else ""
		system_prompt, user_prompt = context_prompts(previous, lines[index])
		try:
			text = self.batch_chat(system_prompt, user_prompt)
		except LocalServiceUnavailable:
			raise
		except LocalLLMError:
			return ""
		return text if translation_problem(text, lines[index], glossary) is None else ""

	# ---------- translation-only (Riva) profile ----------

	def _plain_call(self, source_text: str, *, timeout: float, batch: bool) -> str:
		user_prompt = f"{RIVA_TRANSLATION_INSTRUCTION} {source_text}"
		if batch:
			return self.batch_chat("", user_prompt)
		return self.postprocess(self.client.chat("", user_prompt, timeout=timeout)).strip()

	def _plain_segment(self, segment: str, glossary: dict | None, *, timeout: float, batch: bool, depth: int = 0) -> str:
		masked = mask_glossary(segment, glossary)
		try:
			raw = self._plain_call(masked.text, timeout=timeout, batch=batch)
		except LocalLLMError as error:
			if error.too_large and depth < 4 and len(segment) > 20:
				cut = _safe_cut(segment, len(segment) // 2, glossary)
				return (
					self._plain_segment(segment[:cut], glossary, timeout=timeout, batch=batch, depth=depth + 1)
					+ self._plain_segment(segment[cut:], glossary, timeout=timeout, batch=batch, depth=depth + 1)
				)
			raise
		restored = restore_glossary(raw, masked)
		if restored is None:
			# The model lost or duplicated a marker: translate unmasked and map the terms afterwards.
			restored = apply_glossary_targets(self._plain_call(segment, timeout=timeout, batch=batch), glossary)
		return restored

	def _plain_translate(self, text: str, glossary: dict | None, *, timeout: float, batch: bool) -> str:
		pieces = [
			self._plain_segment(segment, glossary, timeout=timeout, batch=batch)
			for segment in split_for_translation(text, self.segment_chars, glossary)
		]
		return "".join(pieces)

	def _plain_line(self, text: str, glossary: dict | None) -> str:
		if not text.strip():
			return ""
		try:
			translated = self._plain_translate(text, glossary, timeout=self.client.timeout, batch=True)
		except LocalServiceUnavailable:
			raise
		except LocalLLMError:
			return ""
		return translated if translation_problem(translated, text, glossary) is None else ""

	# ---------- numbered batches (offline pre-transcription) ----------

	def translate_numbered(
		self,
		texts: list[str],
		*,
		glossary: dict | None,
		system_prompt: str,
		build_user_prompt,
		parse_numbered,
		context_prompts,
		tail_lines: int | None = None,
		progress=None,
	) -> BatchOutcome:
		"""Translate ``texts`` one-to-one with the caller's numbered-batch prompts."""
		tail_lines = self.context_lines if tail_lines is None else tail_lines
		results = [""] * len(texts)
		outcome = BatchOutcome(results)
		total = (len(texts) + self.batch_lines - 1) // self.batch_lines
		try:
			for batch_no, start in enumerate(range(0, len(texts), self.batch_lines), start=1):
				indices = list(range(start, min(start + self.batch_lines, len(texts))))
				if self.is_translation_only:
					for index in indices:
						results[index] = self._plain_line(texts[index], glossary)
				else:
					self._numbered_chunk(
						indices, texts, results, glossary, system_prompt, build_user_prompt,
						parse_numbered, context_prompts, tail_lines,
					)
				if progress is not None:
					progress(batch_no, total)
		except LocalServiceUnavailable as error:
			outcome.aborted = True
			outcome.reason = str(error)
		outcome.failed = sum(1 for text, zh in zip(texts, results) if text.strip() and not zh)
		return outcome

	def _numbered_chunk(
		self, indices, texts, results, glossary, system_prompt, build_user_prompt,
		parse_numbered, context_prompts, tail_lines,
	) -> None:
		batch = [texts[index] for index in indices]
		context_tail = "\n".join(texts[max(0, indices[0] - tail_lines):indices[0]])
		parsed = None
		try:
			reply = self.batch_chat(
				system_prompt, build_user_prompt(batch, context_tail),
				max_tokens=self._budget(batch, 2.0), splittable=len(indices) > 1,
			)
			parsed = parse_numbered(reply, len(batch))
		except LocalServiceUnavailable:
			raise
		except LocalLLMError as error:
			if not self._should_split(error, indices):
				return
		if parsed is None:
			if len(indices) > 1:
				middle = len(indices) // 2
				for part in (indices[:middle], indices[middle:]):
					self._numbered_chunk(part, texts, results, glossary, system_prompt, build_user_prompt, parse_numbered, context_prompts, tail_lines)
				return
			parsed = [""]
		missing = []
		for index, text in zip(indices, parsed):
			if translation_problem(text, texts[index], glossary) is None:
				results[index] = text
			else:
				missing.append(index)
		if not missing:
			return
		if len(indices) == 1:
			results[indices[0]] = self._line_with_context(indices[0], texts, glossary, context_prompts)
		elif len(missing) < len(indices):
			self._numbered_chunk(missing, texts, results, glossary, system_prompt, build_user_prompt, parse_numbered, context_prompts, tail_lines)
		else:
			middle = len(indices) // 2
			for part in (indices[:middle], indices[middle:]):
				self._numbered_chunk(part, texts, results, glossary, system_prompt, build_user_prompt, parse_numbered, context_prompts, tail_lines)

	# ---------- bilingual reconstruction (after playback) ----------

	def polish_bilingual(
		self,
		ja_lines: list[str],
		*,
		glossary: dict | None,
		system_prompt: str,
		build_user_prompt,
		context_prompts,
		tail_lines: int | None = None,
		progress=None,
	) -> list[str] | None:
		"""Return rendered "日文\\n中文" blocks per batch, or None if the server went away."""
		tail_lines = self.context_lines if tail_lines is None else tail_lines
		parts = []
		total = (len(ja_lines) + self.batch_lines - 1) // self.batch_lines
		try:
			for batch_no, start in enumerate(range(0, len(ja_lines), self.batch_lines), start=1):
				indices = list(range(start, min(start + self.batch_lines, len(ja_lines))))
				if self.is_translation_only:
					pairs = [(ja_lines[index], self._plain_line(ja_lines[index], glossary) or TRANSLATION_FAILED_TEXT) for index in indices]
				else:
					pairs = self._bilingual_chunk(indices, ja_lines, glossary, system_prompt, build_user_prompt, context_prompts, tail_lines)
				parts.append("\n\n".join(f"{ja}\n{zh}" for ja, zh in pairs))
				if progress is not None:
					progress(batch_no, total)
		except LocalServiceUnavailable as error:
			self.log(f"本機模型服務中斷，停止整理：{error}")
			return None
		return parts

	def _bilingual_chunk(self, indices, lines, glossary, system_prompt, build_user_prompt, context_prompts, tail_lines):
		batch = [lines[index] for index in indices]
		context_tail = "\n".join(lines[max(0, indices[0] - tail_lines):indices[0]])
		pairs = None
		try:
			reply = self.batch_chat(
				system_prompt, build_user_prompt(batch, context_tail),
				max_tokens=self._budget(batch, 3.0), splittable=len(indices) > 1,
			)
			candidate = parse_bilingual_pairs(reply)
			if bilingual_pairs_acceptable(candidate, batch, glossary):
				pairs = candidate
		except LocalServiceUnavailable:
			raise
		except LocalLLMError as error:
			if not self._should_split(error, indices):
				# A permanent request error will not improve by splitting; keep the source visible.
				return [(ja, TRANSLATION_FAILED_TEXT) for ja in batch]
		if pairs is not None:
			return pairs
		if len(indices) > 1:
			middle = len(indices) // 2
			return (
				self._bilingual_chunk(indices[:middle], lines, glossary, system_prompt, build_user_prompt, context_prompts, tail_lines)
				+ self._bilingual_chunk(indices[middle:], lines, glossary, system_prompt, build_user_prompt, context_prompts, tail_lines)
			)
		zh = self._line_with_context(indices[0], lines, glossary, context_prompts)
		return [(batch[0], zh or TRANSLATION_FAILED_TEXT)]
