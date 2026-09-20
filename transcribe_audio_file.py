"""
transcribe_audio_file.py
把離線音檔（例如用瀏覽器擴充套件之類的工具下載下來的廣播節目錄音）整段
轉錄成帶精確時間戳記的中日對照字幕檔，給 live_caption_gemini.py 的
「讀取預先轉錄字幕」模式使用。

適合純語音、內容已經完整存在（時差重播、自己先下載好）的節目——這種情況下
不用受即時處理的延遲限制，可以讓 Whisper 看得到整段上下文重新辨識、翻譯
也能分批看整段脈絡處理，準確度比即時逐句處理時更好；正式收聽時字幕也可以
幾乎零延遲顯示，因為所有運算都已經在這一步事先做完了。

用法：
    python transcribe_audio_file.py 音檔路徑.mp3 [--title "節目名稱"]

    會在音檔同一個資料夾產生：
      <音檔檔名>_cues.json        給 live_caption_gemini.py 讀取用的時間軸字幕檔
      <音檔檔名>_transcript.md    給你自己看的可讀逐字稿（帶時間戳記）

音檔格式：mp3/wav/m4a 等 ffmpeg 支援的格式都可以（faster-whisper 內部靠
ffmpeg 讀取）。本機語言模型連線設定沿用 live_caption_gemini.py；不需要
Google SDK 或 Gemini API 金鑰。
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

from live_caption_gemini import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE,
    load_glossary,
    glossary_hint,
    Translator,
    _HALLUCINATION_DENYLIST,
)

_CHUNK_LINES = 40  # 每批送幾句給翻譯模型，避免單次回應太長


def _parse_numbered_translations(text: str, expected: int) -> list[str]:
    """解析翻譯模型回傳的「1. xxx / 2. xxx」這種帶編號清單，抓不到對應編號的
    就補空字串，確保回傳長度一定等於 expected，後面才能安全地照順序對應回
    原本的句子，不會因為模型輸出格式跑掉就整批對錯位。
    """
    result = [""] * expected
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.\、\)]\s*(.+)", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < expected:
            result[idx] = m.group(2).strip()
    return result


def transcribe_audio_file(audio_path: Path, episode_title: str = ""):
    """回傳 (cues_path, transcript_path)；沒辨識到任何內容則回傳 None。"""
    print(f"載入語音辨識模型（Whisper {WHISPER_MODEL_SIZE}）...")
    asr = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

    glossary = load_glossary()
    initial_prompt = "、".join(glossary.keys()) if glossary else None
    if glossary:
        print(f"已載入 {len(glossary)} 個專有名詞：{'、'.join(glossary.keys())}")

    print(f"正在辨識音檔：{audio_path}（沒有即時限制，可能要花一些時間）...")
    segments, _info = asr.transcribe(
        str(audio_path),
        language="ja",
        vad_filter=True,   # 交給 Whisper 內建 VAD 處理整段音訊的斷句，不用自己切
        beam_size=WHISPER_BEAM_SIZE,
        initial_prompt=initial_prompt,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )

    cues = []  # [{"start": float, "end": float, "ja": str, "zh": str}]
    for seg in segments:
        text = seg.text.strip()
        if not text or text in _HALLUCINATION_DENYLIST:
            continue
        cues.append({"start": seg.start, "end": seg.end, "ja": text})
        if len(cues) % 20 == 0:
            print(f"  已辨識 {len(cues)} 句（目前到 {seg.end:.0f} 秒）...")

    if not cues:
        print("沒有辨識到任何內容，結束。")
        return None

    print(f"辨識完成，共 {len(cues)} 句。載入翻譯模型...")
    translator = Translator()

    system_prompt = (
        "你是專業的日文-繁體中文口譯。使用者會分批貼上同一集日文廣播節目的逐句"
        "語音辨識稿，每句前面有編號。請針對這一批內容，把每一句翻成自然流暢、"
        "口語化的繁體中文，不要逐字直譯，也請忠實呈現內容、不要自己刪減語氣詞。\n"
        "輸出格式：每行「編號. 翻譯內容」，編號要跟原文一一對應，總共幾句就輸出"
        "幾行，不要合併、不要跳號、不要加多餘的說明或標題。"
    )
    hint = glossary_hint(glossary) if glossary else ""
    if hint:
        system_prompt += "\n" + hint

    context_tail = ""
    total_batches = (len(cues) + _CHUNK_LINES - 1) // _CHUNK_LINES
    for batch_no, i in enumerate(range(0, len(cues), _CHUNK_LINES), start=1):
        batch = cues[i:i + _CHUNK_LINES]
        numbered = "\n".join(f"{n + 1}. {c['ja']}" for n, c in enumerate(batch))
        user_prompt = (
            (f"（前面幾句當上下文參考，不用重複翻譯：\n{context_tail}\n\n") if context_tail else ""
        ) + f"請翻譯這一批句子（共 {len(batch)} 句）：\n{numbered}"

        # This offline batch keeps its caller-level retries because completion matters
        # more than latency; the transport itself deliberately performs no retries.
        result = ""
        for attempt in range(3):
            result = translator._chat(system_prompt, user_prompt)
            if result:
                break
            print(f"  第 {batch_no} 批翻譯失敗，20 秒後重試（第 {attempt + 1} 次）...")
            time.sleep(20)

        translations = _parse_numbered_translations(result, len(batch)) if result else [""] * len(batch)
        for c, zh in zip(batch, translations):
            c["zh"] = zh

        context_tail = "\n".join(c["ja"] for c in batch[-2:])
        print(f"  已翻譯第 {batch_no}/{total_batches} 批")
        if batch_no < total_batches:
            time.sleep(4.5)  # Preserve pacing between long offline requests.

    cues_path = audio_path.with_name(audio_path.stem + "_cues.json")
    cues_path.write_text(
        json.dumps({"episode_title": episode_title, "cues": cues}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    transcript_path = audio_path.with_name(audio_path.stem + "_transcript.md")
    lines = [
        f"# {episode_title}" if episode_title else "# 中日對照逐字稿（預先轉錄版）",
        "",
        f"來源音檔：{audio_path.name}",
        f"轉錄時間：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
    ]
    for c in cues:
        m, s = divmod(int(c["start"]), 60)
        h, m = divmod(m, 60)
        lines.append(f"**[{h:02d}:{m:02d}:{s:02d}]**")
        lines.append(c["ja"])
        lines.append(c.get("zh", ""))
        lines.append("")
    transcript_path.write_text("\n".join(lines), encoding="utf-8")

    print("完成！")
    print(f"  時間軸字幕檔：{cues_path}")
    print(f"  可讀逐字稿：{transcript_path}")
    return cues_path, transcript_path


def main():
    parser = argparse.ArgumentParser(description="把離線音檔轉錄成帶時間戳記的中日對照字幕檔")
    parser.add_argument("audio_path", help="音檔路徑（mp3/wav/m4a 等）")
    parser.add_argument("--title", default="", help="節目名稱（選填）")
    args = parser.parse_args()

    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        print(f"找不到檔案：{audio_path}")
        sys.exit(1)

    transcribe_audio_file(audio_path, episode_title=args.title)


if __name__ == "__main__":
    main()
