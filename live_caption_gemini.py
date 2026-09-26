"""
live_caption_gemini.py
跟 live_caption.py 完全一樣的內建擷取版，翻譯引擎可以在啟動畫面「⑥ 翻譯引擎」選：
  - Gemini API（預設，GEMINI_MODEL）：雲端翻譯，需要 API 金鑰，行為跟以前完全一樣
  - 本機語言模型：呼叫你自己先啟動的 llama-server（或 Ollama 等 OpenAI 相容服務），
    不需要金鑰、不用額度、逐字稿不會離開你的電腦；送出的提示詞、前一句上下文、
    事後重新辨識+潤稿、預先轉錄的流程都跟 Gemini 版相同（設定方式見下方與 README）
檔名保留 gemini，既有的捷徑、啟動器、transcribe_audio_file.py 都不用改。

使用情境：
  你是合法訂閱會員，在瀏覽器裡播放影片。這支程式只會擷取你電腦上「你自己選定的
  那個視窗」的畫面跟「送到喇叭/耳機」的聲音，完全不會下載影片、不會存取網站帳號
  或繞過任何驗證機制，純粹是「延遲播放 + 即時聽打翻譯」的個人輔助工具。

  為什麼要延遲播放：翻譯需要時間，直接看即時畫面的話字幕一定會慢個幾秒才跳出來。
  這支程式會把畫面+聲音存進一個緩衝區，你看到的畫面其實是「N 秒前」的，這樣
  翻譯字幕就有時間在畫面播到那一刻之前先算好，變成精準對上這個人開口瞬間的字幕。

安裝（一次就好）：
    pip install -r requirements.txt
    pip install google-genai

    需要自己去 Google AI Studio（https://aistudio.google.com/apikey）申請一組
    免費的 Gemini API 金鑰，然後在終端機設定成環境變數再執行（金鑰不要寫進程式
    碼裡，也不要貼給任何人）：

        PowerShell（只在這次視窗有效）：
            $env:GEMINI_API_KEY = "你的金鑰"

        想每次開新視窗都自動生效，改用「系統管理」設定成永久的使用者環境變數，
        或是每次執行前都重新設定一次也可以。

    免費額度大約：gemini-2.5-flash-lite 每天 1,000 次、每分鐘 15 次請求，
    大概夠一天看一集左右的節目份量，實際額度以 Google 官方頁面當下顯示為準。

    改用本機語言模型的話（不需要金鑰、不用 google-genai）：先在另一個視窗啟動
    llama-server 並載入模型，例如：
        llama-server -m 你的模型.gguf --host 127.0.0.1 --port 8766 -c 8192 -ngl 99
    再到啟動畫面把「⑥ 翻譯引擎」選成本機語言模型（會記住上次的選擇）。
    服務網址不是 http://127.0.0.1:8766 的話，用環境變數 LOCAL_LLM_BASE_URL 指定。
    想要跟 Gemini 一樣會看上下文、會校正辨識錯字的效果，請載入通用指令模型
    （例如 Qwen2.5-14B-Instruct）；Riva-Translate 這類翻譯專用模型會自動改成逐句翻譯。

執行：
    python live_caption_gemini.py

    一開始會跳出一個視窗清單，選你要擷取的視窗（通常是瀏覽器）。
    接著問節目名稱、延遲秒數（直接按 Enter 用預設的 6 秒），
    然後在瀏覽器正常播放影片即可。

    畫面會是這樣：延遲播放視窗最上面一排是操作按鈕，下面是延遲播放的畫面、
    再下面是字幕（日文原文 + 中文翻譯）；另外還有一個可以滾輪往回翻的完整逐字稿視窗。

    操作按鈕：
      ⏸ 暫停 / ▶ 繼續 —— 擷取、辨識、翻譯、播放全部一起暫停/繼續
      ⏺ 開始錄製 / ⏹ 停止錄製 —— 手動控制要不要錄下完整音訊（預設不錄）
      🔄 切換視窗 —— 執行中隨時可以換成擷取另一個視窗，不用整個重開程式

    完整逐字稿會即時寫進 transcripts/transcript_YYYYMMDD_HHMMSS.txt，
    結束播放後會自動整理成方便閱讀的 transcript_YYYYMMDD_HHMMSS_polished.md。

    如果有按過「開始錄製」，完整音訊會存成 transcript_YYYYMMDD_HHMMSS_audio.wav
    （播放期間暫存用），結束播放後自動用這份完整錄音重新辨識+翻譯一次（沒有
    即時限制、看得到完整上下文，準確度比即時逐句處理時更好，內容忠實呈現、
    不會自己刪減），產生 transcript_YYYYMMDD_HHMMSS_notebooklm_style.md——概念
    上就是手動把錄音丟 NotebookLM 轉錄、再丟 Gemini 潤稿那個流程，差別是全自動、
    不用你自己動手。這步會多花幾分鐘，結束播放後稍等一下。整理完成後那份 WAV
    錄音檔會自動刪除（本來就只是拿來重新辨識用，不是給人聽的，用完即刪）。
    沒按過「開始錄製」的話，這步會直接略過。

    要結束就關掉視窗，或在終端機按 Ctrl+C。

如果沒有跳出字幕、或提示找不到裝置：
    - 確認 Windows「音效設定」裡有正常的輸出裝置（喇叭/耳機）在播放聲音
    - 確認瀏覽器音量、系統音量都沒有靜音
    - 可以先用小音量的測試影片試跑，觀察終端機印出的裝置名稱是否正確

關於「聽起來有點小聲的背景音」：
    程式會自動把你選的來源視窗音量調低（不是靜音），這樣你主要聽到的是延遲
    播放出來的聲音，但還是會有一點點很小聲的即時背景音——這是刻意的，因為
    擷取聲音的技術需要那個聲音「有在播放」才抓得到，完全靜音的話程式會聽不到
    任何聲音。這個小聲的背景音通常不會很干擾，習慣了就好。
"""

import os
import sys
import re
import json
import time
import ctypes
import queue
import threading
import multiprocessing
from multiprocessing import shared_memory
import datetime
import wave
from collections import deque
from pathlib import Path

import numpy as np
import torch
from math import gcd
from scipy.signal import resample_poly
import cv2
import pyaudiowpatch as pyaudio
import sounddevice as sd
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk

from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, VADIterator
from local_translation import LocalTranslationEngine
from llama_server import LlamaServerError, ensure_llama_server

# The Gemini SDK is imported only when the Gemini backend is chosen, so local-model
# users do not need google-genai installed.
genai = None
genai_types = None
genai_errors = None


def _import_gemini_sdk():
	global genai, genai_types, genai_errors
	if genai is None:
		try:
			from google import genai as sdk
			from google.genai import types as sdk_types
			from google.genai import errors as sdk_errors
		except ImportError as error:
			raise RuntimeError(
				"找不到 google-genai 套件：請執行 pip install google-genai，或在啟動畫面改選本機語言模型。"
			) from error
		genai, genai_types, genai_errors = sdk, sdk_types, sdk_errors

# ---------- 硬體自動偵測 (5800X3D + RTX 5060 Ti 會自動吃 GPU) ----------
_CUDA_OK = torch.cuda.is_available()
if _CUDA_OK:
    print(f"偵測到 GPU：{torch.cuda.get_device_name(0)}，將使用 CUDA 加速。")
else:
    print("未偵測到可用的 CUDA GPU，將使用 CPU（速度會慢很多）。")

# ---------- 可調參數 ----------
def resolve_whisper_model_size(cuda_available: bool, environ: dict | None = None) -> str:
    """WHISPER_MODEL env override > CUDA/CPU default.

    large-v3-turbo 在有雜音的直播片段上會出現重複片語與罐頭字句的幻覺，實測 large-v3
    沒有這個問題（p50 每句 0.32 秒，可接受）；CPU 沒有這個問題所以維持較小的 small。
    """
    environ = os.environ if environ is None else environ
    override = environ.get("WHISPER_MODEL")
    if override and override.strip():
        return override.strip()
    return "large-v3" if cuda_available else "small"


WHISPER_MODEL_SIZE = resolve_whisper_model_size(_CUDA_OK)
WHISPER_DEVICE = "cuda" if _CUDA_OK else "cpu"
WHISPER_COMPUTE_TYPE = "float16" if _CUDA_OK else "int8"
WHISPER_BEAM_SIZE = 5             # 用 beam search 選最佳辨識結果，GPU 才有餘裕開這個

# 翻譯改呼叫雲端的 Gemini API，取代本機 Ollama + Qwen2.5：
# 雲端大型模型的語言理解能力明顯更好，翻起來更自然、更少怪異的字句，
# 代價是需要申請 API 金鑰、有免費額度限制、逐字稿內容會傳到 Google 的伺服器。
# 金鑰優先讀環境變數 GEMINI_API_KEY；沒有的話改讀 gemini_api_key.txt 這個本機檔案
# （給雙擊 exe 這種沒有終端機可以先設環境變數的用法）。兩種都不寫死在程式碼裡。
GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_API_KEY_FILE = Path(__file__).parent / "gemini_api_key.txt"

# 翻譯引擎：gemini（預設，跟以前一樣）或 local（本機語言模型，見檔案開頭說明）。
# 啟動畫面會記住上次的選擇；環境變數 TRANSLATION_BACKEND 可以指定預設值，
# transcribe_audio_file.py 也吃同一個環境變數（或用 --backend 參數）。
TRANSLATION_BACKEND_GEMINI = "gemini"
TRANSLATION_BACKEND_LOCAL = "local"
TRANSLATION_BACKENDS = (TRANSLATION_BACKEND_GEMINI, TRANSLATION_BACKEND_LOCAL)
TRANSLATION_BACKEND_LABELS = {
    TRANSLATION_BACKEND_GEMINI: "Gemini API（雲端，需要 API 金鑰）",
    TRANSLATION_BACKEND_LOCAL: "本機語言模型（llama-server，設定好會自動啟動，不需要金鑰）",
}

TARGET_SR = 16000                 # Whisper / VAD / 延遲音訊緩衝區使用的取樣率
VAD_CHUNK_SAMPLES = 512           # Silero VAD 在 16kHz 下要求的固定窗格大小 (32ms)
VAD_THRESHOLD = 0.5               # 0~1，語音機率超過這個值才算「有人在講話」
VAD_SPEECH_PAD_MS = 100           # 語句前後多保留一點音訊，避免咬字被切掉

# 這兩個值故意拉長：日文常有動詞放句尾、句中助詞換氣停頓的狀況，
# 停頓門檻太短會把一句話切成兩半，導致辨識/翻譯失去前後文法脈絡。
# 有 GPU 加速運算很快，多等一點點換取整句完整度是划算的。
SILENCE_END_MS = 700              # 停頓超過這麼久，才視為一句話講完，送去辨識
# 這個只是「安全上限」：平常斷句主要靠上面那個真正停頓的偵測，這個只有在
# 完全沒停頓、一直講很久的狀況下才會強制切斷。試過拉到 20 秒想改善念信這種
# 連續發言的斷句問題，但實測延遲感明顯變重（單一句子累積越久，字幕越晚才
# 出現），改回 10 秒——多數句子本來就會在幾秒內遇到真正的停頓，很少真的
# 撐到這個上限，這個平衡點對一般對話的字幕節奏比較剛好。
MAX_UTTERANCE_SEC = 10

# 畫面+聲音會延遲這麼多秒才播放給你看/聽，翻譯字幕也會延遲相同秒數才顯示，
# 兩者對在一起，達到「精準對上這個人開口瞬間」的效果。啟動時可以覆蓋這個值。
DISPLAY_DELAY_SEC = 6.0

# 結束播放後，整理逐字稿時用：停頓超過這麼久，就在整理稿裡另起一個段落標題
PARAGRAPH_GAP_SEC = 8

# 畫面擷取：你有 64GB 記憶體，儘量拉高解析度沒問題（緩衝區全部存在 RAM 裡，
# 不會碰硬碟）。fps 沒有跟著拉到螢幕更新率(165)，是因為網頁影片內容本身通常
# 就只有 30~60fps，擷取超過內容本身的幀率只是重複抓同一張畫面，純粹浪費記憶體，
# 肉眼也看不出差異，所以 fps 維持 60（已經覆蓋掉絕大部分影片/直播平台的實際幀率）。
CAPTURE_MAX_WIDTH = 1920
CAPTURE_TARGET_FPS = 60

# 自動把來源視窗的音量調到這個比例（不是靜音，原因見檔案開頭說明）。
# 這是數位訊號直接縮放，不是類比訊號，調再低也不會有雜訊，可以放心壓低。
VOLUME_DUCK_LEVEL = 0.05

# 畫面/聲音緩衝區最多留幾秒（要比 DISPLAY_DELAY_SEC 大，多留一點緩衝空間）
BUFFER_MARGIN_SEC = 8.0

TRANSCRIPT_DIR = Path(__file__).parent / "transcripts"
TRANSCRIPT_DIR.mkdir(exist_ok=True)

GLOSSARY_PATH = Path(__file__).parent / "glossary.json"

# 用來自動把瀏覽器輸出裝置切到虛擬音訊線、結束時再切回去的小工具（NirSoft SoundVolumeView）
SOUND_VOLUME_VIEW_PATH = Path(__file__).parent / "tools" / "SoundVolumeView.exe"

# 第一次執行會照這份預設值建立 glossary.json，之後你可以自己編輯增減
# （常客來賓名字、節目裡的哏、常出現的專有名詞都可以加進去）
_DEFAULT_GLOSSARY = {
    "羊宮妃那": "羊宮妃那",
    "羊宮妃那のHOOOOPE!": "羊宮妃那のHOOOOPE!",
    "羊宮妃那のこもれびじかん": "羊宮妃那のこもれびじかん",
    "HOOOOPE": "HOOOOPE",
    "こもれびじかん": "こもれびじかん",
}


def load_glossary() -> dict:
    """讀取專有名詞對照表（glossary.json）。

    格式：{"日文詞": "希望顯示的中文"}
    左邊填容易被聽錯/翻錯的專有名詞（人名、節目名、常客來賓名等），
    右邊填你想要畫面上顯示的樣子；如果不想被翻譯、想直接保留原文，
    右邊填跟左邊一樣的文字就好（範例都是這樣設定的）。

    這份清單會做兩件事：
      1. 當提示詞餵給 Whisper，讓語音辨識比較容易正確拼出這些詞
      2. 翻譯前先保護起來，避免被翻譯模型誤譯成不相干的字
    """
    if not GLOSSARY_PATH.exists():
        with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
            json.dump(_DEFAULT_GLOSSARY, f, ensure_ascii=False, indent=2)
        print(f"已建立專有名詞對照表：{GLOSSARY_PATH}（可以自己編輯增減）")
        return dict(_DEFAULT_GLOSSARY)

    try:
        with open(GLOSSARY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("glossary.json 格式應該是一個 {\"日文\": \"中文\"} 物件")
        return data
    except (json.JSONDecodeError, OSError, ValueError) as e:
        print(f"讀取 glossary.json 失敗，改用預設內容：{e}")
        return dict(_DEFAULT_GLOSSARY)


def glossary_hint(glossary: dict) -> str:
    """把專有名詞對照表變成給 LLM 看的提示詞。

    用條列式格式（一行一個詞），比全部擠成一句用頓號隔開更容易讓模型每個都
    注意到，詞多的時候（例如 20 幾個）尤其有差；LLM 不像傳統翻譯模型那樣需要
    用佔位符「保護」專有名詞，但也不是 100% 保證會遵守，詞單一多還是可能漏。
    """
    if not glossary:
        return ""
    terms = sorted({t for t in glossary if t})
    bullet_list = "\n".join(f"- {t}" for t in terms)
    return (
        "以下是專有名詞對照表（人名、節目名、團名等），這些詞不管出現在句子的哪個"
        "位置，一律「完全保留原文」，絕對不要翻譯、意譯、或拆解成別的字：\n"
        f"{bullet_list}"
    )


_REPEAT_RUN_RE = re.compile(r"(.{1,4}?)\1{4,}")


def _collapse_repetition(text: str) -> str:
    """保險絲：翻譯模型偶爾會卡進重複同一小段文字的退化迴圈，
    這裡把「連續重複 5 次以上」的片段砍成只留 2 次，避免洗版整個字幕視窗。
    正常語句裡的疊字（哈哈哈、等等等）通常不會連續重複到 5 次以上，不受影響。
    """
    return _REPEAT_RUN_RE.sub(lambda m: m.group(1) * 2, text)


def resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """簡單線性內插重取樣：夠快、辨識用途夠準，但拿來播放給人聽會有點粗糙/失真，
    語音辨識/VAD 這種不需要人耳聽的地方用這個就好。"""
    if orig_sr == target_sr:
        return x.astype(np.float32)
    duration = len(x) / orig_sr
    target_len = max(1, int(duration * target_sr))
    orig_idx = np.linspace(0, len(x) - 1, num=len(x))
    target_idx = np.linspace(0, len(x) - 1, num=target_len)
    return np.interp(target_idx, orig_idx, x).astype(np.float32)


# Whisper 吃到雜音/靜音時常見的「腦補」制式片語，遇到整句剛好等於這些就直接丟掉。
# 這些都是訓練資料裡到處都有的通用套話，不是特定內容
_HALLUCINATION_DENYLIST = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "最後までご視聴いただきありがとうございました",
    "チャンネル登録よろしくお願いします",
    "字幕視聴者向け",
    "ご清聴ありがとうございました",
}


def resample_high_quality(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """給人耳朵聽的音訊重取樣用這個：多相濾波（polyphase filtering），
    比單純線性內插乾淨很多，延遲播放器輸出聲音就是靠這個轉成裝置需要的取樣率。"""
    if orig_sr == target_sr or len(x) == 0:
        return x.astype(np.float32)
    g = gcd(orig_sr, target_sr)
    up, down = target_sr // g, orig_sr // g
    return resample_poly(x, up, down).astype(np.float32)


_LOG_LINE_RE = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\] (JP|ZH): (.*)$")
_TITLE_LINE_RE = re.compile(r"^#TITLE:\s*(.*)$")


def _ensure_terminal_punct(text: str) -> str:
    """如果句子結尾沒有標點，補一個句號，純排版用，不改變原本文字內容"""
    text = text.strip()
    if text and text[-1] not in "。！？!?…":
        text += "。"
    return text


def polish_transcript(raw_log_path: Path, paragraph_gap_sec: float = PARAGRAPH_GAP_SEC):
    """把即時存檔、每句都帶時間戳的逐字稿，整理成方便閱讀的段落格式。

    只做排版、合併段落、補標點這類不更動原意的整理；辨識不清楚或翻得
    奇怪的地方一律照實保留原文，不會自己腦補或「順」成聽起來更通順的內容。
    回傳整理後的檔案路徑；如果沒有任何內容則回傳 None（原始逐字稿不受影響）。
    """
    if not raw_log_path.exists():
        return None

    entries = []  # (當天秒數, 時間字串, 日文, 中文)
    episode_title = None
    pending_ja = None
    pending_ts = None

    with open(raw_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            title_m = _TITLE_LINE_RE.match(line)
            if title_m:
                episode_title = title_m.group(1).strip() or None
                continue

            m = _LOG_LINE_RE.match(line)
            if not m:
                continue
            h, mi, s, kind, text = m.groups()
            if kind == "JP":
                pending_ja = text
                pending_ts = (int(h) * 3600 + int(mi) * 60 + int(s), f"{h}:{mi}:{s}")
            elif kind == "ZH" and pending_ja is not None:
                sec_of_day, ts_str = pending_ts
                entries.append((sec_of_day, ts_str, pending_ja, text))
                pending_ja = None
                pending_ts = None

    if not entries:
        return None

    out_path = raw_log_path.with_name(raw_log_path.stem + "_polished.md")

    lines = [
        f"# {episode_title}" if episode_title else "# 中日對照逐字稿",
        "",
        f"原始逐字稿：{raw_log_path.name}",
        f"整理時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"共 {len(entries)} 句",
        "",
        "---",
    ]

    prev_sec = None
    for sec, ts_str, ja, zh in entries:
        if prev_sec is None or (sec - prev_sec) >= paragraph_gap_sec:
            lines.append(f"\n## {ts_str}\n")
        prev_sec = sec

        lines.append(_ensure_terminal_punct(ja))
        lines.append(_ensure_terminal_punct(zh))
        lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return out_path


_POLISH_CHUNK_LINES = 60  # 事後整理時，每批送幾句給翻譯模型，避免單次回應太長

_REBUILD_SYSTEM_PROMPT = (
    "你是專業的日文-繁體中文口譯兼編輯。使用者會分批貼上同一集日文廣播/網路節目的"
    "逐句語音辨識稿（可能有少數同音錯字，請憑上下文合理修正，不要另外編造內容）。"
    "請針對這一批內容，把每一句翻成自然流暢、口語化的繁體中文，不要逐字直譯。\n"
    "請忠實呈現內容，不要自己刪減或改寫語氣詞、口頭禪這類東西——錄音裡實際講了"
    "什麼，就照樣翻出來，不要自己判斷哪些是「贅字」就省略掉。\n"
    "輸出格式：每句先輸出修正過的日文原句，換行輸出對應的繁體中文翻譯，"
    "句子之間空一行；只要輸出這些內容，不要加任何額外的說明、前言或標題。"
)


def _rebuild_user_prompt(batch: list[str], context_tail: str) -> str:
	"""Build the reconstruction batch prompt shared by the Gemini and local backends."""
	return (
		(f"（前面幾句當上下文參考，不用重複翻譯：\n{context_tail}\n\n") if context_tail else ""
	) + "請處理這一批句子：\n" + "\n".join(batch)


def _polish_with_gemini(translator: "Translator", ja_lines: list[str], system_prompt: str) -> list[str]:
    polished_parts = []
    context_tail = ""
    total_batches = (len(ja_lines) + _POLISH_CHUNK_LINES - 1) // _POLISH_CHUNK_LINES
    for batch_no, i in enumerate(range(0, len(ja_lines), _POLISH_CHUNK_LINES), start=1):
        batch = ja_lines[i:i + _POLISH_CHUNK_LINES]
        user_prompt = _rebuild_user_prompt(batch, context_tail)

        # 這裡跟即時字幕不一樣：這是背景整理工作，不趕時間，撞到免費額度限制
        # 的話等久一點重試，把它做完比做快更重要——所以不用 translator._chat
        # 那個「立刻放棄」的快速失敗邏輯，這裡自己重試
        result = ""
        for attempt in range(3):
            result = translator._chat(system_prompt, user_prompt)
            if result:
                break
            print(f"  第 {batch_no} 批翻譯失敗（可能撞到額度限制），20 秒後重試（第 {attempt + 1} 次）...")
            time.sleep(20)

        if result:
            polished_parts.append(result)
        context_tail = "\n".join(batch[-2:])
        print(f"  已處理第 {batch_no}/{total_batches} 批")

        if batch_no < total_batches:
            # 主動放慢節奏，盡量不要撞到免費額度的「每分鐘請求數」上限
            time.sleep(4.5)
    return polished_parts


def _polish_with_local_model(
	translator: "Translator",
	ja_lines: list[str],
	glossary: dict | None,
	system_prompt: str,
) -> list[str] | None:
	"""Same prompts and output layout as Gemini; batches split and retry on bad replies."""
	engine = translator.local
	return engine.polish_bilingual(
		ja_lines,
		glossary=glossary,
		system_prompt=system_prompt,
		build_user_prompt=_rebuild_user_prompt,
		context_prompts=lambda prev_ja, current_ja: _context_prompts(prev_ja, current_ja, glossary),
		progress=lambda batch_no, total: print(f"  已處理第 {batch_no}/{total} 批"),
	)


def rebuild_transcript_from_full_audio(
    wav_path: "Path",
    asr_model: "WhisperModel",
    translator: "Translator",
    glossary: dict | None,
    episode_title: str = "",
) -> "Path | None":
    """事後（沒有即時限制）用完整錄音重新整理一份逐字稿。

    即時觀看時，Whisper 一次只看幾秒鐘的音訊、翻譯只看前一句當上下文；這裡因為
    已經結束播放、不用趕時間，可以讓 Whisper 用內建 VAD 重新辨識整段錄音（前後
    文連貫，準確度比逐句處理時更好），再把整批辨識結果交給翻譯模型潤過、統一
    用詞——概念上就是手動把錄音丟 NotebookLM 轉錄、再丟 Gemini 潤稿的那個流程，
    差別是全部自動跑完，不用你自己動手。
    """
    if not wav_path.exists():
        return None

    print("正在用完整錄音重新辨識（沒有即時限制，準確度會比即時逐句辨識時更好，可能要等幾分鐘）...")
    segments, _ = asr_model.transcribe(
        str(wav_path),
        language="ja",
        vad_filter=True,   # 交給 Whisper 內建 VAD 處理整段音訊的斷句，不用自己切
        beam_size=WHISPER_BEAM_SIZE,
        initial_prompt="、".join(glossary.keys()) if glossary else None,
        no_speech_threshold=0.6,
        log_prob_threshold=-1.0,
    )

    ja_lines = []
    for seg in segments:
        text = seg.text.strip()
        if not text or text in _HALLUCINATION_DENYLIST:
            continue
        ja_lines.append(text)

    if not ja_lines:
        print("完整錄音重新辨識沒有偵測到內容，略過這份整理稿。")
        return None

    print(f"重新辨識完成，共 {len(ja_lines)} 句，正在請翻譯模型分批整理成完整逐字稿...")

    system_prompt = _REBUILD_SYSTEM_PROMPT
    hint = glossary_hint(glossary) if glossary else ""
    if hint:
        system_prompt += "\n" + hint

    translation_only = False
    if getattr(translator, "backend", TRANSLATION_BACKEND_GEMINI) == TRANSLATION_BACKEND_LOCAL:
        translation_only = translator.local.is_translation_only
        polished_parts = _polish_with_local_model(translator, ja_lines, glossary, system_prompt)
        if polished_parts is None:
            # 服務中途停掉：保留完整錄音，之後可以用 transcribe_audio_file.py 重新處理
            print(f"本機模型服務無法使用，已保留完整錄音：{wav_path}")
            return None
    else:
        polished_parts = _polish_with_gemini(translator, ja_lines, system_prompt)

    if not polished_parts:
        print("翻譯模型沒有回傳任何內容，略過這份整理稿。")
        return None

    if translation_only:
        summary = (
            "這份逐字稿是結束播放後，用完整錄音重新辨識，再由本機翻譯專用模型逐句翻譯的版本"
            "（翻譯專用模型不會校正辨識錯字、也不會參考上下文潤稿），內容照實呈現、沒有自己刪減。"
        )
    else:
        summary = (
            "這份逐字稿是結束播放後，用完整錄音重新辨識、翻譯模型看過完整上下文潤過的版本，"
            "準確度會比即時觀看時看到的字幕更好，內容照實呈現、沒有自己刪減。"
        )
    out_path = wav_path.with_name(wav_path.stem.replace("_audio", "") + "_notebooklm_style.md")
    header = [
        f"# {episode_title}" if episode_title else "# 中日對照逐字稿（事後重新辨識版）",
        "",
        f"整理時間：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        summary,
        "",
        "---",
        "",
    ]
    out_path.write_text("\n".join(header) + "\n\n".join(polished_parts) + "\n", encoding="utf-8")

    # 逐字稿已經整理好了，完整錄音本來就只是拿來重新辨識用、不是給人聽的，
    # 用完就刪掉，不用留著佔硬碟空間
    try:
        wav_path.unlink()
        print(f"逐字稿已產生，完整錄音（{wav_path.name}）已刪除。")
    except OSError as e:
        print(f"整理完成，但刪除完整錄音時發生錯誤（不影響逐字稿內容）：{e}")

    return out_path


def load_cue_file(cue_path: Path) -> dict | None:
    """讀取 transcribe_audio_file.py 產生的時間軸字幕檔。失敗回傳 None。"""
    try:
        data = json.loads(cue_path.read_text(encoding="utf-8"))
        cues = data.get("cues", [])
        if not cues:
            print(f"{cue_path} 裡沒有任何字幕內容。")
            return None
        return data
    except Exception as e:
        print(f"讀取字幕檔失敗：{e}")
        return None


def push_cues_to_queues(
    cues: list[dict],
    caption_queue: "queue.Queue",
    transcript_queue: "queue.Queue",
    playback_start_time: float,
):
    """把整份預先轉錄好的字幕，依照「播放開始的那一刻」換算成實際該顯示的
    時間點，一次全部塞進兩個 queue——不用另外寫一個即時處理迴圈，畫面/
    逐字稿視窗本來就有的 FIFO 顯示邏輯會自己照 release_at 排隊顯示。
    """
    for c in cues:
        release_at = playback_start_time + c["start"] + DISPLAY_DELAY_SEC
        ja, zh = c["ja"], c.get("zh", "")
        caption_queue.put((release_at, ja, zh))
        transcript_queue.put((release_at, ja, zh))


# ============================================================
# 視窗選擇 / 畫面擷取 / 延遲播放緩衝區
# ============================================================


def duck_process_volume(pid: int, level: float = VOLUME_DUCK_LEVEL):
    """把來源視窗的系統音量調低（不是靜音），避免跟延遲播放的聲音互相干擾。

    之所以不能直接靜音：音量在混音前被歸零的話，我們用來擷取聲音的
    WASAPI loopback 也會跟著聽到靜音，字幕跟延遲播放都會沒有聲音可用。
    """
    try:
        from pycaw.pycaw import AudioUtilities
        import psutil

        target_name = None
        try:
            target_name = psutil.Process(pid).name()
        except Exception:
            pass

        matched = 0
        for session in AudioUtilities.GetAllSessions():
            proc = session.Process
            if proc is None:
                continue
            if proc.pid == pid or (target_name and proc.name() == target_name):
                session.SimpleAudioVolume.SetMasterVolume(level, None)
                matched += 1

        if matched:
            print(f"已自動把來源視窗音量調低到 {int(level * 100)}%，避免跟延遲播放的聲音重疊。")
        else:
            print("找不到來源視窗對應的音訊工作階段，可能需要自己在音量混音器裡手動調低。")
    except Exception as e:
        print(f"自動調整音量失敗（不影響其他功能，可以自己手動調整）：{e}")


def get_app_output_device(process_name: str) -> str | None:
    """查詢某個程式目前實際輸出到哪個裝置（用 SoundVolumeView 匯出目前的音訊工作階段）。

    回傳裝置名稱；查不到、或 SoundVolumeView 不存在的話回傳 None。
    """
    if not SOUND_VOLUME_VIEW_PATH.exists():
        return None
    try:
        import subprocess
        import csv
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "sessions.csv"
            subprocess.run(
                [str(SOUND_VOLUME_VIEW_PATH), "/scomma", str(csv_path)],
                timeout=10, capture_output=True,
            )
            if not csv_path.exists():
                return None
            with open(csv_path, "r", encoding="utf-8-sig", errors="ignore") as f:
                for row in csv.reader(f):
                    # 欄位順序：Name, Type, Direction, Device Name, ...
                    if len(row) > 3 and process_name.lower() in row[0].lower() and row[1] == "Application":
                        return row[3] or None
    except Exception:
        pass
    return None


def set_app_output_device(process_name: str, device_name: str) -> bool:
    """用 SoundVolumeView 把某個程式的輸出裝置切換成指定裝置，成功回傳 True"""
    if not SOUND_VOLUME_VIEW_PATH.exists():
        return False
    try:
        import subprocess

        for role in ("console", "multimedia"):
            subprocess.run(
                [str(SOUND_VOLUME_VIEW_PATH), "/SetAppDefault", device_name, role, process_name],
                timeout=10, capture_output=True,
            )
        return True
    except Exception as e:
        print(f"切換 {process_name} 輸出裝置失敗：{e}")
        return False


class PauseState:
    """跨執行緒共用的暫停旗標：畫面擷取、音訊擷取、辨識、翻譯、延遲播放全部會
    檢查這個旗標——按下暫停鍵時全部一起停住，再按一次繼續，從「現在」重新
    接續，不會把暫停期間硬接成好像真的有發生過的畫面/聲音/字幕。
    """

    def __init__(self):
        self.event = threading.Event()  # set = 目前是暫停狀態

    def is_paused(self) -> bool:
        return self.event.is_set()

    def toggle(self) -> bool:
        """切換暫停/繼續，回傳切換後的狀態（True 代表現在是暫停中）"""
        if self.event.is_set():
            self.event.clear()
            return False
        self.event.set()
        return True


class FrameBuffer:
    """執行緒安全的畫面緩衝區：擷取執行緒寫入畫面，播放視窗讀取「N 秒前」的那一張"""

    def __init__(self, max_seconds: float):
        self.max_seconds = max_seconds
        self.lock = threading.Lock()
        self.frames = deque()  # [(timestamp, PIL.Image)]，依時間先後排列

    def push(self, ts: float, image):
        with self.lock:
            self.frames.append((ts, image))
            cutoff = ts - self.max_seconds
            while self.frames and self.frames[0][0] < cutoff:
                self.frames.popleft()

    def get_at(self, target_ts: float):
        with self.lock:
            if not self.frames:
                return None
            best = self.frames[0][1]
            for ts, image in self.frames:
                if ts <= target_ts:
                    best = image
                else:
                    break
            return best


class AudioRingBuffer:
    """執行緒安全的音訊環形緩衝區：即時寫入音訊，播放時讀取「N 秒前」的位置"""

    def __init__(self, sample_rate: int, max_seconds: float):
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(sample_rate * max_seconds))
        self.lock = threading.Lock()
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_written = 0
        self.start_time = time.time()

    def write(self, samples: np.ndarray):
        with self.lock:
            n = len(samples)
            if n == 0:
                return

            # 寫入位置永遠用「絕對樣本編號 % 緩衝區大小」計算，不能自己單獨累加 write_pos，
            # 不然單次寫入量超過緩衝區容量時（理論上不會發生，但求穩固）位置會對不齊。
            store = samples if n <= self.max_samples else samples[-self.max_samples:]
            store_n = len(store)

            self.total_written += n
            store_start_abs = self.total_written - store_n
            start_pos = store_start_abs % self.max_samples

            end = start_pos + store_n
            if end <= self.max_samples:
                self.buffer[start_pos:end] = store
            else:
                first = self.max_samples - start_pos
                self.buffer[start_pos:] = store[:first]
                self.buffer[:end - self.max_samples] = store[first:]
            self.write_pos = end % self.max_samples

    def read_at(self, target_time: float, n: int) -> np.ndarray:
        with self.lock:
            elapsed = target_time - self.start_time
            target_sample = int(elapsed * self.sample_rate)
            available_start = max(0, self.total_written - self.max_samples)

            if target_sample < available_start:
                target_sample = available_start
            if target_sample >= self.total_written:
                return np.zeros(n, dtype=np.float32)

            end_sample = min(target_sample + n, self.total_written)
            count = end_sample - target_sample
            start_pos = target_sample % self.max_samples

            out = np.zeros(n, dtype=np.float32)
            if start_pos + count <= self.max_samples:
                out[:count] = self.buffer[start_pos:start_pos + count]
            else:
                first = self.max_samples - start_pos
                out[:first] = self.buffer[start_pos:]
                out[first:count] = self.buffer[:count - first]
            return out

    def latest_time(self) -> float:
        """回傳「目前實際已經寫到哪個時間點」，給延遲播放器起算用。

        不能直接用 time.time() 去反推，因為 AudioRingBuffer 建立的當下（在載入
        Whisper/選裝置這些步驟之前）到真正開始寫入音訊之間，會有一段等待時間；
        如果起算基準沒扣掉這段等待，播放器會永遠在跟一個不存在的未來時間點要
        資料，導致緩衝區裡「有資料」但內容永遠是靜音，卻不會顯示任何錯誤。
        """
        with self.lock:
            return self.start_time + self.total_written / self.sample_rate


class ScreenCapture:
    """擷取指定視窗的畫面，存進 FrameBuffer 供延遲播放。

    實際擷取邏輯（PrintWindow + PW_RENDERFULLCONTENT）跑在一個獨立的行程
    （process）裡，定義在 screen_capture_worker.py——不是像之前那樣用同一個
    行程裡的執行緒。原因：Python 同一行程內不管開幾條執行緒，同一時間真正在
    跑的 Python 程式碼還是只有一個（GIL 限制），擷取畫面這種吃 CPU 的工作
    跟語音辨識/翻譯/UI 擠在一起搶執行權，就是隨機雜音/卡頓的來源。獨立行程
    有自己的直譯器和 GIL，兩邊真正並行、不會互搶，才是治本的做法。

    畫面資料改用共享記憶體傳回主行程，不是 multiprocessing.Queue：Queue 要
    pickle 序列化、複製進系統管道，1920 寬、60fps 的資料量（一秒幾百 MB）
    撐不住，會掉幀、卡頓；共享記憶體是直接讀寫同一塊記憶體，沒有序列化/複製
    的成本。用 3 個輪替緩衝格（見 screen_capture_worker.NUM_SLOTS），寫入端
    寫完整格才更新「最新是哪一格」的索引，讀取端只讀完整寫完的那一格，繞一圈
    回來蓋掉同一格之前，讀取端早就複製走了，不用額外加鎖。

    對外行為跟原本的執行緒版本完全一樣：start()/stop()/switch_window()，
    呼叫端不用改。
    """

    NUM_SLOTS = 3

    def __init__(
        self,
        hwnd: int,
        frame_buffer: "FrameBuffer",
        max_width: int = CAPTURE_MAX_WIDTH,
        target_fps: int = CAPTURE_TARGET_FPS,
        pause_state: "PauseState | None" = None,
    ):
        self.frame_buffer = frame_buffer
        self.max_width = max_width
        self.target_fps = target_fps
        self.pause_state = pause_state

        # HWND 用 signed 64-bit 存，Windows 上視窗代碼本質是指標大小的值，
        # 用 32-bit 存在少數情況下可能裝不下
        self._hwnd_shared = multiprocessing.Value(ctypes.c_longlong, hwnd)

        # 每一格的容量保守抓「寬 x 寬」的正方形上限（一般影片視窗高度不會超過
        # 寬度），RGB 3 bytes/像素；真的遇到更高的畫面，擷取行程那邊會跳過
        # 那一張並印出警告，不會寫爆記憶體
        self._slot_nbytes = max_width * max_width * 3
        self._shm = shared_memory.SharedMemory(create=True, size=self._slot_nbytes * self.NUM_SLOTS)
        self._shm_view = np.ndarray(
            (self.NUM_SLOTS, self._slot_nbytes), dtype=np.uint8, buffer=self._shm.buf
        )
        # 每格存 (height, width, timestamp) 三個值
        self._meta = multiprocessing.Array(ctypes.c_double, self.NUM_SLOTS * 3)
        self._latest_slot = multiprocessing.Value(ctypes.c_int, -1)  # -1 = 還沒有任何畫面

        self._stop_event = multiprocessing.Event()
        self._pause_event = multiprocessing.Event()
        self._process: "multiprocessing.Process | None" = None
        self._receiver_thread: "threading.Thread | None" = None
        self._receiver_stop = threading.Event()

    def start(self):
        from screen_capture_worker import run_capture_process

        self._process = multiprocessing.Process(
            target=run_capture_process,
            args=(
                self._hwnd_shared, self.max_width, self.target_fps,
                self._shm.name, self._slot_nbytes, self._meta,
                self._latest_slot, self._stop_event, self._pause_event,
            ),
            daemon=True,
        )
        self._process.start()

        # 擷取行程只負責產生畫面，真正寫進 FrameBuffer 這件事在主行程這邊一條
        # 很輕量的執行緒做（只是複製現成資料塞進去，幾乎不耗 CPU），這樣
        # frame_buffer 這個物件完全不用改，其他地方也不用知道畫面是從哪裡來的
        self._receiver_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receiver_thread.start()

    def _receive_loop(self):
        last_seen = -1
        while not self._receiver_stop.is_set():
            # 把主行程這邊的暫停旗標同步給擷取行程，按暫停時畫面擷取才會一起停
            if self.pause_state is not None:
                if self.pause_state.is_paused():
                    self._pause_event.set()
                else:
                    self._pause_event.clear()

            slot = self._latest_slot.value
            if slot == -1 or slot == last_seen:
                time.sleep(0.005)
                continue
            last_seen = slot

            idx = slot * 3
            h = int(self._meta[idx])
            w = int(self._meta[idx + 1])
            ts = self._meta[idx + 2]
            nbytes = h * w * 3
            frame = self._shm_view[slot, :nbytes].reshape(h, w, 3).copy()
            self.frame_buffer.push(ts, frame)

    def switch_window(self, hwnd: int):
        """切換擷取目標到另一個視窗，執行中隨時可以呼叫，下一次擷取就會生效"""
        self._hwnd_shared.value = hwnd

    def stop(self):
        self._stop_event.set()
        self._receiver_stop.set()
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
        if self._receiver_thread is not None:
            self._receiver_thread.join(timeout=1.0)
        self._shm.close()
        self._shm.unlink()


class DelayedAudioPlayer:
    """把 AudioRingBuffer 裡「N 秒前」的聲音，即時播放到你選定的輸出裝置。

    刻意把「取樣率轉換」這種比較耗運算的工作，丟給獨立的背景執行緒提前算好、
    放進一個小緩衝區；真正即時播放用的 callback 只單純從緩衝區拿現成資料，
    不做任何運算。這是因為 callback 是音訊播放的關鍵路徑，只要偶爾被其他
    執行緒（例如畫面更新）卡住沒跑到，就會產生喀喀聲/雜訊；音訊播放不能有
    任何延誤，所以把耗運算的部分挪出這個路徑之外。
    """

    def __init__(
        self, audio_ring: "AudioRingBuffer", device: int | None = None,
        pause_state: "PauseState | None" = None,
    ):
        self.audio_ring = audio_ring
        self.device = device
        self.pause_state = pause_state
        self._stream = None
        self._feeder_stop = threading.Event()
        self._feeder_thread = None
        self._out_lock = threading.Lock()
        self._out_chunks = deque()  # 已經轉換好取樣率、隨時可以播放的資料
        self._out_available = 0     # _out_chunks 裡總共還有幾個樣本

    def start(self):
        # 很多輸出裝置不支援 16kHz 這種非標準取樣率，改用裝置自己的預設取樣率開串流
        device_index = self.device if self.device is not None else sd.default.device[1]
        out_sr = int(sd.query_devices(device_index)["default_samplerate"])
        if out_sr <= 0:
            print(f"警告：輸出裝置回報的取樣率異常（{out_sr}），改用 48000 Hz。")
            out_sr = 48000

        print(f"延遲播放輸出取樣率：{out_sr} Hz")

        self._feeder_thread = threading.Thread(
            target=self._feeder_loop, args=(out_sr,), daemon=True
        )
        self._feeder_thread.start()

        status_error_count = {"n": 0}
        underrun_count = {"n": 0}

        def callback(outdata, frames, time_info, status):
            if status:
                # PortAudio 回報底層有異常（例如裝置連結被切斷、underflow）時會設這個旗標，
                # 印出來才看得到「聲音其實已經連不到硬體了」這種靜默失敗的狀況
                status_error_count["n"] += 1
                if status_error_count["n"] == 1 or status_error_count["n"] % 50 == 0:
                    print(f"延遲播放輸出串流回報異常（已發生 {status_error_count['n']} 次）：{status}")
            with self._out_lock:
                out = np.zeros(frames, dtype=np.float32)
                filled = 0
                while filled < frames and self._out_chunks:
                    chunk = self._out_chunks[0]
                    take = min(len(chunk), frames - filled)
                    out[filled:filled + take] = chunk[:take]
                    filled += take
                    if take == len(chunk):
                        self._out_chunks.popleft()
                    else:
                        self._out_chunks[0] = chunk[take:]
                    self._out_available -= take
                outdata[:, 0] = out

            if filled < frames:
                # 緩衝區被掏空、補不出足夠的資料，只能吐部分靜音——這就是「有訊息
                # 說運作正常，但實際上聽起來沒聲音」最可能發生的地方，記錄下來方便確認
                underrun_count["n"] += 1
                if underrun_count["n"] == 1 or underrun_count["n"] % 100 == 0:
                    print(f"延遲播放緩衝區來不及補資料（已發生 {underrun_count['n']} 次），畫面可能因此聽起來沒聲音或斷斷續續。")

        self._stream = sd.OutputStream(
            samplerate=out_sr, channels=1, dtype="float32",
            blocksize=1024, latency="high", callback=callback, device=self.device,
        )
        self._stream.start()

    def _feeder_loop(self, out_sr: int):
        # audio_ring 現在存的是「擷取裝置原始取樣率」的音訊（不是給 Whisper 用的
        # 16kHz 版本），播放品質才不會被 Whisper 需要的低取樣率拖累——這裡要用
        # ring 自己記錄的取樣率，不能再假設是 TARGET_SR
        ring_sr = self.audio_ring.sample_rate

        # 每次重取樣的份量拉大一點：resample_poly 這類濾波器在每一小段的頭尾都會有
        # 一點邊界效應，段落切得越細、接縫越多，聽起來就會變成連續的「吱吱聲」；
        # 段落拉大到 0.5 秒，接縫次數降到 1/10，這個雜音應該會明顯改善
        chunk_samples_in = max(1, int(0.5 * ring_sr))
        # 拉大一點緩衝目標：真正執行時這條背景執行緒要跟畫面擷取/Whisper/翻譯
        # 一起搶執行資源，比單獨測試時容易被排擠、來不及補資料，緩衝大一點
        # 才有本錢撐過這種被搶資源的空檔，不然 callback 只能吐靜音。改用原始
        # 取樣率存音訊之後，每次重取樣要處理的資料量比之前的 16kHz 版本多了
        # 2~3 倍，這條執行緒更容易跟不上，緩衝空間也跟著拉大一點做為緩衝。
        target_buffer_sec = 4.0
        # 用 audio_ring 實際「寫到哪裡了」當基準，不能直接用 time.time()：
        # 建立緩衝區到真正開始寫音訊之間，中間隔了載入模型、選裝置這些步驟的
        # 等待時間，用 wall clock 反推會一直對到一個永遠不會有資料的未來時間點
        next_read_time = self.audio_ring.latest_time() - DISPLAY_DELAY_SEC

        error_count = 0
        first_chunk_logged = False
        was_paused = False

        while not self._feeder_stop.is_set():
            try:
                if self.pause_state is not None and self.pause_state.is_paused():
                    was_paused = True
                    time.sleep(0.05)
                    continue
                if was_paused:
                    # 剛從暫停恢復：讀取位置要重新對齊「現在」，不然會一直對到
                    # 暫停期間那段沒有任何音訊被寫進來的空白，永遠補不上進度
                    next_read_time = self.audio_ring.latest_time() - DISPLAY_DELAY_SEC
                    was_paused = False

                with self._out_lock:
                    buffered_sec = self._out_available / out_sr if out_sr else 0
                if buffered_sec >= target_buffer_sec:
                    time.sleep(0.01)
                    continue

                samples = self.audio_ring.read_at(next_read_time, chunk_samples_in)
                next_read_time += chunk_samples_in / ring_sr
                # 這段是真的要播給你聽的，用高品質重取樣，不要用 resample_linear
                # （那個給 Whisper 辨識用可以，人耳朵聽得出差別）
                resampled = resample_high_quality(samples, ring_sr, out_sr)

                with self._out_lock:
                    self._out_chunks.append(resampled)
                    self._out_available += len(resampled)

                if not first_chunk_logged:
                    first_chunk_logged = True
                    print("延遲播放音訊產生執行緒運作正常，開始供應音訊資料。")
                error_count = 0
            except Exception as e:
                # 這條背景執行緒如果沒有捕捉例外，出錯會直接靜默死掉，
                # 之後就再也沒有音訊資料可以播放，卻完全不會有任何錯誤訊息可以看——
                # 這裡印出來、稍微等一下再繼續，不要讓它整個掛掉
                error_count += 1
                if error_count == 1 or error_count % 50 == 0:
                    print(f"延遲播放音訊產生時發生錯誤（已重試 {error_count} 次）：{e}")
                time.sleep(0.05)

    def stop(self):
        self._feeder_stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass


class AudioCapture(threading.Thread):
    """從系統輸出裝置 (loopback) 擷取音訊：
    1. 用 Silero VAD（神經網路）切成一句一句丟進 utterance_queue，供辨識+翻譯用
    2. 同時把每一小段原始音訊寫進 AudioRingBuffer，供延遲播放用（不受 VAD 影響，
       靜音的部分也會寫進去，這樣延遲播放出來的聲音才會是連續、自然的）
    3. 如果手動按下「開始錄製」，同時把完整音訊寫成一個 WAV 檔存起來——結束播放
       後可以拿這份完整錄音重新整段辨識+翻譯，準確度比即時逐句處理時更好（概念
       上就是手動把錄音丟 NotebookLM 轉錄、再丟 Gemini 潤稿那個流程，差別是自動）
    """

    def __init__(
        self,
        utterance_queue: "queue.Queue",
        audio_ring: "AudioRingBuffer | None" = None,
        device: dict | None = None,
        record_path: "Path | None" = None,
        pause_state: "PauseState | None" = None,
        enable_vad: bool = True,
    ):
        super().__init__(daemon=True)
        self.utterance_queue = utterance_queue
        self.audio_ring = audio_ring
        self.pause_state = pause_state
        # 讀取預先轉錄好字幕檔的模式不需要即時辨識，關掉 VAD 省下語音偵測的
        # 運算，也不會再把任何東西塞進 utterance_queue（反正沒有 Worker 會去讀）
        self.enable_vad = enable_vad
        self.pa = pyaudio.PyAudio()
        # 記住指定的裝置名稱，串流意外斷線要重連時，優先找回同一個裝置
        # （不然重連可能會抓回系統預設輸出，跟原本指定的虛擬音訊線對不上）
        self._preferred_name = device["name"] if device else None
        self.device = device if device is not None else self._pick_loopback_device()
        self.stream_sr = int(self.device["defaultSampleRate"])
        self.channels = self.device["maxInputChannels"]

        # 錄製現在是手動控制（按鈕觸發），這裡先只記住檔名，不自動開始寫檔；
        # 完整錄音存 16kHz 單聲道就好（跟餵給 Whisper 的版本同一份，不用另外
        # 轉檔），事後重新辨識用途足夠，比存原始取樣率省很多硬碟空間
        self._record_path = record_path
        self._wav_writer = None
        self._wav_lock = threading.Lock()

        self.vad_model = None
        self.vad_iterator = None
        if self.enable_vad:
            print("載入語音活動偵測模型（Silero VAD，第一次執行會自動下載）...")
            self.vad_model = load_silero_vad()
            self.vad_iterator = VADIterator(
                self.vad_model,
                sampling_rate=TARGET_SR,
                threshold=VAD_THRESHOLD,
                min_silence_duration_ms=SILENCE_END_MS,
                speech_pad_ms=VAD_SPEECH_PAD_MS,
            )
        self._stop = threading.Event()

    def start_recording(self) -> bool:
        """手動開始錄製完整音訊；已經在錄、或沒有指定檔名的話回傳 False"""
        if self._record_path is None:
            return False
        with self._wav_lock:
            if self._wav_writer is not None:
                return False
            writer = wave.open(str(self._record_path), "wb")
            writer.setnchannels(1)
            writer.setsampwidth(2)  # int16
            writer.setframerate(TARGET_SR)
            self._wav_writer = writer
        return True

    def stop_recording(self) -> bool:
        """手動停止錄製；本來就沒在錄的話回傳 False"""
        with self._wav_lock:
            if self._wav_writer is None:
                return False
            self._wav_writer.close()
            self._wav_writer = None
        return True

    def is_recording(self) -> bool:
        with self._wav_lock:
            return self._wav_writer is not None

    def _pick_loopback_device(self):
        if self._preferred_name:
            try:
                for d in self.pa.get_loopback_device_info_generator():
                    if d["name"] == self._preferred_name:
                        return d
            except Exception:
                pass  # 找不到就繼續往下退回系統預設

        try:
            device = self.pa.get_default_wasapi_loopback()
        except OSError:
            print("找不到預設輸出裝置的 loopback，請確認 Windows 音效輸出裝置正常運作。")
            sys.exit(1)
        return device

    def _open_stream(self, block_frames: int):
        return self.pa.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.stream_sr,
            input=True,
            input_device_index=self.device["index"],
            frames_per_buffer=block_frames,
        )

    def stop(self):
        self._stop.set()

    def run(self):
        # 擷取用的區塊大小可以跟 VAD 窗格大小脫鉤，這裡用接近 32ms 的原生取樣數即可
        block_frames = max(1, int(self.stream_sr * VAD_CHUNK_SAMPLES / TARGET_SR))
        stream = self._open_stream(block_frames)

        pending = np.empty(0, dtype=np.float32)   # 還沒湊滿一個 VAD 窗格的殘餘音訊
        utterance_frames = []
        speech_active = False
        utterance_ms = 0
        utterance_start_time = None
        consecutive_errors = 0
        was_paused = False

        print(f"開始擷取系統音訊：{self.device['name']} ({self.stream_sr} Hz, {self.channels} ch)")
        print("請在瀏覽器播放影片... (Ctrl+C 或關閉視窗可結束)")

        while not self._stop.is_set():
            try:
                data = stream.read(block_frames, exception_on_overflow=False)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                # 有些虛擬音效裝置（例如音效增強軟體）在系統音訊拓樸改變時會把
                # 既有的 loopback 串流關掉，遇到這種狀況就嘗試重新開一個新的串流，
                # 而不是無限重試、瘋狂洗版一樣的錯誤訊息
                if consecutive_errors == 1:
                    print(f"讀取音訊發生錯誤，嘗試重新開啟音訊串流：{e}")
                elif consecutive_errors % 10 == 0:
                    print(f"音訊串流持續異常中（已重試 {consecutive_errors} 次）：{e}")

                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

                time.sleep(0.5)
                if self._stop.is_set():
                    break

                try:
                    self.device = self._pick_loopback_device()
                    self.stream_sr = int(self.device["defaultSampleRate"])
                    self.channels = self.device["maxInputChannels"]
                    block_frames = max(1, int(self.stream_sr * VAD_CHUNK_SAMPLES / TARGET_SR))
                    stream = self._open_stream(block_frames)
                    print(f"音訊串流已重新開啟：{self.device['name']} ({self.stream_sr} Hz, {self.channels} ch)")
                    consecutive_errors = 0
                except Exception:
                    pass  # 開不成就等下一輪迴圈再試一次
                continue

            if self.pause_state is not None and self.pause_state.is_paused():
                if not was_paused:
                    # 剛進入暫停：把還沒講完的那句話直接丟掉，避免暫停前後的音訊
                    # 被硬接成同一句話；VAD 內部狀態也重置，避免帶著暫停前的殘留
                    # 狀態去判斷暫停後全新的音訊
                    speech_active = False
                    utterance_frames = []
                    utterance_ms = 0
                    pending = np.empty(0, dtype=np.float32)
                    try:
                        self.vad_iterator.reset_states()
                    except Exception:
                        pass
                    was_paused = True
                continue  # 音訊還是要讀掉避免緩衝區溢位，但暫停中不寫進任何緩衝區
            was_paused = False

            audio = np.frombuffer(data, dtype=np.float32)
            if self.channels > 1:
                audio = audio.reshape(-1, self.channels).mean(axis=1)

            if self.audio_ring is not None:
                # 存原始擷取取樣率的音訊給延遲播放用，不要存降到 16kHz 之後的版本——
                # resample_linear 是單純線性內插、沒有做降頻前該有的抗鋸齒濾波，
                # 拿去餵 Whisper 沒差，但直接播給人聽會多一層可聽見的雜訊/悶悶的感覺，
                # 之後又要再重取樣回輸出裝置的取樣率，等於兩次有損轉換疊在一起。
                self.audio_ring.write(audio)

            audio_16k = resample_linear(audio, self.stream_sr, TARGET_SR)

            with self._wav_lock:
                if self._wav_writer is not None:
                    pcm16 = np.clip(audio_16k * 32767, -32768, 32767).astype(np.int16)
                    self._wav_writer.writeframes(pcm16.tobytes())

            if not self.enable_vad:
                # 讀取預先轉錄字幕檔的模式：只需要音訊進 audio_ring 給延遲播放，
                # 不用即時辨識，跳過整段 VAD／斷句處理
                continue

            pending = np.concatenate([pending, audio_16k])

            # 用固定窗格大小餵給 Silero VAD，累積出一句一句的語音
            while len(pending) >= VAD_CHUNK_SAMPLES:
                chunk = pending[:VAD_CHUNK_SAMPLES]
                pending = pending[VAD_CHUNK_SAMPLES:]

                chunk_ms = VAD_CHUNK_SAMPLES / TARGET_SR * 1000
                event = self.vad_iterator(torch.from_numpy(chunk), return_seconds=False)

                if event is not None and "start" in event:
                    speech_active = True
                    utterance_frames = []
                    utterance_ms = 0
                    utterance_start_time = time.time()

                if speech_active:
                    utterance_frames.append(chunk)
                    utterance_ms += chunk_ms

                force_flush = utterance_ms / 1000 >= MAX_UTTERANCE_SEC
                is_end = event is not None and "end" in event

                if speech_active and (is_end or force_flush):
                    utterance = np.concatenate(utterance_frames)
                    # 記下這句話「開口講的那一刻」，不是「講完的那一刻」——字幕要對上
                    # 延遲畫面裡這個人開口的瞬間，用開始時間才對；如果用講完的時間去加
                    # DISPLAY_DELAY_SEC，長句子（例如講滿 10 秒才被強制切斷）算出來的
                    # release_at 會剛好等於延遲畫面「整句話已經播完」的那一刻，變成要等
                    # 聽完一整句才看到字幕，完全喪失延遲緩衝原本要爭取的時間。
                    self.utterance_queue.put((utterance, utterance_start_time or time.time()))
                    utterance_frames = []
                    utterance_ms = 0
                    speech_active = not is_end  # 強制切斷後若還在講話，繼續累積下一段
                    if speech_active:
                        # 被迫切斷但這個人還在講、還沒停頓，剩下這段話就從「現在」這一刻算起
                        utterance_start_time = time.time()

        stream.stop_stream()
        stream.close()
        self.pa.terminate()
        with self._wav_lock:
            if self._wav_writer is not None:
                self._wav_writer.close()
                self._wav_writer = None


_SYSTEM_PROMPT_BASE = (
    "你是專業的日文-繁體中文口譯，正在幫忙即時翻譯一個日文廣播/網路節目的口語對話。"
    "內容常有語助詞、停頓、話講到一半重講、省略主詞這類真人講話的狀況，"
    "請翻成自然通順、口語化的繁體中文，像是真人在說話，不要翻得死板生硬、也不要逐字直譯。"
    "只要輸出翻譯結果本身，不要加任何說明、引號、備註、拼音或原文。"
)


def _context_prompts(prev_ja: str, current_ja: str, glossary: dict | None) -> tuple[str, str]:
	"""Build the live-caption prompts shared by the Gemini and local backends."""
	system_prompt = _SYSTEM_PROMPT_BASE
	hint = glossary_hint(glossary) if glossary else ""
	if hint:
		system_prompt += "\n" + hint
	if prev_ja:
		user_prompt = f"前一句話（僅供參考上下文，不用翻譯）：\n{prev_ja}\n\n請翻譯這一句：\n{current_ja}"
	else:
		user_prompt = f"請翻譯這一句：\n{current_ja}"
	return system_prompt, user_prompt


def resolve_translation_backend(value: str | None) -> str:
	"""Normalize a backend name; an empty value keeps the Gemini default."""
	name = str(value or "").strip().lower()
	if not name:
		return TRANSLATION_BACKEND_GEMINI
	if name not in TRANSLATION_BACKENDS:
		raise RuntimeError(f"翻譯引擎設定無效：{value}（請使用 gemini 或 local）")
	return name


class Translator:
    """翻譯引擎：預設呼叫 Gemini API；backend="local" 時改呼叫本機語言模型。

    Gemini：雲端大型模型的語言理解能力比塞得進消費級顯卡的本機模型好上一截，
    翻起來更自然、更少怪異的字句；代價是要申請 API 金鑰、有免費額度限制（撞到額度
    會自動重試幾次）、逐字稿內容會傳到 Google 的伺服器，這是雲端服務本來就
    會有的取捨。

    本機模型：送出跟 Gemini 完全相同的提示詞（同一組函式產生），另外加上本機小模型
    需要的保護——即時字幕逾時就快速放棄、回覆有問題時不帶上下文重翻一次、批次整理
    遇到截斷或格式跑掉會自動切小重送，服務停掉會快速失敗而不是卡住。
    """

    def __init__(self, backend: str | None = None):
        if backend is None:
            backend = os.environ.get("TRANSLATION_BACKEND")
        self.backend = resolve_translation_backend(backend)
        self.local = None
        if self.backend == TRANSLATION_BACKEND_LOCAL:
            self.client = None
            self.llama_server = None
            try:
                self.local = LocalTranslationEngine(postprocess=_collapse_repetition)
                # 只有在沒有任何服務回應、且設定了 model_path 時才會真的啟動子行程；
                # 已有服務在跑（Ontime/Ollama/自己手動啟動）一律原樣使用，不會被停掉。
                self.llama_server = ensure_llama_server(self.local.client)
                warmup_system, warmup_user = _context_prompts("", "こんにちは。", None)
                self.local.prepare(warmup_system, warmup_user, "こんにちは。")
            except LlamaServerError as e:
                # 這是自動啟動設定本身的問題（路徑找不到、網址不是本機等），訊息已經
                # 講清楚原因，不用再套一層「請先手動啟動」的樣板文字。
                raise RuntimeError(str(e)) from e
            except RuntimeError as e:
                raise RuntimeError(
                    f"無法使用本機語言模型：{e}\n"
                    "請先啟動 llama-server 並載入模型（見 README「本機語言模型」），確認 LOCAL_LLM_BASE_URL"
                    "（預設 http://127.0.0.1:8766）設定正確，或在啟動畫面改選 Gemini API。\n"
                    "提示：設定 LOCAL_LLM_MODEL_PATH（或 local_llm_server.json）可以讓本程式自動啟動"
                    "llama-server，不用自己先手動啟動（見 README）。"
                ) from e
            return

        _import_gemini_sdk()
        api_key = self._load_api_key()
        if not api_key:
            raise RuntimeError(
                "找不到 Gemini API 金鑰。可以用任一種方式提供：\n"
                "  1. 打開 gemini_api_key.txt，把你的金鑰貼進去存檔（雙擊 exe 這種\n"
                "     不會另外設環境變數的用法，就是靠這個檔案）\n"
                "  2. 或在終端機執行：$env:GEMINI_API_KEY = \"你的金鑰\"\n"
                "金鑰去 https://aistudio.google.com/apikey 免費申請。"
            )
        self.client = genai.Client(api_key=api_key)
        print(f"翻譯將呼叫 Gemini API：{GEMINI_MODEL}")

    @staticmethod
    def _load_api_key() -> str | None:
        # 先看環境變數（給想手動控制的人用），沒有的話改讀本機這個檔案——
        # 雙擊 exe 啟動沒有終端機可以先設環境變數，靠這個檔案才能一鍵直接用。
        # 這個檔案只會留在你自己的電腦，不會被讀到、上傳、或經過我們的對話紀錄，
        # 只是提醒你：這個檔案跟資料夾如果要分享/備份到別的地方，記得先拿掉金鑰。
        env_key = os.environ.get("GEMINI_API_KEY")
        if env_key:
            return env_key.strip()

        if GEMINI_API_KEY_FILE.exists():
            content = GEMINI_API_KEY_FILE.read_text(encoding="utf-8").strip()
            # 過濾掉檔案裡的說明文字（用 # 開頭的行），只留真正像金鑰的那行
            lines = [ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith("#")]
            if lines:
                return lines[0]
        return None

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.backend == TRANSLATION_BACKEND_LOCAL:
            return self.local.live_chat(system_prompt, user_prompt)
        # 這裡故意不做「撞到額度限制就睡幾秒重試」這件事——Worker 是單一執行緒
        # 依序處理每一句話，這裡卡多久，後面所有排隊中的字幕就跟著晚多久出現，
        # 睡眠等待幾秒鐘乘上好幾次重試，很容易就吃光延遲緩衝爭取來的時間，變成
        # 字幕塞車、過一陣子才一次噴出一大串（接 OpenRouter 時踩過這個坑）。
        # 遇到額度限制直接跳過這一句就好，下一句照常繼續嘗試，不會被卡住。
        try:
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.3,
                ),
            )
            result = (response.text or "").strip()
            return _collapse_repetition(result)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit:
                print("Gemini API 撞到免費額度限制，這句先跳過，下一句會繼續嘗試。")
                return ""

            print(f"呼叫 Gemini API 翻譯失敗：{e}")
            if "404" in str(e) or "NOT_FOUND" in str(e):
                # Google 常常改型號名稱/下架舊型號，型號名稱錯的話直接列出
                # 這把金鑰目前實際能用的型號，不用回來猜、也不用等我查
                self._print_available_models()
            return ""

    def _print_available_models(self):
        try:
            print("目前這把金鑰實際可用的型號（把 GEMINI_MODEL 換成其中一個試試看）：")
            for m in self.client.models.list():
                name = getattr(m, "name", str(m))
                if "flash" in name.lower() or "lite" in name.lower():
                    print(f"  {name}")
        except Exception as list_err:
            print(f"（列出可用型號也失敗了：{list_err}）")

    def translate(self, text: str, glossary: dict | None = None) -> str:
        """翻整段文字（螢幕字幕列用，可能是好幾句接在一起）"""
        if not text.strip():
            return ""
        system_prompt = _SYSTEM_PROMPT_BASE
        hint = glossary_hint(glossary) if glossary else ""
        if hint:
            system_prompt += "\n" + hint
        if self.backend == TRANSLATION_BACKEND_LOCAL:
            return self.local.translate_live(system_prompt, f"請翻譯：\n{text}", text, glossary)
        return self._chat(system_prompt, f"請翻譯：\n{text}")

    def translate_with_context(self, prev_ja: str, current_ja: str, glossary: dict | None = None) -> str:
        """翻「這一句」，但先讓模型看過前一句當作上下文（代名詞/省略主詞會準很多）。

        LLM 可以直接看懂「只翻最後一句、前面只是給你參考」這種指示，
        不用像傳統翻譯模型那樣還要用文字切割去猜哪一段是「這一句」的翻譯。
        """
        if not current_ja.strip():
            return ""

        system_prompt, user_prompt = _context_prompts(prev_ja, current_ja, glossary)
        if self.backend == TRANSLATION_BACKEND_LOCAL:
            # 同一組提示詞；回覆有問題（空白、照抄提示、沒翻成中文）時不帶前一句重翻一次
            return self.local.translate_live(
                system_prompt, user_prompt, current_ja, glossary, fallback_user_prompt=f"請翻譯：\n{current_ja}"
            )
        return self._chat(system_prompt, user_prompt)


class Worker(threading.Thread):
    """把語句音訊丟給 Whisper 辨識 + 翻譯，結果丟進兩個 queue，同時寫入逐字稿檔"""

    def __init__(
        self,
        utterance_queue: "queue.Queue",
        caption_queue: "queue.Queue",
        transcript_queue: "queue.Queue",
        episode_title: str = "",
        date_str: str | None = None,
        pause_state: "PauseState | None" = None,
        translator: "Translator | None" = None,
    ):
        super().__init__(daemon=True)
        self.utterance_queue = utterance_queue
        self.caption_queue = caption_queue
        self.transcript_queue = transcript_queue
        self.episode_title = episode_title
        self.pause_state = pause_state

        print("載入語音辨識模型（Whisper，第一次執行會自動下載）...")
        self.asr = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE
        )
        # main() 會先建好翻譯引擎（金鑰或本機服務有問題就在動到音訊設定前結束）
        self.translator = translator if translator is not None else Translator()
        self._stop = threading.Event()

        # 用來偵測 Whisper 卡進「幻覺迴圈」：音訊有雜音/glitch 時，有時候會連續
        # 辨識出同一句幾乎一模一樣的短句（常常剛好是提示詞裡的專有名詞），而且
        # 速度很快，會把逐字稿視窗瞬間灌爆、UI 整個卡死
        self._recent_ja_texts = deque(maxlen=4)
        self._hallucination_loop_warned = False

        self.prev_ja = ""  # 只看前一句當上下文，用來讓翻譯知道代名詞/省略主詞指的是誰

        self.glossary = load_glossary()
        # 把專有名詞清單當提示詞餵給 Whisper，幫助它正確拼出這些字，
        # 而不是純粹憑發音亂猜（「、」是日文的頓號，模型看得懂這種列舉格式）
        self.initial_prompt = "、".join(self.glossary.keys()) if self.glossary else None
        if self.glossary:
            print(f"已載入 {len(self.glossary)} 個專有名詞：{'、'.join(self.glossary.keys())}")

        date_str = date_str or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = TRANSCRIPT_DIR / f"transcript_{date_str}.txt"
        with open(self.log_path, "w", encoding="utf-8") as f:
            if self.episode_title:
                f.write(f"#TITLE: {self.episode_title}\n\n")
        print(f"逐字稿將儲存於：{self.log_path}")

    def stop(self):
        self._stop.set()

    def run(self):
        while not self._stop.is_set():
            if self.pause_state is not None and self.pause_state.is_paused():
                time.sleep(0.1)
                continue
            try:
                audio, capture_time = self.utterance_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            segments, _ = self.asr.transcribe(
                audio,
                language="ja",
                vad_filter=False,
                beam_size=WHISPER_BEAM_SIZE,
                condition_on_previous_text=False,  # 避免長時間播放時的重複幻覺
                initial_prompt=self.initial_prompt,  # 提示專有名詞正確拼法
                no_speech_threshold=0.6,   # 這個機率以上就當作沒有真人講話，過濾掉
                log_prob_threshold=-1.0,   # 平均信心太低的結果（很可能是腦補）也丟掉
            )
            segments = list(segments)
            # Whisper 遇到雜音/靜音這種沒有真人講話的片段，偶爾會「腦補」出訓練資料裡
            # 常見的制式片語（最典型的就是日文影片常見的結尾語「ご視聴ありがとう
            # ございました」），不是真的有人講這句話。no_speech_prob 太高的片段代表
            # 模型自己都覺得這裡可能沒人在講話，直接濾掉
            if any(getattr(seg, "no_speech_prob", 0.0) > 0.6 for seg in segments):
                continue
            ja_text = "".join(seg.text for seg in segments).strip()
            if not ja_text or ja_text in _HALLUCINATION_DENYLIST:
                continue

            # 除了固定的黑名單詞組，還要防「連續好幾句都一模一樣」這種一般性的
            # 幻覺迴圈——不管卡住的是哪句話，只要連續重複就先濾掉，不要真的塞進
            # queue 裡把畫面灌爆；等真的出現新內容，偵測會自動解除
            self._recent_ja_texts.append(ja_text)
            if (
                len(self._recent_ja_texts) == self._recent_ja_texts.maxlen
                and len(set(self._recent_ja_texts)) == 1
            ):
                if not self._hallucination_loop_warned:
                    self._hallucination_loop_warned = True
                    print(
                        f"偵測到 Whisper 疑似卡進幻覺迴圈（連續辨識出同一句「{ja_text}」），"
                        f"暫時濾掉不顯示；等真的出現新內容會自動恢復正常。"
                    )
                continue
            self._hallucination_loop_warned = False

            # 這句話該在幾點幾分顯示：開口的當下 + 延遲播放秒數，跟畫面/聲音用同一個延遲值，
            # 三者才會對在一起。如果翻譯花的時間比這個還久，release_at 會排到過去，
            # 視窗會直接顯示（不會反而更晚跳出）
            release_at = capture_time + DISPLAY_DELAY_SEC

            # 看過前一句上下文，準確度比單句翻譯好，而且是逐句對齊的乾淨結果；
            # 逐字稿面板、畫面下方字幕列都用同一份結果，不用多算一次
            # （字幕列以前會把最近幾句接起來整段重翻，除了拖慢速度，塞太多字
            # 也會把固定高度的字幕區撐爆，所以改成只顯示當下這一句就好）
            zh_final = self.translator.translate_with_context(self.prev_ja, ja_text, self.glossary)
            self.prev_ja = ja_text
            self.transcript_queue.put((release_at, ja_text, zh_final))
            self.caption_queue.put((release_at, ja_text, zh_final))

            with open(self.log_path, "a", encoding="utf-8") as f:
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                f.write(f"[{ts}] JP: {ja_text}\n")
                f.write(f"[{ts}] ZH: {zh_final}\n\n")


class PlayerWindow:
    """延遲播放畫面視窗，字幕（日文/中文）貼在畫面下方，像一般播放器內建字幕那樣"""

    def __init__(
        self,
        root: tk.Tk,
        container: tk.Frame,
        frame_buffer: "FrameBuffer",
        caption_queue: "queue.Queue",
        episode_title: str = "",
        pause_state: "PauseState | None" = None,
        screen_capture: "ScreenCapture | None" = None,
        audio_capture: "AudioCapture | None" = None,
        cue_playback: "dict | None" = None,
    ):
        self.root = root            # 整個視窗；控制按鈕列橫跨全寬，掛在這裡
        self.container = container  # PanedWindow 左邊那塊；畫面/字幕只蓋在這個範圍
        self.frame_buffer = frame_buffer
        self.caption_queue = caption_queue
        self.pause_state = pause_state
        self.screen_capture = screen_capture
        self.audio_capture = audio_capture
        # 讀取預先轉錄字幕檔模式才會有這個：{"cues": [...], "transcript_queue": ...}
        self.cue_playback = cue_playback
        self._cue_synced = False

        self._pending_caption = []  # [(release_at, ja_lines, zh)]
        self._next_caption_switch = 0.0  # 目前這句字幕最早可以幾點被換掉
        self._photo = None  # 保留參考，不然 Tkinter 的圖片會被垃圾回收

        # 操作按鈕列：暫停/繼續、切換擷取視窗、開始/停止錄製完整音訊——橫跨整個
        # 視窗頂端（不是只 pack 進 container，不然會只蓋住左半邊，右邊逐字稿看不到）
        control_frame = tk.Frame(self.root, bg="#1a1a1a", height=36)
        control_frame.pack(fill="x", side="top")
        control_frame.pack_propagate(False)

        if self.cue_playback is not None:
            # 讀取預先轉錄字幕檔模式：多一個「開始計時」鍵，按下的那一刻要跟
            # 瀏覽器裡按下播放的那一刻對齊，之後字幕才會準確跟著音訊進度跑
            self.sync_btn = tk.Button(
                control_frame, text="▶ 開始計時（跟播放對齊）", font=("Microsoft JhengHei", 10),
                command=self._on_start_sync, bg="#2e7d32", fg="white",
            )
            self.sync_btn.pack(side="left", padx=6, pady=4)

        self.pause_btn = tk.Button(
            control_frame, text="⏸ 暫停", font=("Microsoft JhengHei", 10),
            command=self._on_toggle_pause,
        )
        self.pause_btn.pack(side="left", padx=6, pady=4)

        if self.cue_playback is None:
            # 已經有預先轉錄好的完整逐字稿了，不需要再另外錄音重新辨識一次
            self.record_btn = tk.Button(
                control_frame, text="⏺ 開始錄製", font=("Microsoft JhengHei", 10),
                command=self._on_toggle_recording,
            )
            self.record_btn.pack(side="left", padx=6, pady=4)

        switch_btn = tk.Button(
            control_frame, text="🔄 切換視窗", font=("Microsoft JhengHei", 10),
            command=self._on_switch_window,
        )
        switch_btn.pack(side="left", padx=6, pady=4)

        self.container.configure(bg="black")
        self.canvas = tk.Canvas(self.container, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # 字幕區固定高度、關掉 pack_propagate：字幕內容一多一少，高度都不會變，
        # 不會因為文字長度不同就把上面的畫面區域推來推去。字太多就在固定範圍裡
        # 換行顯示，超出的部分會被裁掉，總比畫面一直被擠壓好。
        self.SUBTITLE_HEIGHT = 150
        subtitle_frame = tk.Frame(self.container, bg="black", height=self.SUBTITLE_HEIGHT)
        subtitle_frame.pack(fill="x", side="bottom")
        subtitle_frame.pack_propagate(False)

        self.ja_label = tk.Label(
            subtitle_frame, text="", fg="white", bg="black",
            font=("Meiryo", 15), wraplength=960, justify="center",
        )
        self.ja_label.pack(pady=(8, 0))
        self.zh_label = tk.Label(
            subtitle_frame, text="", fg="yellow", bg="black",
            font=("Microsoft JhengHei", 20, "bold"), wraplength=960, justify="center",
        )
        self.zh_label.pack(pady=(4, 10))

        # 綁在 container（左邊那塊）上，不是整個 root——現在兩邊比例可以拖曳
        # 分隔線調整，字幕換行寬度要跟著「左邊實際多寬」走，不是跟著整個視窗寬度
        self.container.bind("<Configure>", self._on_resize)

        self.root.after(1000 // CAPTURE_TARGET_FPS, self._poll_video)
        self.root.after(200, self._poll_captions)

    def _on_resize(self, event):
        if event.widget is self.container:
            wrap = max(200, event.width - 40)
            self.ja_label.config(wraplength=wrap)
            self.zh_label.config(wraplength=wrap)

    def _on_start_sync(self):
        if self.cue_playback is None or self._cue_synced:
            return
        # 按下的這一刻要跟瀏覽器裡按下播放的那一刻對齊，之後所有字幕的顯示
        # 時間點都是從這一刻換算出來的，按太早/太晚會讓字幕整批提早或延後
        playback_start_time = time.time()
        push_cues_to_queues(
            self.cue_playback["cues"],
            self.caption_queue,
            self.cue_playback["transcript_queue"],
            playback_start_time,
        )
        self._cue_synced = True
        self.sync_btn.config(text="✓ 已對齊", state="disabled")
        print(f"已對齊播放時間，共 {len(self.cue_playback['cues'])} 句字幕已排入顯示佇列。")

    def _on_toggle_pause(self):
        if self.pause_state is None:
            return
        now_paused = self.pause_state.toggle()
        self.pause_btn.config(text="▶ 繼續" if now_paused else "⏸ 暫停")
        print("已暫停擷取/辨識/翻譯/播放。" if now_paused else "已繼續。")

    def _on_toggle_recording(self):
        if self.audio_capture is None:
            return
        if self.audio_capture.is_recording():
            self.audio_capture.stop_recording()
            self.record_btn.config(text="⏺ 開始錄製")
            print("已停止錄製完整音訊。")
        else:
            if self.audio_capture.start_recording():
                self.record_btn.config(text="⏹ 停止錄製")
                print("已開始錄製完整音訊（結束播放後會拿去重新辨識、整理成更精準的逐字稿）。")

    def _on_switch_window(self):
        if self.screen_capture is None:
            return
        try:
            import pygetwindow as gw
            from startup_gui import pick_single_window
        except Exception as e:
            print(f"開啟切換視窗選單失敗：{e}")
            return

        candidates = [
            w for w in gw.getAllWindows()
            if w.title.strip() and w.visible and w.width > 200 and w.height > 150
        ]
        if not candidates:
            print("找不到可以切換的視窗。")
            return

        idx = pick_single_window([w.title for w in candidates], parent=self.root)
        if idx is None:
            return
        chosen = candidates[idx]
        self.screen_capture.switch_window(chosen._hWnd)
        print(f"已切換擷取視窗：{chosen.title}")

    def _poll_video(self):
        is_paused = self.pause_state is not None and self.pause_state.is_paused()
        if is_paused:
            # 暫停中：畫面就停在最後一格不動，只在上面疊一個「已暫停」的字，
            # 不要清掉畫布重畫，不然背景會變黑、看不到暫停前最後停在哪一格
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                self.canvas.delete("pause_overlay")
                self.canvas.create_text(
                    cw // 2, ch // 2, text="⏸ 已暫停", fill="yellow",
                    font=("Microsoft JhengHei", 28, "bold"), tags="pause_overlay",
                )
            self.root.after(1000 // CAPTURE_TARGET_FPS, self._poll_video)
            return

        target_time = time.time() - DISPLAY_DELAY_SEC
        frame = self.frame_buffer.get_at(target_time)
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()

        if frame is not None and cw > 10 and ch > 10:
            fh, fw = frame.shape[:2]  # frame 是 numpy 陣列 (H, W, 3)
            scale = min(cw / fw, ch / fh)
            new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
            # 用 cv2 縮放，比 PIL 的 resize 快很多，減少畫面更新這一步造成的掉幀
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            self._photo = ImageTk.PhotoImage(Image.fromarray(resized))
            self.canvas.delete("all")
            self.canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")
        elif cw > 10 and ch > 10:
            self.canvas.delete("all")
            self.canvas.create_text(
                cw // 2, ch // 2, text="緩衝中...", fill="white",
                font=("Microsoft JhengHei", 16),
            )

        self.root.after(1000 // CAPTURE_TARGET_FPS, self._poll_video)

    def _poll_captions(self):
        try:
            while True:
                self._pending_caption.append(self.caption_queue.get_nowait())
        except queue.Empty:
            pass

        now = time.time()
        # 依到達順序（FIFO）一句一句顯示，不能一次到期好幾句就只挑最新那句、
        # 把中間的句子直接丟掉——主持人講比較久、句子比較密的時候，翻譯很容易
        # 一次到期好幾句，之前的寫法會讓中間那些句子完全不會顯示、看起來像斷句。
        if self._pending_caption and self._pending_caption[0][0] <= now and now >= self._next_caption_switch:
            _, ja_text, zh = self._pending_caption.pop(0)
            self.ja_label.config(text=ja_text)
            self.zh_label.config(text=zh)

            # 停留時間依文字長度抓一個大概的閱讀時間，太短的話根本來不及看完；
            # 但如果後面已經堆了好幾句還沒顯示（代表進度落後了），就縮短停留時間
            # 趕快往下顯示，不然字幕會越拖越遠、追不上目前實際播到哪裡
            backlog = sum(1 for item in self._pending_caption if item[0] <= now)
            hold_sec = max(1.2, len(zh) / 7.0)
            if backlog >= 1:
                hold_sec = min(hold_sec, 1.0)
            self._next_caption_switch = now + hold_sec

        self.root.after(200, self._poll_captions)


class TranscriptWindow:
    """嵌在主視窗右側的逐字稿面板（不再是獨立視窗，而是跟左邊播放畫面共用同一個
    視窗、中間用可拖曳的分隔線隔開，拖動分隔線就能自由調整兩邊比例）。

    可以用滑鼠滾輪往回翻頁看之前的內容；只有捲軸還停在最底部時，
    才會在新的一句出現時自動跟著捲到最新內容，避免你正在往回看時被強制拉走。
    """

    def __init__(
        self,
        container: tk.Frame,
        transcript_queue: "queue.Queue",
        episode_title: str = "",
    ):
        self.container = container
        self.transcript_queue = transcript_queue
        self._pending = []  # [(release_at, ja, zh)]

        self.container.configure(bg="#111111")

        header_text = episode_title or "中日對照逐字稿"
        tk.Label(
            self.container, text=header_text, fg="white", bg="#111111",
            font=("Microsoft JhengHei", 14, "bold"),
        ).pack(pady=(10, 4))

        self.text = scrolledtext.ScrolledText(
            self.container, wrap="word", bg="#1a1a1a", fg="white",
            insertbackground="white", font=("Microsoft JhengHei", 12),
            state="disabled", borderwidth=0,
        )
        self.text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.text.tag_configure("ja", foreground="#bbbbbb")
        self.text.tag_configure("zh", foreground="#ffd54a", font=("Microsoft JhengHei", 13, "bold"))

        self.container.after(200, self._poll)

    def _near_bottom(self) -> bool:
        return self.text.yview()[1] >= 0.98

    def _poll(self):
        try:
            while True:
                self._pending.append(self.transcript_queue.get_nowait())
        except queue.Empty:
            pass

        now = time.time()
        ready = [item for item in self._pending if item[0] <= now]
        if ready:
            self._pending = [item for item in self._pending if item[0] > now]
            ready.sort(key=lambda item: item[0])  # 照講話的先後順序寫進逐字稿

            should_scroll = self._near_bottom()
            self.text.configure(state="normal")
            for _, ja, zh in ready:
                self.text.insert("end", ja + "\n", "ja")
                self.text.insert("end", zh + "\n\n", "zh")
            self.text.configure(state="disabled")
            if should_scroll:
                self.text.see("end")

        self.container.after(200, self._poll)


SETTINGS_PATH = Path(__file__).parent / "last_settings.json"
INPUT_DEFAULT_LABEL = "系統預設輸出裝置（無虛擬音源分離）"
OUTPUT_DEFAULT_LABEL = "系統預設（可能會有回音問題）"


def main():
    global DISPLAY_DELAY_SEC

    print("=== 即時中日對照字幕（Gemini API / 本機語言模型） ===")

    from tkinter import filedialog, messagebox
    from startup_gui import run_startup_dialog, load_last_settings, save_last_settings, index_of_name
    import pygetwindow as gw
    import win32process

    cue_data = None

    last = load_last_settings(SETTINGS_PATH)

    candidates = [
        w for w in gw.getAllWindows()
        if w.title.strip() and w.visible and w.width > 200 and w.height > 150
    ]
    if not candidates:
        print("找不到可以擷取的視窗，結束程式。")
        sys.exit(0)
    window_titles = [w.title for w in candidates]

    _pa_probe = pyaudio.PyAudio()
    try:
        loopback_devices = list(_pa_probe.get_loopback_device_info_generator())
    except Exception as e:
        print(f"列出擷取來源失敗，使用系統預設：{e}")
        loopback_devices = []
    _pa_probe.terminate()
    input_names = [INPUT_DEFAULT_LABEL] + [d["name"] for d in loopback_devices]

    output_candidates = []
    try:
        hostapis = sd.query_hostapis()
        wasapi_idx = next((i for i, h in enumerate(hostapis) if "WASAPI" in h["name"]), None)
        for i, d in enumerate(sd.query_devices()):
            if d["max_output_channels"] <= 0:
                continue
            if wasapi_idx is not None and d["hostapi"] != wasapi_idx:
                continue
            output_candidates.append((i, d["name"]))
    except Exception as e:
        print(f"列出播放裝置失敗，使用系統預設：{e}")
    output_names = [OUTPUT_DEFAULT_LABEL] + [name for _, name in output_candidates]

    # 翻譯引擎：環境變數 TRANSLATION_BACKEND 優先，其次是上次的選擇，都沒有就用 Gemini
    try:
        preferred_backend = resolve_translation_backend(
            os.environ.get("TRANSLATION_BACKEND") or last.get("translation_backend")
        )
    except RuntimeError as e:
        print(f"{e}，改用 Gemini API。")
        preferred_backend = TRANSLATION_BACKEND_GEMINI
    backend_labels = [TRANSLATION_BACKEND_LABELS[name] for name in TRANSLATION_BACKENDS]

    dialog = run_startup_dialog(
        window_titles=window_titles,
        input_device_names=input_names,
        output_device_names=output_names,
        default_delay=last.get("delay", DISPLAY_DELAY_SEC),
        default_episode_title=(cue_data.get("episode_title") if cue_data else None) or last.get("episode_title", ""),
        # 讀取預先轉錄字幕檔模式不會用到翻譯，就不顯示翻譯引擎選項
        model_options=backend_labels if cue_data is None else None,
        model_label="翻譯引擎（本機語言模型需先啟動 llama-server，見 README）",
        default_window_index=index_of_name(window_titles, last.get("window_title")),
        default_input_index=index_of_name(input_names, last.get("input_device")),
        default_output_index=index_of_name(output_names, last.get("output_device")),
        default_model_index=TRANSLATION_BACKENDS.index(preferred_backend),
    )
    if dialog is None:
        print("已取消。")
        sys.exit(0)
    translation_backend = (
        TRANSLATION_BACKENDS[dialog["model_index"]] if dialog.get("model_index") is not None else preferred_backend
    )

    chosen_window = candidates[dialog["window_index"]]
    hwnd = chosen_window._hWnd
    window_title = chosen_window.title
    _, pid = win32process.GetWindowThreadProcessId(hwnd)
    print(f"已選擇視窗：{window_title}")

    episode_title = dialog["episode_title"]
    DISPLAY_DELAY_SEC = dialog["delay"]
    print(f"延遲播放：{DISPLAY_DELAY_SEC:.1f} 秒")

    input_device_was_default = dialog["input_index"] == 0
    input_device = loopback_devices[dialog["input_index"] - 1] if dialog["input_index"] > 0 else None
    if input_device is None:
        # 選「系統預設」時要先在這裡把實際裝置解出來（而不是留給 AudioCapture
        # 自己晚點才解析），這樣才能在建立 audio_ring 之前就知道它原始取樣率是多少
        try:
            _pa_probe2 = pyaudio.PyAudio()
            input_device = _pa_probe2.get_default_wasapi_loopback()
            _pa_probe2.terminate()
        except OSError:
            print("找不到系統預設輸出裝置的 loopback，請確認 Windows 音效輸出裝置正常運作。")
            sys.exit(1)
    output_device = output_candidates[dialog["output_index"] - 1][0] if dialog["output_index"] > 0 else None

    save_last_settings(SETTINGS_PATH, {
        **last,
        "window_title": window_title,
        "episode_title": episode_title,
        "delay": DISPLAY_DELAY_SEC,
        "input_device": INPUT_DEFAULT_LABEL if input_device_was_default else input_device["name"],
        "output_device": output_names[dialog["output_index"]] if dialog["output_index"] == 0 else output_candidates[dialog["output_index"] - 1][1],
        "translation_backend": translation_backend,
    })

    # 先把翻譯引擎建好（Gemini 金鑰、本機模型服務有問題的話，在動到來源程式的音訊
    # 輸出設定之前就先結束，不會留下要自己手動切回來的設定）
    translator = None
    if cue_data is None:
        try:
            translator = Translator(translation_backend)
        except RuntimeError as e:
            print(f"翻譯引擎初始化失敗：{e}")
            sys.exit(1)

    using_virtual_cable = bool(input_device and "CABLE" in input_device["name"].upper())
    original_output_device = None
    source_process_name = None

    if using_virtual_cable:
        try:
            import psutil
            source_process_name = psutil.Process(pid).name()
        except Exception:
            source_process_name = None

        if source_process_name and SOUND_VOLUME_VIEW_PATH.exists():
            original_output_device = get_app_output_device(source_process_name)
            if set_app_output_device(source_process_name, "CABLE Input (VB-Audio Virtual Cable)"):
                print(f"已自動把 {source_process_name} 的輸出裝置切到虛擬音訊線。")
                print("提醒：切換裝置指令不會影響「已經在播放」的分頁——")
                print("請去瀏覽器重新整理一次影片頁面（或暫停再重新播放），聲音才會真的改道過去，不然會抓到全程靜音、字幕不會跑出來。")
            else:
                print("自動切換輸出裝置失敗，可能需要自己在 Windows 音量混音器裡手動設定。")
        else:
            print("找不到 SoundVolumeView 或來源程式名稱，請自己在 Windows 音量混音器裡把")
            print("瀏覽器輸出裝置設定成「CABLE Input」。")
        print("使用虛擬音訊線擷取，不需要調低來源音量。")
    # 不再自動調低來源音量了；沒用虛擬音訊線分開的話，背景音問題需要自己在
    # Windows 音量混音器裡手動調整（見 duck_process_volume 函式，保留著沒刪，
    # 之後想要的話可以自己在這裡重新呼叫）

    buffer_seconds = DISPLAY_DELAY_SEC + BUFFER_MARGIN_SEC
    frame_buffer = FrameBuffer(max_seconds=buffer_seconds)
    # 用擷取裝置的原始取樣率存延遲緩衝，不要用 TARGET_SR（那是 16kHz，給 Whisper
    # 用的降頻版本）——不然延遲播放出來的聲音品質會被拖到跟語音辨識輸入一樣低
    native_sr = int(input_device["defaultSampleRate"])
    audio_ring = AudioRingBuffer(sample_rate=native_sr, max_seconds=buffer_seconds)

    utterance_queue: "queue.Queue" = queue.Queue()
    caption_queue: "queue.Queue" = queue.Queue()
    transcript_queue: "queue.Queue" = queue.Queue()

    date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 跨執行緒共用的暫停旗標，播放視窗上的暫停鍵、以及擷取/辨識/翻譯/播放
    # 執行緒全部共用同一個，按一下全部一起停住
    pause_state = PauseState()
    screen_capture = ScreenCapture(hwnd, frame_buffer, pause_state=pause_state)

    worker = None
    raw_audio_path = None
    cue_log_path = None

    if cue_data is not None:
        # 讀取預先轉錄字幕檔模式：不需要即時辨識，AudioCapture 只負責延遲播放
        # 用的音訊緩衝（enable_vad=False），不錄音、不跑 VAD；逐字稿內容已經
        # 全部確定了，一次直接寫完，不用像即時模式那樣靠 Worker 一句一句累積
        audio_capture = AudioCapture(
            utterance_queue, audio_ring=audio_ring, device=input_device,
            record_path=None, pause_state=pause_state, enable_vad=False,
        )
        cue_log_path = TRANSCRIPT_DIR / f"transcript_{date_str}.txt"
        with open(cue_log_path, "w", encoding="utf-8") as f:
            if episode_title:
                f.write(f"#TITLE: {episode_title}\n\n")
            for c in cue_data["cues"]:
                ts = time.strftime("%H:%M:%S", time.gmtime(c["start"]))
                f.write(f"[{ts}] JP: {c['ja']}\n")
                f.write(f"[{ts}] ZH: {c.get('zh', '')}\n\n")
        print(f"逐字稿將儲存於：{cue_log_path}")
    else:
        # 逐字稿、完整錄音共用同一個時間戳，事後才找得到「這份錄音對應哪份逐字稿」
        raw_audio_path = TRANSCRIPT_DIR / f"transcript_{date_str}_audio.wav"
        audio_capture = AudioCapture(
            utterance_queue, audio_ring=audio_ring, device=input_device,
            record_path=raw_audio_path, pause_state=pause_state,
        )
        worker = Worker(
            utterance_queue, caption_queue, transcript_queue, episode_title,
            date_str=date_str, pause_state=pause_state, translator=translator,
        )

    # 延遲聲音一定要從「不是被擷取來源」的裝置播出來，不然會變成回音疊加的無限循環
    audio_player = DelayedAudioPlayer(audio_ring, device=output_device, pause_state=pause_state)

    screen_capture.start()
    audio_capture.start()
    audio_player.start()
    if worker is not None:
        worker.start()

    root = tk.Tk()
    root.title(f"即時中日對照字幕 - {episode_title}" if episode_title else "即時中日對照字幕")
    root.configure(bg="black")
    root.minsize(640, 400)

    # 合併成一個視窗：播放畫面（左）+ 逐字稿（右）用 PanedWindow 隔開，中間那條
    # 分隔線可以直接拖曳調整兩邊比例，初始比例沿用原本兩個獨立視窗的大小分配
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    transcript_w = min(520, screen_w // 3)
    player_w = max(480, screen_w - transcript_w - 60)
    root.geometry(f"{screen_w - 40}x{screen_h - 100}+20+20")

    paned = tk.PanedWindow(
        root, orient="horizontal", sashwidth=6, sashrelief="raised",
        bg="#333333", showhandle=False,
    )
    left_frame = tk.Frame(paned, bg="black")
    right_frame = tk.Frame(paned, bg="#111111")
    paned.add(left_frame, minsize=320, width=player_w)
    paned.add(right_frame, minsize=200, width=transcript_w)

    # PlayerWindow 會先把控制按鈕列 pack 進 root 最上方，接著 paned 才 pack 進來
    # 填滿剩下的空間，上下順序才會對（按鈕列要固定在最頂端，不會被分隔線蓋住）
    cue_playback = (
        {"cues": cue_data["cues"], "transcript_queue": transcript_queue} if cue_data is not None else None
    )
    PlayerWindow(
        root, left_frame, frame_buffer, caption_queue, episode_title,
        pause_state=pause_state, screen_capture=screen_capture, audio_capture=audio_capture,
        cue_playback=cue_playback,
    )
    paned.pack(fill="both", expand=True)

    TranscriptWindow(right_frame, transcript_queue, episode_title)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        screen_capture.stop()
        audio_capture.stop()
        audio_player.stop()
        if worker is not None:
            worker.stop()

        if source_process_name:
            # 結束時把來源程式的輸出裝置切回原本的，不留下你要自己手動改回去的麻煩
            if original_output_device:
                if set_app_output_device(source_process_name, original_output_device):
                    print(f"已把 {source_process_name} 的輸出裝置切回原本的「{original_output_device}」。")
            else:
                print(f"沒能記住 {source_process_name} 原本的輸出裝置，可能需要自己在音量混音器裡切回來。")

        if worker is not None:
            worker.join(timeout=5)  # 等背景執行緒真的寫完檔案，再去讀來整理
        # 完整錄音的 WAV 檔要等 AudioCapture 執行緒真的跑完、把檔案關閉寫入磁碟，
        # 才能安全地拿去重新辨識——不然可能讀到還沒寫完、不完整的檔案
        audio_capture.join(timeout=5)
        print("已停止擷取，逐字稿已存檔。")

        final_log_path = worker.log_path if worker is not None else cue_log_path
        try:
            polished_path = polish_transcript(final_log_path) if final_log_path else None
        except Exception as e:
            polished_path = None
            print(f"整理逐字稿時發生錯誤（原始逐字稿不受影響）：{e}")

        if polished_path:
            print(f"已產生整理過、方便閱讀的完整逐字稿：{polished_path}")
        else:
            print("這次沒有擷取到語句，不產生整理版逐字稿。")

        if worker is not None:
            # 讀取預先轉錄字幕檔模式已經有完整準確的逐字稿了，不需要再事後
            # 重新辨識一次
            if not raw_audio_path.exists():
                print("這次沒有按下「開始錄製」，沒有完整錄音可以拿去重新整理。")
            else:
                try:
                    notebooklm_style_path = rebuild_transcript_from_full_audio(
                        raw_audio_path, worker.asr, worker.translator, worker.glossary, episode_title
                    )
                except Exception as e:
                    notebooklm_style_path = None
                    print(f"用完整錄音重新整理逐字稿時發生錯誤（不影響前面已經產生的逐字稿）：{e}")

                if notebooklm_style_path:
                    print(f"已產生用完整錄音重新辨識、準確度更好的逐字稿：{notebooklm_style_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # 打包成 exe 的情況下保護用，一般直接跑 python 也不影響
    main()
