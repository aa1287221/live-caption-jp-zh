"""
startup_gui.py
共用的「開始擷取前設定」圖形化視窗，給 live_caption.py / live_caption_gemini.py /
live_caption_openrouter.py 共用，取代原本一個一個在終端機打數字/文字的流程。

只負責「畫面 + 收集使用者選了什麼」，不碰擷取視窗、音訊裝置、翻譯 API 這些
實際邏輯——那些還是留在各自的主程式裡，這裡回傳的只是純資料（索引/字串），
由呼叫端自己對應回真正的視窗代碼/裝置物件。
"""

import json
import tkinter as tk
from pathlib import Path


FONT = ("Microsoft JhengHei", 11)
FONT_SMALL = ("Microsoft JhengHei", 9)
FONT_BOLD = ("Microsoft JhengHei", 11, "bold")


def _section_label(parent, text):
    tk.Label(parent, text=text, font=FONT_BOLD, anchor="w").pack(fill="x", pady=(12, 2))


def _labeled_listbox(parent, items, height=5, exportselection=False, default_index=0):
    frame = tk.Frame(parent)
    frame.pack(fill="x", pady=(0, 2))
    scrollbar = tk.Scrollbar(frame, orient="vertical")
    listbox = tk.Listbox(
        frame, font=FONT, height=height, exportselection=exportselection,
        yscrollcommand=scrollbar.set, selectmode="browse",
    )
    scrollbar.config(command=listbox.yview)
    scrollbar.pack(side="right", fill="y")
    listbox.pack(side="left", fill="both", expand=True)
    for item in items:
        listbox.insert("end", item)
    if items:
        idx = default_index if 0 <= default_index < len(items) else 0
        listbox.selection_set(idx)
        listbox.see(idx)
    return listbox


# ---- 記住上次選項 ----

def load_last_settings(path: Path) -> dict:
    """讀取上次的設定；檔案不存在或壞掉就當作沒有紀錄。"""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_last_settings(path: Path, settings: dict):
    try:
        path.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # 記錄失敗不影響本次執行，下次頂多沒記到而已


def index_of_name(items: list[str], name: str | None, fallback: int = 0) -> int:
    """在清單裡找跟上次記住的名稱一樣（或部分吻合）的項目，回傳索引；找不到就用預設值。"""
    if not name:
        return fallback
    for i, it in enumerate(items):
        if it == name:
            return i
    for i, it in enumerate(items):
        if name in it or it in name:
            return i
    return fallback


def run_startup_dialog(
    window_titles: list[str],
    input_device_names: list[str],
    output_device_names: list[str],
    default_delay: float,
    default_episode_title: str = "",
    model_options: list[str] | None = None,
    model_label: str = "翻譯模型",
    default_window_index: int = 0,
    default_input_index: int = 0,
    default_output_index: int = 0,
    default_model_index: int = 0,
) -> dict | None:
    """顯示一個視窗，把原本要在終端機一個一個輸入的設定值都集中在這裡選。

    input_device_names / output_device_names 的第一個選項固定代表「系統預設」，
    呼叫端請自行在清單最前面塞一個「系統預設」的字串，選到索引 0 時自己視為 None。

    回傳一個 dict：
        {
            "window_index": int,
            "episode_title": str,
            "delay": float,
            "input_index": int,
            "output_index": int,
            "model_index": int | None,   # 沒給 model_options 就是 None
        }
    使用者按取消或關閉視窗則回傳 None。
    """
    result = {}
    root = tk.Tk()
    root.title("開始擷取前設定")
    root.geometry("620x760")

    outer = tk.Frame(root)
    outer.pack(fill="both", expand=True, padx=16, pady=10)

    # ---- 視窗選擇 ----
    _section_label(outer, "① 要擷取畫面的視窗（通常是你的瀏覽器）")
    window_listbox = _labeled_listbox(outer, window_titles, height=6, default_index=default_window_index)

    # ---- 節目名稱 / 延遲秒數 ----
    row = tk.Frame(outer)
    row.pack(fill="x", pady=(12, 2))

    left = tk.Frame(row)
    left.pack(side="left", fill="x", expand=True, padx=(0, 8))
    tk.Label(left, text="② 節目/集數名稱（可留空）", font=FONT_BOLD, anchor="w").pack(fill="x")
    title_entry = tk.Entry(left, font=FONT)
    title_entry.pack(fill="x", pady=(2, 0))
    title_entry.insert(0, default_episode_title)

    right = tk.Frame(row)
    right.pack(side="left", fill="x")
    tk.Label(right, text="③ 延遲秒數", font=FONT_BOLD, anchor="w").pack(fill="x")
    delay_entry = tk.Entry(right, font=FONT, width=8)
    delay_entry.pack(pady=(2, 0))
    delay_entry.insert(0, f"{default_delay:.0f}")

    # ---- 音訊輸入 ----
    _section_label(
        outer,
        "④ 音訊擷取來源（有裝虛擬音訊線的話，選含 CABLE 字樣的那項）",
    )
    input_listbox = _labeled_listbox(outer, input_device_names, height=4, default_index=default_input_index)

    # ---- 音訊輸出 ----
    _section_label(outer, "⑤ 延遲聲音要播放到哪個裝置（建議選耳機，跟④不同裝置）")
    output_listbox = _labeled_listbox(outer, output_device_names, height=4, default_index=default_output_index)

    warn_label = tk.Label(outer, text="", font=FONT_SMALL, fg="#c0392b", anchor="w")
    warn_label.pack(fill="x")

    def _refresh_warning(*_):
        try:
            in_name = input_device_names[input_listbox.curselection()[0]]
            out_name = output_device_names[output_listbox.curselection()[0]]
        except IndexError:
            warn_label.config(text="")
            return
        if in_name != "系統預設輸出裝置（無虛擬音源分離）" and in_name == out_name:
            warn_label.config(text="⚠ ④跟⑤選了同一個裝置，會造成回音疊加，請選不同的輸出裝置。")
        else:
            warn_label.config(text="")

    input_listbox.bind("<<ListboxSelect>>", _refresh_warning)
    output_listbox.bind("<<ListboxSelect>>", _refresh_warning)

    # ---- 翻譯模型（選用） ----
    model_listbox = None
    if model_options:
        _section_label(outer, f"⑥ {model_label}")
        model_listbox = _labeled_listbox(outer, model_options, height=6, default_index=default_model_index)

    # ---- 按鈕 ----
    btn_row = tk.Frame(outer)
    btn_row.pack(fill="x", pady=(16, 0))

    def confirm():
        if not window_listbox.curselection():
            warn_label.config(text="⚠ 請先選擇要擷取的視窗。")
            return
        try:
            delay = max(1.0, float(delay_entry.get().strip() or default_delay))
        except ValueError:
            warn_label.config(text="⚠ 延遲秒數請輸入數字。")
            return

        result["window_index"] = window_listbox.curselection()[0]
        result["episode_title"] = title_entry.get().strip()
        result["delay"] = delay
        result["input_index"] = input_listbox.curselection()[0] if input_listbox.curselection() else 0
        result["output_index"] = output_listbox.curselection()[0] if output_listbox.curselection() else 0
        result["model_index"] = (
            model_listbox.curselection()[0] if model_listbox and model_listbox.curselection() else None
        )
        root.destroy()

    def cancel():
        result.clear()
        root.destroy()

    tk.Button(btn_row, text="開始", font=FONT_BOLD, command=confirm, width=12).pack(side="right")
    tk.Button(btn_row, text="取消", font=FONT, command=cancel, width=12).pack(side="right", padx=(0, 8))

    window_listbox.bind("<Double-Button-1>", lambda e: confirm())
    root.protocol("WM_DELETE_WINDOW", cancel)

    root.mainloop()
    return result or None


def show_blocking_notice(title: str, message: str):
    """跳出一個「確定」按鈕的視窗，按下前會擋住後續流程繼續執行。

    取代原本終端機的 input("按 Enter 繼續：")，例如提醒使用者去重新整理瀏覽器頁面。
    """
    root = tk.Tk()
    root.title(title)
    root.geometry("460x180")

    tk.Label(
        root, text=message, font=FONT, wraplength=420, justify="left",
    ).pack(padx=16, pady=16, fill="both", expand=True)

    tk.Button(
        root, text="我已完成，繼續", font=FONT_BOLD,
        command=root.destroy, width=16,
    ).pack(pady=(0, 16))

    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()


def pick_single_window(window_titles: list[str], parent: "tk.Misc | None" = None) -> int | None:
    """執行中途「切換擷取視窗」用的簡化版視窗選擇窗，只選視窗、不用跑完整的
    開始前設定流程。回傳選到的索引；取消或關閉視窗回傳 None。

    有給 parent（現有正在跑 mainloop 的視窗，例如按鈕所在的那個視窗）的話，
    用 Toplevel 掛在它底下、真正 modal 地擋住直到選完再繼續——不能在一個
    Tkinter 應用程式已經在跑 mainloop 的時候，又另外開一個獨立的 tk.Tk()，
    那樣容易出現視窗行為不穩定的狀況。沒給 parent（例如程式一開始、還沒
    進入主迴圈前）才退回自己開一個獨立的 Tk() 視窗。
    """
    result = {}

    if parent is not None:
        root = tk.Toplevel(parent)
        root.transient(parent)
        root.grab_set()  # 真正 modal：擋住不讓你操作背後的視窗，直到選完/取消
    else:
        root = tk.Tk()

    root.title("切換要擷取的視窗")
    root.geometry("520x420")

    outer = tk.Frame(root)
    outer.pack(fill="both", expand=True, padx=16, pady=10)

    _section_label(outer, "選擇要切換過去的視窗")
    listbox = _labeled_listbox(outer, window_titles, height=12)

    btn_row = tk.Frame(outer)
    btn_row.pack(fill="x", pady=(12, 0))

    def confirm():
        if listbox.curselection():
            result["index"] = listbox.curselection()[0]
        root.destroy()

    def cancel():
        result.clear()
        root.destroy()

    tk.Button(btn_row, text="切換", font=FONT_BOLD, command=confirm, width=12).pack(side="right")
    tk.Button(btn_row, text="取消", font=FONT, command=cancel, width=12).pack(side="right", padx=(0, 8))
    listbox.bind("<Double-Button-1>", lambda e: confirm())
    root.protocol("WM_DELETE_WINDOW", cancel)

    if parent is not None:
        parent.wait_window(root)
    else:
        root.mainloop()

    return result.get("index")
