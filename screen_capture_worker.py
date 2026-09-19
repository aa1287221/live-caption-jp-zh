"""
screen_capture_worker.py
畫面擷取邏輯的獨立行程（process）版本，給 live_caption_gemini.py 的
ScreenCapture 呼叫。

刻意獨立成這個檔案、只 import 這裡真正需要的東西（不 import
live_caption_gemini.py 本體），是因為 Windows 上開新的 process（spawn 模式）
時，會重新 import 目標函式所在的整個模組——如果擷取邏輯寫在
live_caption_gemini.py 裡面，等於每次開這個擷取 process 都要連 torch、
faster-whisper 這些大型函式庫也一起重新 import 一次，白白拖慢啟動速度、
浪費記憶體。獨立成這個輕量檔案，新開的 process 幾乎是瞬間啟動。

為什麼要搬成獨立 process（而不是原本的執行緒）：
Python 同一個「行程」裡，不管開幾條執行緒，同一時間真正在執行 Python
位元碼的還是只有一個（GIL 限制）。擷取畫面（PrintWindow + 縮放/轉色）這種
吃 CPU 的工作，如果跟語音辨識、翻譯、UI 擠在同一個行程裡搶執行權，遇到大家
都想動的瞬間就會互相卡到，這就是隨機雜音/卡頓的來源——工作管理員的總 CPU
用量看起來不高，因為那是全部核心平均後的數字，實際上是「排隊等執行權」被卡住，
不是「運算量太大算不完」，所以加開執行緒或換更強的 CPU 都解決不了。獨立行程
有自己完全獨立的直譯器和 GIL，兩邊真正並行、不會互搶，才是治本的做法。
"""

import ctypes
import queue
import time

import cv2
import numpy as np

PW_RENDERFULLCONTENT = 0x00000002


def _capture_once(hwnd: int):
    import win32gui
    import win32ui

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return None

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, w, h)
    save_dc.SelectObject(save_bitmap)

    result = ctypes.windll.user32.PrintWindow(
        hwnd, save_dc.GetSafeHdc(), PW_RENDERFULLCONTENT
    )

    img = None
    if result:
        bmpinfo = save_bitmap.GetInfo()
        bmpstr = save_bitmap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8).reshape(
            (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
        ).copy()  # BGRA，複製一份，底下的 DC/Bitmap 馬上就要被釋放掉

    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return img


def run_capture_process(hwnd_shared, max_width, target_fps, frame_queue, stop_event, pause_event):
    """在獨立行程裡執行的主迴圈。跟主行程（語音辨識/翻譯/UI）完全分開跑，
    各自有自己的直譯器和 GIL，彼此不會互搶執行權。
    """
    try:
        import win32api
        import win32process
        # 把這個行程整體的排程優先權調低一點：畫面只是拿來疊字幕用的裝飾內容，
        # 稍微晚一點更新不影響體驗；真正重要的聲音處理/辨識/翻譯留在主行程，
        # 系統資源緊張時，作業系統排程會優先讓主行程先跑，字幕不會被畫面擷取卡到。
        win32process.SetPriorityClass(
            win32api.GetCurrentProcess(), win32process.BELOW_NORMAL_PRIORITY_CLASS
        )
    except Exception:
        pass

    interval = 1.0 / target_fps
    while not stop_event.is_set():
        loop_start = time.time()
        if pause_event.is_set():
            time.sleep(0.1)
            continue
        try:
            hwnd = hwnd_shared.value
            buf = _capture_once(hwnd)
            if buf is not None:
                h, w = buf.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    buf = cv2.resize(
                        buf, (max_width, max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                rgb = cv2.cvtColor(buf, cv2.COLOR_BGRA2RGB)
                try:
                    frame_queue.put_nowait((time.time(), rgb))
                except queue.Full:
                    # 主行程那邊來不及消化時，寧可丟掉這一張畫面，也不要讓
                    # 佇列越積越多、造成畫面延遲越拖越長
                    pass
        except Exception as e:
            print(f"[擷取行程] 處理擷取畫面時發生錯誤：{e}")

        elapsed = time.time() - loop_start
        remaining = interval - elapsed
        if remaining > 0:
            time.sleep(remaining)
