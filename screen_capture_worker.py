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
位元碼的還是只有一個（GIL 限制）。擷取畫面（抓圖 + 縮放/轉色）這種
吃 CPU 的工作，如果跟語音辨識、翻譯、UI 擠在同一個行程裡搶執行權，遇到大家
都想動的瞬間就會互相卡到，這就是隨機雜音/卡頓的來源——工作管理員的總 CPU
用量看起來不高，因為那是全部核心平均後的數字，實際上是「排隊等執行權」被卡住，
不是「運算量太大算不完」，所以加開執行緒或換更強的 CPU 都解決不了。獨立行程
有自己完全獨立的直譯器和 GIL，兩邊真正並行、不會互搶，才是治本的做法。

畫面資料怎麼傳回主行程——改用共享記憶體，不用 Queue：
第一版用 multiprocessing.Queue 傳畫面，實測撐不住 1920 寬 / 60fps：Queue
不是零複製的傳輸方式，每一張畫面都要 pickle 序列化、複製進系統管道、主行程
那邊再複製一次解出來，一秒 60 張、每張好幾 MB，等於每秒要擠好幾百 MB 資料量
過這條管道，明顯撐不住，畫面因此掉幀、卡頓，看起來就像聲音畫面對不上（其實
聲音那邊完全正常，只是畫面追不上）。改成用共享記憶體：擷取行程直接把畫面
位元組寫進一塊主行程也看得到的記憶體，完全不用序列化/複製，主行程直接讀。
用 3 個輪替的緩衝格（slot）而不是 1 個，讀寫各自有餘裕，不用加鎖也不會撞在
一起——寫入端寫完一格才把「目前最新是哪一格」這個索引更新，讀取端永遠讀
「已經完整寫完的那一格」，寫入端要繞回來蓋掉同一格之前，讀取端早就讀完/
複製走了（一輪只需要複製幾 MB，遠比一張畫面的產生時間快很多）。
"""

import ctypes
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

PW_RENDERFULLCONTENT = 0x00000002
NUM_SLOTS = 3  # 輪替緩衝格數，3 格給讀寫雙方足夠餘裕，不用額外加鎖


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


# 平均亮度低於這個值就當成「全黑」；連續黑這麼久，就拿 PrintWindow 抓一張比對，
# 確認是 WGC 拿不到畫面（不是影片本身剛好是黑畫面），才切換過去
BLACK_MEAN_THRESHOLD = 3.0
BLACK_FALLBACK_SEC = 2.0


class _SlotWriter:
    """把一張 BGRA 畫面縮放、轉成 RGB，寫進共享記憶體的下一格。
    WGC 的回呼執行緒跟 PrintWindow 迴圈都用這一個，寫入規則只有一份。"""

    def __init__(self, shm_name, slot_nbytes, meta_array, latest_slot, max_width):
        self.shm = shared_memory.SharedMemory(name=shm_name)
        self.slots = np.ndarray((NUM_SLOTS, slot_nbytes), dtype=np.uint8, buffer=self.shm.buf)
        self.slot_nbytes = slot_nbytes
        self.meta_array = meta_array
        self.latest_slot = latest_slot
        self.max_width = max_width
        self.write_slot = 0

    def write(self, bgra: np.ndarray) -> None:
        h, w = bgra.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            bgra = cv2.resize(
                bgra, (self.max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        h, w = rgb.shape[:2]
        nbytes = rgb.nbytes
        if nbytes > self.slot_nbytes:
            # 理論上不會發生（共享記憶體照 max_width 的正方形上限配置，
            # 一般影片視窗不會比它還高），真的遇到就跳過這張避免寫爆記憶體
            print(f"[擷取行程] 畫面 {w}x{h} 超出共享記憶體容量，跳過這一張")
            return
        self.slots[self.write_slot, :nbytes] = rgb.reshape(-1)
        idx = self.write_slot * 3
        self.meta_array[idx] = h
        self.meta_array[idx + 1] = w
        self.meta_array[idx + 2] = time.time()
        self.latest_slot.value = self.write_slot
        self.write_slot = (self.write_slot + 1) % NUM_SLOTS

    def close(self):
        self.shm.close()


def _is_black(bgra: np.ndarray) -> bool:
    return float(bgra[::16, ::16, :3].mean()) < BLACK_MEAN_THRESHOLD


def _run_wgc(hwnd, writer, hwnd_shared, stop_event, pause_event) -> str:
    """用 Windows Graphics Capture 擷取，直到結束、切換視窗或需要退回 PrintWindow。

    WGC 是影片每換一格就主動送一張過來（跟 OBS 視窗擷取同一套），不像
    PrintWindow 要自己定時去抓：定時抓的時間點跟影片換格對不上，就會漏格、
    重複，看起來一頓一頓；PrintWindow 抓一張 Brave 畫面還要 20ms 以上，
    60fps 的影片根本抓不完。

    回傳 "stop"（要結束）、"switch"（視窗換了，重新開始）、"fallback"（WGC
    一直拿到黑畫面，但 PrintWindow 抓得到內容，改用 PrintWindow）、"error"。
    """
    state = {"black_since": None, "fallback": False}

    # 用 git pull 更新卻沒重跑 pip install 的人會缺這個套件或停在舊版，
    # 這裡失敗要回 "error" 讓這個視窗改用 PrintWindow，不能讓整個擷取行程掛掉
    try:
        from windows_capture import WindowsCapture

        capture = WindowsCapture(cursor_capture=False, draw_border=False, window_hwnd=hwnd)
    except ModuleNotFoundError as e:
        if e.name == "windows_capture":
            print(
                "[擷取行程] 找不到 windows-capture 套件，先改用 PrintWindow 擷取（fps 會比較低）。"
                "請執行 pip install -r requirements.txt 安裝，裝好後重新啟動程式。"
            )
        else:
            # 安裝壞掉（套件在、少了裡面的模組）時 pip install -r 會當成已經裝好，安裝提示沒用
            print(f"[擷取行程] 無法使用 Windows Graphics Capture，改用 PrintWindow：{e}")
        return "error"
    except TypeError:
        # 1.x 版沒有 window_hwnd 參數。這時舊版的 .pyd 已經載入，Windows 會鎖住它，
        # 程式還開著就執行 pip install -U 可能會失敗，所以要先關掉程式
        print(
            "[擷取行程] windows-capture 版本太舊（需要 2.0.0 以上），先改用 PrintWindow 擷取"
            "（fps 會比較低）。請先關掉程式，執行 pip install -U windows-capture 更新，再重新啟動程式。"
        )
        return "error"
    except Exception as e:
        print(f"[擷取行程] 無法使用 Windows Graphics Capture，改用 PrintWindow：{e}")
        return "error"

    @capture.event
    def on_frame_arrived(frame, control):
        if stop_event.is_set() or hwnd_shared.value != hwnd or state["fallback"]:
            control.stop()
            return
        if pause_event.is_set():
            return
        try:
            bgra = frame.frame_buffer
            if _is_black(bgra):
                now = time.time()
                if state["black_since"] is None:
                    state["black_since"] = now
                elif now - state["black_since"] >= BLACK_FALLBACK_SEC:
                    state["black_since"] = now
                    probe = _capture_once(hwnd)
                    if probe is not None and not _is_black(probe):
                        state["fallback"] = True
                        control.stop()
                        return
            else:
                state["black_since"] = None
            writer.write(bgra)
        except Exception as e:
            print(f"[擷取行程] 處理擷取畫面時發生錯誤：{e}")

    @capture.event
    def on_closed():
        pass

    try:
        control = capture.start_free_threaded()
    except Exception as e:
        print(f"[擷取行程] 無法使用 Windows Graphics Capture，改用 PrintWindow：{e}")
        return "error"

    while not control.is_finished():
        if stop_event.is_set() or hwnd_shared.value != hwnd:
            control.stop()
            break
        time.sleep(0.05)
    try:
        control.wait()
    except Exception:
        pass

    if stop_event.is_set():
        return "stop"
    if state["fallback"]:
        print("[擷取行程] Windows Graphics Capture 一直抓到黑畫面，改用 PrintWindow 擷取。")
        return "fallback"
    if hwnd_shared.value != hwnd:
        return "switch"
    # 視窗被關掉之類的情況，WGC 會自己結束；稍等一下再重試
    time.sleep(0.5)
    return "switch"


def _run_printwindow(hwnd, writer, hwnd_shared, stop_event, pause_event, target_fps) -> str:
    """PrintWindow 輪詢擷取（舊方法，WGC 用不了時的備案）。回傳 "stop" 或 "switch"。"""
    interval = 1.0 / target_fps
    while not stop_event.is_set():
        if hwnd_shared.value != hwnd:
            return "switch"
        loop_start = time.time()
        if pause_event.is_set():
            time.sleep(0.1)
            continue
        try:
            buf = _capture_once(hwnd)
            if buf is not None:
                writer.write(buf)
        except Exception as e:
            print(f"[擷取行程] 處理擷取畫面時發生錯誤：{e}")
        remaining = interval - (time.time() - loop_start)
        if remaining > 0:
            time.sleep(remaining)
    return "stop"


def run_capture_process(
    hwnd_shared, max_width, target_fps, shm_name, slot_nbytes, meta_array,
    latest_slot, stop_event, pause_event,
):
    """在獨立行程裡執行的主迴圈。跟主行程（語音辨識/翻譯/UI）完全分開跑，
    各自有自己的直譯器和 GIL，彼此不會互搶執行權。

    預設用 Windows Graphics Capture；某個視窗 WGC 一直抓到黑畫面（但
    PrintWindow 抓得到），那個視窗就改用 PrintWindow。切換到別的視窗時，
    會重新先試 WGC。

    shm_name/slot_nbytes：主行程建立好的共享記憶體名稱、每一格的大小上限。
    meta_array：長度 NUM_SLOTS*3 的 double 陣列，依序存每一格的 (height, width,
    timestamp)——寫完畫面位元組後才寫這三個值，最後才更新 latest_slot，
    讀取端看到 latest_slot 換了，才會去讀對應那一格的 meta/畫面，保證讀到的
    一定是完整寫完的一份。
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

    writer = _SlotWriter(shm_name, slot_nbytes, meta_array, latest_slot, max_width)
    printwindow_hwnds = set()  # 這些視窗 WGC 會抓到黑畫面，直接用 PrintWindow
    try:
        while not stop_event.is_set():
            hwnd = hwnd_shared.value
            if hwnd in printwindow_hwnds:
                result = _run_printwindow(hwnd, writer, hwnd_shared, stop_event, pause_event, target_fps)
            else:
                result = _run_wgc(hwnd, writer, hwnd_shared, stop_event, pause_event)
                if result in ("fallback", "error"):
                    printwindow_hwnds.add(hwnd)
            if result == "stop":
                break
    finally:
        writer.close()
