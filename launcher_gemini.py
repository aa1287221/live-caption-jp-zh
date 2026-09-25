"""
launcher_gemini.py
一鍵啟動器：雙擊後在同一個資料夾裡執行 live_caption_gemini.py（Gemini API / 本機語言模型，
在啟動畫面選翻譯引擎）。
打包成 exe 後放在跟 live_caption_gemini.py 同一層資料夾即可使用。
"""

import os
import subprocess
import sys


def get_base_dir() -> str:
    if getattr(sys, "frozen", False):
        # 被 PyInstaller 打包成 exe 時，用 exe 自己的所在資料夾
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    base_dir = get_base_dir()
    target = os.path.join(base_dir, "live_caption_gemini.py")

    print("=== 即時中日對照字幕（Gemini API / 本機語言模型）啟動器 ===")
    print(f"程式資料夾：{base_dir}\n")

    if not os.path.exists(target):
        print(f"找不到 live_caption_gemini.py（預期路徑：{target}）")
        print("請確認這個啟動器跟 live_caption_gemini.py 放在同一個資料夾。")
        input("\n按 Enter 鍵關閉此視窗...")
        return

    try:
        result = subprocess.run(["python", target], cwd=base_dir)
    except FileNotFoundError:
        print("找不到 python 指令。")
        print("請確認已安裝 Python，並且在終端機輸入 python 可以正常執行。")
        input("\n按 Enter 鍵關閉此視窗...")
        return

    if result.returncode != 0:
        print(f"\n程式結束，代碼：{result.returncode}")
        input("按 Enter 鍵關閉此視窗...")


if __name__ == "__main__":
    main()
