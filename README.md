# 即時中日對照字幕

看日文直播、廣播或會員影片時，即時在畫面下方顯示**日文原文 + 繁體中文翻譯**的個人工具。

選一個視窗（通常是瀏覽器），程式會把那個視窗的畫面和聲音**延遲幾秒播放**，同時在背景做日文語音辨識和翻譯。
因為播放是延遲的，翻譯可以在畫面播到那一刻之前先算好，字幕就能對上說話的瞬間。結束後還會自動整理一份好讀的中日對照逐字稿。

> 這個工具只會擷取你自己電腦上、你選定的那個視窗的畫面和聲音，不會下載影片、不會存取網站帳號、不會繞過任何驗證機制。

## 功能

- **對時字幕**：左邊是延遲播放的畫面，下方是日文原文加中文翻譯，右邊是可以往回捲的完整逐字稿
- **兩種翻譯引擎**，啟動時選：
  - **Gemini API**（預設）：翻譯最自然，會參考前一句上下文；需要 API 金鑰，有免費額度限制
  - **本機語言模型**：用 `llama-server` 或 Ollama 在自己的電腦上翻譯，不用金鑰、沒有額度限制，但品質通常不如 Gemini
- **流暢的畫面**：用 Windows Graphics Capture 擷取，60 fps 的影片約可播到 55 fps
- **自動切換音訊**：開始時自動把瀏覽器的聲音導到虛擬音訊線，關掉程式時自動切回原本的裝置
- **事後整理逐字稿**：按下「開始錄製」後，結束時會用完整錄音重新辨識，再請翻譯模型校正錯字、統一用詞
- **錄音保護**：事後整理只要有一批失敗，完整錄音就會保留，內容不會遺失
- **專有名詞修正**：在 `glossary.json` 寫下常被聽錯的名字，翻譯時會自動改成正確寫法

## 快速開始

需要：Windows 10/11、NVIDIA 顯示卡（建議 8 GB 以上顯存）、Python 3.10 以上。

1. **安裝 CUDA 版 PyTorch**，再裝其他套件（詳見下方「安裝」第 1、2 步）
   ```bash
   pip install -r requirements.txt
   ```
2. **準備 Gemini API 金鑰**：到 https://aistudio.google.com/apikey 申請，把金鑰貼進程式資料夾的 `gemini_api_key.txt`
3. **安裝虛擬音訊線和 SoundVolumeView**（見「安裝」第 4 步）。這一步可以不裝，但不裝的話會錄到電腦上所有聲音
4. **執行**
   ```bash
   python live_caption_gemini.py
   ```

## 使用方式

1. 在瀏覽器打開要看的影片
2. 執行程式，出現「開始擷取前設定」視窗：
   - **① 擷取視窗**：選瀏覽器
   - **② 節目名稱**：可以留空，會寫在逐字稿開頭
   - **③ 延遲秒數**：預設 6 秒。字幕常常來不及出現的話，就調長一點（例如 10～20 秒）
   - **④ 音訊擷取來源**：選含 `CABLE` 的那一項
   - **⑤ 延遲聲音**：選你的耳機或喇叭（**不要**選 CABLE）
   - **⑥ 翻譯引擎**：Gemini API 或本機語言模型（會記住這次的選擇）
3. 按開始後，**到瀏覽器重新整理一次影片頁面**，聲音才會改走虛擬音訊線（原因見「常見問題」）
4. 想要事後整理版的逐字稿，就按上方的「⏺ 開始錄製」
5. 看完直接關掉視窗。程式會把瀏覽器的聲音切回原本的裝置，並產生整理好的逐字稿

### 使用時要注意

- **不要把瀏覽器縮小**：Windows 不會繪製縮小的視窗，畫面會變黑；很多影片網站也會在視窗縮小時自動暫停。
  可以用其他視窗蓋住它，或者把它放在副螢幕上。
- **瀏覽器被其他視窗擋住時**，Chrome / Brave / Edge 預設會為了省電停止繪製影片。
  用下面的參數啟動瀏覽器就不會了（對瀏覽器捷徑按右鍵 →「內容」→ 在「目標」欄位最後面加上）：
  ```
  --disable-features=CalculateNativeWinOcclusion --disable-backgrounding-occluded-windows
  ```
  加完要**先完全關掉瀏覽器**（包括工作列右下角的背景程式），再用這個捷徑重新打開才會生效。

### 產生的檔案

都放在 `transcripts/`：

| 檔案 | 內容 |
| --- | --- |
| `transcript_YYYYMMDD_HHMMSS.txt` | 即時字幕的原始紀錄（邊看邊寫入） |
| `transcript_..._polished.md` | 結束後整理成好讀的中日對照版本 |
| `transcript_..._notebooklm_style.md` | 有按「開始錄製」才會有：用完整錄音重新辨識，並由翻譯模型看過整段上下文、校正後的版本 |
| `transcript_..._audio.wav` | 錄音暫存檔。整理成功後會自動刪除；有任何一批整理失敗時會保留 |

## 安裝

### 1. 安裝 CUDA 版 PyTorch（不做這步會退回 CPU，會非常慢）

到 https://pytorch.org/get-started/locally/ 選 Stable / Windows / Pip / Python，Compute Platform 選你的驅動程式支援的 CUDA 版本（用 `nvidia-smi` 查）。
RTX 50 系列（Blackwell）需要 **CUDA 12.9 以上**，例如：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

驗證（要印出 `True` 和你的顯卡名稱）：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**用 cu130（CUDA 13）版 torch 時要多做一步**：語音辨識用的 faster-whisper 需要 CUDA 12 的 cuBLAS，
不然第一次辨識就會出現 `cublas64_12.dll is not found`。裝完第 2 步後執行：

```bash
pip install nvidia-cublas-cu12
python -c "import glob,os,shutil,importlib.util as u; src=os.path.join(u.find_spec('nvidia.cublas').submodule_search_locations[0],'bin'); dst=u.find_spec('ctranslate2').submodule_search_locations[0]; [shutil.copy2(f,dst) for f in glob.glob(os.path.join(src,'cublas*64_12.dll'))]; print('copied to', dst)"
```

### 2. 安裝其他套件

```bash
pip install -r requirements.txt
```

第一次執行時會自動下載 Whisper 語音辨識模型（約 1.5～3 GB）和 Silero VAD。

### 3. 準備翻譯引擎

**Gemini API（預設）**：到 https://aistudio.google.com/apikey 申請金鑰，擇一提供：

- 在程式資料夾建立 `gemini_api_key.txt`，把金鑰貼進去（雙擊捷徑、exe 的用法要用這個）
- 或在 PowerShell 設定 `$env:GEMINI_API_KEY = "你的金鑰"`

`gemini_api_key.txt` 已經加進 `.gitignore`，不會被上傳。金鑰不要寫進程式碼，也不要貼給別人。
撞到免費額度時，即時字幕會跳過那一句。

**本機語言模型（選用）**：見下方「本機語言模型」。

### 4. 安裝虛擬音訊線 + SoundVolumeView（建議）

沒有虛擬音訊線的話，只能擷取「系統預設輸出裝置」：會錄到電腦上所有的聲音，而且原本的聲音會和延遲播放的聲音疊在一起。

1. **VB-Audio Virtual Cable**（免費）：從 https://vb-audio.com/Cable/ 下載 `VBCABLE_Driver_Pack`，
   解壓後**對 `VBCABLE_Setup_x64.exe` 按右鍵 →「以系統管理員身分執行」**→「Install Driver」，裝完重開機。
2. **NirSoft SoundVolumeView**（免安裝）：從 https://www.nirsoft.net/utils/sound_volume_view.html 下載 64 位元版，
   把 `SoundVolumeView.exe` 放到 `tools\SoundVolumeView.exe`。
   程式靠它在開始時把瀏覽器的輸出切到 `CABLE Input`、結束時切回原本的裝置
   （開始時瀏覽器沒在播放聲音、查不到原本裝置的話，就切回系統預設的播放裝置）。
   `tools\` 不會上傳到 GitHub，換電腦或換資料夾時要記得一起複製。

### 5. 打包成 exe、建立桌面捷徑（選用）

```bash
pip install pyinstaller
pyinstaller --onefile --console --distpath . launcher_gemini.py
```

會在程式資料夾產生 `launcher_gemini.exe`，對它按右鍵 →「傳送到」→「桌面（建立捷徑）」。
這個 exe 只是在同一個資料夾執行 `python live_caption_gemini.py`，所以必須和程式放在一起，
而且終端機輸入 `python` 時要能執行到裝了上面這些套件的那個 Python。

## 專有名詞（glossary.json）

語音辨識常把人名聽錯，例如把「羊宮妃那」寫成「陽宮ひな」。在 `glossary.json` 寫下
「辨識可能寫出來的樣子 → 正確寫法」，翻譯時就會附上一段說明，請翻譯模型改成正確寫法：

```json
{
  "羊宮妃那": "羊宮妃那",
  "ゆみやひな": "羊宮妃那",
  "陽宮ひな": "羊宮妃那"
}
```

- 左邊是聽寫可能出現的寫法，右邊是正確寫法；正確寫法本身也可以列一行，左右填一樣
- 改完**下次啟動**才會生效
- 第一次執行時會自動建立一份預設的清單。這個檔案不會上傳到 GitHub，自己改的內容也不會被 `git pull` 蓋掉
- 這份清單**只給翻譯模型參考，不會交給 Whisper**：實測當成 Whisper 的提示詞時，名字雖然比較容易拼對，
  卻會讓它整句漏聽（一段 3 分鐘的錄音漏了 8 句）

## 本機語言模型

不想花 Gemini 額度、或是想讓逐字稿完全不離開自己的電腦時可以使用。實測翻譯品質通常不如 Gemini
（`qwen2.5:14b` 盲評 48 句：勝 6、平 2、負 40），比較適合在 Gemini 額度用完時當備用。

**方法一：已經有 Ollama**，開程式前先設定：

```powershell
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434"
$env:LOCAL_LLM_MODEL = "qwen2.5:14b"
```

**方法二：讓程式自動啟動 llama-server**。先執行一次安裝腳本，它會下載 llama.cpp 的 Windows CUDA 版和預設模型
（Gemma 4 26B-A4B QAT q4_0，連同 Whisper 顯存峰值約 12.4 GB），驗證 sha256 後寫入 `local_llm_server.json`：

```bash
python setup_local_llm.py            # 預設 CUDA 13.4；CUDA 12 系列顯卡加 --cuda 12.4
```

之後選「本機語言模型」時，如果沒有服務在跑，程式會自動啟動 `llama-server`，關掉程式時也一併關掉。
已經在跑的服務（Ollama、自己手動啟動的 llama-server）會直接使用，不會被關掉。
這支腳本是唯一會下載東西的地方，主程式本身不會下載模型或執行檔。

**方法三：自己啟動 llama-server**：

```bash
llama-server -m 你的模型.gguf --host 127.0.0.1 --port 8766 -c 8192 -np 1 -ngl 99
```

**模型怎麼選**：想要接近 Gemini 的效果，用通用指令模型（例如 Qwen2.5-14B-Instruct、Gemma 4），
程式會送出和 Gemini 完全相同的提示詞。Riva-Translate 這類翻譯專用模型會被自動偵測，
改成逐句翻譯：速度快、顯存小，但不看前後文、也不會校正潤稿。

本機模式另外加了一些保護：單句 15 秒沒翻完就放棄、回覆有問題會重翻、批次被截斷會自動切小重送、
服務停掉時快速失敗並保留錄音。

## 設定（環境變數，都可以不設）

| 環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `TRANSLATION_BACKEND` | 上次的選擇，否則 `gemini` | `gemini` 或 `local` |
| `WHISPER_MODEL` | GPU `large-v3`、CPU `small` | 語音辨識模型，例如 `large-v3-turbo`、`medium` |
| `GEMINI_API_KEY` | 讀 `gemini_api_key.txt` | Gemini 金鑰 |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8766` | 本機模型服務網址 |
| `LOCAL_LLM_MODEL` | 空白 | 模型名稱（Ollama 需要，llama-server 不需要） |
| `LOCAL_LLM_MODE` | `auto` | `auto`、`instruct` 或 `riva` |
| `LOCAL_LLM_LIVE_TIMEOUT` | `15` | 即時字幕單句逾時秒數 |
| `LOCAL_LLM_TIMEOUT` | `120` | 事後整理單次請求逾時秒數 |
| `LOCAL_LLM_STARTUP_WAIT` | `120` | 等你自己啟動的服務載入模型的秒數 |
| `LOCAL_LLM_BATCH_LINES` | `20` | 事後整理每批句數 |

自動啟動 llama-server 的設定（`setup_local_llm.py` 會寫進 `local_llm_server.json`，通常不用自己設）：
`LOCAL_LLM_SERVER_EXE`、`LOCAL_LLM_MODEL_PATH`（沒設就不會自動啟動）、`LOCAL_LLM_SERVER_ARGS`、
`LOCAL_LLM_SPAWN_WAIT`（預設 300 秒）。其餘進階參數見 `local_translation.py`、`llama_server.py`。

## 調整準確度與即時性

- **字幕常常來不及出現**：把啟動時的延遲秒數調長
- **字幕常常漏句**：GPU 預設的 Whisper 模型是 `large-v3`。有些內容（例如背景音樂比較大聲的直播）
  它會把整段判斷成沒人說話而丟掉，可以試試 `WHISPER_MODEL=large-v3-turbo`，速度也比較快。
  反過來說，turbo 在雜音多的片段比較容易出現重複片語之類的幻覺，兩個都可以試試看哪個適合你看的節目
- **一句話常被切成兩半**：把 `live_caption_gemini.py` 裡的 `SILENCE_END_MS`（目前 700）調大；
  想要字幕更快出現就調小

## 常見問題

- **用 CABLE 擷取但字幕都不出來（全程靜音）**：切換瀏覽器的輸出裝置後，只有「新開始播放」的聲音會改走 CABLE。
  到瀏覽器重新整理影片頁面（或暫停再播放）就好。還是不行的話，確認 `tools\SoundVolumeView.exe` 存在，
  或到 Windows 音量混音器手動把瀏覽器輸出設成 `CABLE Input`
- **關掉程式後瀏覽器沒有聲音**：到 Windows 音量混音器把瀏覽器的輸出改回你的耳機或喇叭。
  正常關閉時程式會自動切回，這通常是程式被強制結束才會發生
- **畫面變黑或停住**：瀏覽器被縮小了，或是被擋住時停止繪製，見「使用時要注意」。
  程式會偵測 Windows Graphics Capture 是否一直抓到黑畫面，是的話會自動改用舊的擷取方式
- **畫面卡卡的**：確認用的是最新版（Windows Graphics Capture 擷取）；影片本身只有 30 fps 的話，擷取出來也只有 30 fps
- **中文字幕那行出現日文**：Gemini 偶爾會照抄原文，程式會自動重翻一次；重翻還是日文就只能跳過那句
- **④ 沒有 CABLE 可以選**：VB-Audio Virtual Cable 還沒裝，或裝完還沒重開機
- **`cublas64_12.dll is not found`**：用的是 cu130 版 torch，照「安裝」第 1 步最後那段補上 CUDA 12 的 cuBLAS
- **顯卡沒被用到**：`pip show torch` 的版本號結尾應該是 `+cu130` 之類，而不是 CPU 版
- **視窗清單找不到要的視窗**：視窗不能是縮小的狀態，而且要夠大（太小的視窗會被過濾掉）
- **Gemini 翻譯失敗**：確認金鑰正確；模型名稱不對時，終端機會列出這把金鑰可以用的模型
- **本機模型連不上**：確認 `LOCAL_LLM_BASE_URL` 和服務的網址、連接埠一致；自動啟動失敗時，
  終端機會寫出是哪個路徑找不到，也可以看 `logs/llama-server.log` 的最後幾行
  （通常是顯存不足、模型檔壞掉或參數打錯）
- **本機模型翻得慢、字幕跟不上**：換小一點的模型、確認 `-ngl` 有把模型放上 GPU，或把延遲秒數調長

## 開發與測試

```bash
python -m unittest discover -s tests      # 不需要 GPU、音訊裝置或網路
```

比較 Gemini 和本機模型的翻譯品質（換模型、調參數之後用）：

```bash
python compare_translation_backends.py transcripts/transcript_XXXXXXXX_XXXXXX.txt --judge
python compare_translation_backends.py 樣本.txt --backends local     # 只測本機模型
```

`--judge` 會請 Gemini 盲評每一句（A/B 順序隨機），會用到 Gemini 額度。
完整的驗證流程見 `docs/VERIFY_DUAL_BACKEND.md`。

### 檔案結構

| 檔案 | 用途 |
| --- | --- |
| `live_caption_gemini.py` | 主程式（檔名保留 gemini 是為了相容舊的捷徑，其實兩種翻譯引擎都在這裡） |
| `screen_capture_worker.py` | 在獨立行程裡擷取畫面，透過共享記憶體傳回主程式 |
| `startup_gui.py` | 「開始擷取前設定」視窗 |
| `local_llm.py`、`local_translation.py` | 本機語言模型的連線和翻譯流程 |
| `llama_server.py`、`setup_local_llm.py` | 自動啟動 llama-server、下載模型 |
| `compare_translation_backends.py` | 翻譯品質比較工具 |
| `launcher_gemini.py` | 打包成 exe 用的啟動器 |
