# 即時中日對照字幕工具（內建擷取版）

雙擊桌面捷徑「即時中日字幕」-> 選一個視窗（通常是你的瀏覽器）-> 自動延遲播放
那個視窗的畫面+聲音，同時即時做日文語音辨識 + 翻譯成中文，字幕跟延遲畫面對時
顯示，並存成逐字稿。

已針對 **Ryzen 7 5800X3D + RTX 5060 Ti** 調整為優先使用 GPU。

翻譯引擎可以在啟動畫面選（會記住上次的選擇）：

| 翻譯引擎 | 需要什麼 | 特色 |
| --- | --- | --- |
| **Gemini API**（預設） | Gemini API 金鑰 | 雲端大模型，會看前一句上下文、事後整理會校正辨識錯字並潤稿；有免費額度限制，逐字稿會送到 Google |
| **本機語言模型** | 自己先啟動 `llama-server`（見下方） | 送出**跟 Gemini 完全相同的提示詞與流程**；不需要金鑰、沒有額度限制、逐字稿不離開你的電腦 |

## 運作原理

1. 你選好視窗後，程式開始持續擷取那個視窗的畫面+聲音，存進一個「幾秒鐘份量」的緩衝區
2. 你實際看到/聽到的，是緩衝區裡「幾秒前」的畫面+聲音（延遲播放），不是即時畫面
3. 因為程式同時也在即時處理最新進來的聲音做辨識+翻譯，翻譯字幕可以在延遲畫面播到
   那一刻之前就先算好，達到「精準對上這個人開口瞬間」的效果
4. 程式目前不會自動降低來源視窗的音量。若使用虛擬音訊線，程式會嘗試將來源程式的
   輸出切到虛擬音訊線；若未使用虛擬音訊線，可能需要自行在 Windows 音量混音器調整
   來源音量，避免即時聲音與延遲播放的聲音重疊

## 安裝

### 1. 先裝有 CUDA 支援的 PyTorch（一定要照這步，不然會退回 CPU 模式）

到 https://pytorch.org/get-started/locally/ 選：
- PyTorch Build: Stable
- OS: Windows / Package: Pip / Language: Python
- Compute Platform: 選你系統目前安裝的 CUDA 版本（用 `nvidia-smi` 看驅動支援到多新的 CUDA）

RTX 5060 Ti 是新一代 Blackwell 顯卡，需要 **CUDA 12.9 以上**的版本，例如：

```bash
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
```

（如果官網有更新版本的 cuXXX 選項，以官網當下給的指令為準；裝完如果版本號不對，
加 `--force-reinstall --no-deps` 強制換版本）

裝完後可以先驗證：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

要印出 `True` 跟你的顯卡名稱，且不要有 `sm_120 is not compatible` 之類的警告才算成功。

### 2. 安裝其餘 Python 套件

```bash
pip install -r requirements.txt
```

### 3. 準備翻譯引擎（兩個都可以裝，執行時再選）

#### A. Gemini API（預設）

到 Google AI Studio（https://aistudio.google.com/apikey）申請免費的 API 金鑰，任選一種方式提供：

- 在程式資料夾建立 `gemini_api_key.txt`，把金鑰貼進去存檔（雙擊捷徑/exe 的用法靠這個檔案）
- 或在 PowerShell 設定：`$env:GEMINI_API_KEY = "你的金鑰"`

金鑰不要寫進程式碼、不要貼給任何人（`*_api_key.txt` 已在 `.gitignore`）。免費額度大約
每天 1,000 次、每分鐘 15 次請求，實際以 Google 官方頁面為準；撞到額度時即時字幕會跳過那一句。

#### B. 本機語言模型（llama-server）

本程式不會幫你啟動、安裝或下載任何模型；請自己先啟動一個 `llama-server`（llama.cpp 的
伺服器程式，可以從 llama.cpp 的 GitHub Releases 下載 Windows CUDA 版），載入 GGUF 模型：

```bash
llama-server -m 你的模型.gguf --host 127.0.0.1 --port 8766 -c 8192 -np 1 -ngl 99
```

- `-c 8192`：context 長度。沒指定的話會用模型的最大值，會吃掉很多顯存
- `-np 1`：只開一個 slot，讓每次請求都能用滿 context（本程式一次只送一個請求）
- `-ngl 99`：盡量把模型放上 GPU

**要選哪種模型？**

- **想要跟 Gemini 一樣的效果（建議）**：用通用指令模型，例如 Qwen2.5-14B-Instruct 的
  Q4_K_M GGUF（約 9 GB 顯存；`live_caption.py`（Ollama 版）用的也是這個系列）。本程式會送出
  跟 Gemini 版一字不差的提示詞：即時字幕帶前一句上下文、事後整理會校正辨識錯字並潤稿、
  預先轉錄用編號批次。16 GB 顯卡要跟 Whisper large-v3-turbo 共用，模型太大會爆顯存。
- **Riva-Translate 這類翻譯專用模型**：程式會自動偵測（模型名稱含 riva 或 Riva 的對話模板），
  改用 Ontime-Translator 的單句翻譯格式、專有名詞用記號保護後換回你設定的顯示文字、
  長句自動切段。速度快、顯存小，但**不會參考前一句、也不會校正或潤稿**，品質可能低於 Gemini；
  啟動時終端機會提醒。

已經裝了 Ollama 的話，也可以用它的 OpenAI 相容介面（沒有 `/props` 時會自動當成通用指令模型；
這條路徑依 Ollama 的 OpenAI 相容 API 實作，主要測試對象是 llama-server）：

```powershell
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434"
$env:LOCAL_LLM_MODEL = "qwen2.5:14b"
```

llama-server 跑在 WSL 裡也可以：在 WSL 用 `--host 0.0.0.0` 啟動，Windows 端透過 WSL2 預設的
localhost 轉送，通常就能用 `http://127.0.0.1:連接埠` 連到（連接埠要跟 `LOCAL_LLM_BASE_URL` 一致）。

## 執行

雙擊桌面捷徑「即時中日字幕」，或手動執行：

```bash
python live_caption_gemini.py
```

流程：
1. 開啟「開始擷取前設定」對話框，選擇要擷取的視窗（通常是瀏覽器）
2. 在同一個對話框填寫節目/集數名稱（可留空）與延遲秒數（預設 6 秒），並選擇音訊輸入與輸出裝置
3. 在「⑥ 翻譯引擎」選 Gemini API 或本機語言模型（會記住這次的選擇；
   也可以用環境變數 `TRANSLATION_BACKEND=gemini` 或 `local` 指定預設值）
4. 開始擷取，畫面上會看到：左邊延遲播放畫面、右邊可滾輪翻頁的完整逐字稿、
   下方一整條字幕（日文原文 + 中文翻譯）

- 翻譯引擎會在開始擷取前先確認可用（Gemini 金鑰、本機模型服務連線與暖機），
  有問題會直接顯示原因並結束，不會動到你的音訊設定
- 第一次執行會自動下載 Whisper 語音辨識模型跟 Silero VAD（共約 1.5GB）
- 逐字稿即時存到 `transcripts/transcript_YYYYMMDD_HHMMSS.txt`
- 結束播放後自動整理成方便閱讀的 `transcript_YYYYMMDD_HHMMSS_polished.md`
- 若曾開始錄製，結束後會用完整錄音重新辨識+翻譯，產生 `transcript_YYYYMMDD_HHMMSS_notebooklm_style.md`，
  成功寫出後刪除暫存 WAV（本機模型服務中途停掉的話會保留 WAV，之後可以用下面的預先轉錄重新處理）
- 關掉視窗，或在終端機按 Ctrl+C 可結束

### 預先轉錄（transcribe_audio_file.py）

內容已經完整存在（時差重播、先下載好的錄音）時，可以先整段轉錄，再用主程式的
「讀取預先轉錄好的字幕檔」模式播放，字幕幾乎零延遲：

```bash
python transcribe_audio_file.py 音檔.mp3 --title "節目名稱"                 # Gemini
python transcribe_audio_file.py 音檔.mp3 --title "節目名稱" --backend local # 本機語言模型
```

## 本機語言模型：跟 Gemini 相同的地方與額外保護

**相同**：提示詞（由同一組函式產生，測試逐字比對兩個引擎送出的內容）、前一句上下文、
事後整理的校正+潤稿格式、預先轉錄的編號批次、`glossary.json`、所有輸出檔案格式與時間軸。

**額外保護**（小模型比較容易出狀況，這些是 Gemini 版沒有的）：

- 即時字幕逾時（預設 15 秒）就放棄那一句，不會拖慢後面的字幕；回覆是空白、照抄提示詞、
  沒翻成中文或長得離譜時，會不帶前一句重翻一次
- 事後整理/預先轉錄的批次如果被截斷、超過 context、格式跑掉或逾時，會自動切成一半重送，
  最後逐句補翻；編號缺漏會補翻，不會留空白也不會對錯行
- 本機批次比 Gemini 小（20 句 vs 60/40 句），所以每批多附前 8 句當上下文
- 服務停掉時連續失敗 3 次就停止（即時字幕暫停 30 秒後自動重試），不會卡住幾個小時

### 設定（環境變數，都可以不設）

| 環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `TRANSLATION_BACKEND` | （上次的選擇，否則 `gemini`） | `gemini` 或 `local` |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8766` | 服務根網址（不含 `/v1`、`/completion`） |
| `LOCAL_LLM_MODE` | `auto` | `auto`（看 `/props` 自動判斷）、`instruct`（`/v1/chat/completions`）、`riva`（`/completion` + Riva 格式） |
| `LOCAL_LLM_MODEL` | （空白） | 模型名稱；llama-server 不需要，Ollama 等多模型服務才需要 |
| `LOCAL_LLM_LIVE_TIMEOUT` | `15` | 即時字幕單句逾時秒數 |
| `LOCAL_LLM_TIMEOUT` | `120` | 批次整理單次請求逾時秒數 |
| `LOCAL_LLM_STARTUP_WAIT` | `120` | 啟動時等待服務載入模型的秒數 |
| `LOCAL_LLM_MAX_TOKENS` | `256` | 單句翻譯的輸出上限 |
| `LOCAL_LLM_BATCH_LINES` | `20` | 事後整理/預先轉錄每批句數（context 夠大、GPU 夠快可以調高） |
| `LOCAL_LLM_BATCH_MAX_TOKENS` | `4096` | 批次輸出上限（也不會超過服務 context 的一半） |
| `LOCAL_LLM_CONTEXT_LINES` | `8` | 每批附帶的前文句數 |
| `LOCAL_LLM_SEGMENT_CHARS` | `120` | 翻譯專用模型的長句切段長度 |

PowerShell 範例：

```powershell
$env:TRANSLATION_BACKEND = "local"
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:8766"
python live_caption_gemini.py
```

## 驗證本機模型「不低於 Gemini」

換模型、換量化、調參數之後，用同一份樣本讓兩個引擎各跑一次，產生並排報告並自動判斷：

```bash
python compare_translation_backends.py transcripts/transcript_20260926_210000.txt --judge
python compare_translation_backends.py 樣本.txt --mode offline      # 比較預先轉錄的編號批次
python compare_translation_backends.py 樣本.txt --backends local    # 只測本機模型（不需要金鑰）
```

- 樣本可以是一行一句的日文，或直接用即時字幕存下來的 `transcripts/transcript_*.txt`
- 報告列出兩邊的問題率（空白、照抄提示、沒翻成中文、過長）、術語命中率、延遲、譯文相似度，
  以及逐句並排對照；`--judge` 會請 Gemini 盲評每一句（A/B 順序隨機）
- 全部通過才回傳 exit code 0：本機問題率不高於 Gemini 2% 以上、術語命中率不低於 Gemini 2% 以上、
  有盲評時本機分數（勝 + 平手/2）≥ 0.45。Gemini 當評審若偏好自己的譯文只會讓本機更難過關
- Gemini 呼叫之間預設間隔 4.5 秒以免撞到每分鐘額度（`--gemini-interval` 可調）

檢查本機服務的 HTTP 介面是否符合程式需求（就緒檢查、模式判斷、截斷與 context 溢出訊號）：

```bash
# PowerShell：先 $env:LLAMA_SERVER_URL = "http://127.0.0.1:8766"
python -m unittest discover -s tests -p test_llama_server_live.py -v
```

其餘自動測試不需要 GPU、音訊裝置或網路：`python -m unittest discover -s tests`

完整的驗證清單（含 Windows 實機步驟與可以直接交給 Claude Code 的審查指示）見 `docs/VERIFY_DUAL_BACKEND.md`。

## 專有名詞對照表（glossary.json）

容易被聽錯/翻錯的人名、節目名，可以編輯 `glossary.json` 增減，格式：

```json
"日文詞": "希望顯示的中文"
```

不想被翻譯、想保留原文的話，右邊填得跟左邊一樣就好。改完**下次重新啟動**才會生效。
（Gemini 與本機通用指令模型用提示詞要求保留原文；翻譯專用模型會用記號保護後換成右邊的文字。）

## 準確度是怎麼拉高的

- **延遲播放**：翻譯有時間在畫面播到那一刻之前先算好（見上方「運作原理」）
- **上下文翻譯**：每句話翻譯時會參考前一句，代名詞/省略主詞這類日文常見狀況準確度好很多
  （Gemini 與本機通用指令模型都有；翻譯專用模型沒有）
- **斷句停頓門檻**（700ms）：避免說話中間換氣被過早切開
- **Whisper 模型**：GPU 預設使用 `large-v3-turbo`，CPU 則使用 `small`
- **事後重新整理**：錄下的完整音訊會重新辨識，翻譯模型看過整批上下文後校正錯字、統一用詞

如果覺得字幕還是常常「等太久才跳出來」，可以把 `live_caption_gemini.py` 裡的
`SILENCE_END_MS`（目前 700）調小一點，抓「準確度」跟「即時性」之間你想要的平衡點；
或是啟動時把延遲秒數設久一點，給翻譯更多緩衝時間。

## 常見問題

- **找不到 loopback 裝置**：確認 Windows 音效設定裡有正常輸出裝置在播放聲音，音量沒靜音
- **顯卡沒被吃到**：重新確認 torch 是否裝成 CUDA 版（`pip show torch` 版本號結尾
  應該是 `+cu130` 之類，而不是純 CPU 版）
- **想換更準的 Whisper 模型**：把 `live_caption_gemini.py` 裡 `WHISPER_MODEL_SIZE` 改成
  `"large-v3"`（需要顯存夠大，8GB 顯卡可能會爆顯存，16GB 應該沒問題）
- **視窗清單找不到你要的視窗**：確認那個視窗沒有被最小化、且視窗大小夠大
  （太小的視窗會被過濾掉，避免清單塞滿一堆工具列小圖示）
- **畫面黑屏**：理論上跟 OBS 視窗擷取用同一套技術，先前測試過對這類會員影音平台
  不會黑屏；如果真的遇到黑屏，把情況告訴我再調整
- **Gemini 翻譯失敗**：確認金鑰正確；型號名稱錯誤時終端機會列出這把金鑰可用的型號
- **本機模型連不上**：確認 llama-server 已啟動、網址與連接埠跟 `LOCAL_LLM_BASE_URL` 一致；
  模型還在載入時程式會等待（最多 `LOCAL_LLM_STARTUP_WAIT` 秒）
- **本機模型翻得慢、字幕跟不上**：換小一點或量化更多的模型、確認 `-ngl` 有把模型放上 GPU，
  或把啟動時的延遲秒數調長
- **終端機提示 context 太小**：用較大的 `-c`（例如 8192）重新啟動 llama-server
