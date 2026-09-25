# 即時中日對照字幕工具（內建擷取版）

雙擊桌面捷徑「即時中日字幕」-> 選一個視窗（通常是你的瀏覽器）-> 自動延遲播放
那個視窗的畫面+聲音，同時即時做日文語音辨識 + 翻譯成中文，字幕跟延遲畫面對時
顯示，並存成逐字稿。

已針對 **Ryzen 7 5800X3D + RTX 5060 Ti** 調整為優先使用 GPU。

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

### 2. 先獨立啟動本機語言模型服務

翻譯會直接呼叫你自行管理的 `llama-server` completion API。請先啟動已載入
本機翻譯模型的服務，並確認它提供 `POST /completion`；本專案不會啟動、安裝或
依賴其他專案，也不需要 Google API 金鑰。例如服務使用預設連接埠時，先在另一個
終端機啟動你的 `llama-server`，再執行本程式。

### 3. 安裝其餘 Python 套件

```bash
pip install -r requirements.txt
```

### 本機語言模型設定

`live_caption_gemini.py` 與 `transcribe_audio_file.py` 都直接使用本機模型的
`POST /completion`；檔名中的 `gemini` 僅為了相容既有捷徑與 Python 匯入路徑。
目前使用固定的低溫度翻譯取樣設定，實際翻譯品質與延遲取決於你載入的模型及硬體。
預設值如下：

| 環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `LOCAL_LLM_BASE_URL` | `http://127.0.0.1:8766` | `llama-server` 服務根網址 |
| `LOCAL_LLM_TIMEOUT` | `120` | 單次翻譯請求的秒數 |
| `LOCAL_LLM_MAX_TOKENS` | `256` | 單次翻譯的輸出上限 |

若服務不是使用預設網址，可在 PowerShell 執行程式前覆寫：

```powershell
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:8766"
$env:LOCAL_LLM_TIMEOUT = "120"
$env:LOCAL_LLM_MAX_TOKENS = "256"
python live_caption_gemini.py
python transcribe_audio_file.py recording.wav --title "節目名稱"
```

服務連線失敗或 HTTP 429/5xx 時，批次翻譯會依既有次數重試；設定錯誤、格式錯誤、
HTTP 其他 4xx、空回覆或輸出遭截斷時會立即保留日文原文。請確認本機服務可從執行
字幕程式的環境連線，並預留語音辨識與翻譯共用的顯示記憶體。

## 執行

雙擊桌面捷徑「即時中日字幕」，或手動執行：

```bash
python live_caption_gemini.py
```

流程：
1. 開啟「開始擷取前設定」對話框，選擇要擷取的視窗（通常是瀏覽器）
2. 在同一個對話框填寫節目/集數名稱（可留空）與延遲秒數（預設 6 秒），並選擇音訊輸入與輸出裝置
3. 開始擷取，畫面上會看到：左邊延遲播放畫面、右邊可滾輪翻頁的完整逐字稿、
   下方一整條字幕（日文原文 + 中文翻譯）

- 第一次執行會自動下載 Whisper 語音辨識模型跟 Silero VAD（共約 1.5GB）
- 逐字稿即時存到 `transcripts/transcript_YYYYMMDD_HHMMSS.txt`
- 結束播放後自動整理成方便閱讀的 `transcript_YYYYMMDD_HHMMSS_polished.md`
- 若曾開始錄製，結束後會用完整錄音重新辨識、產生逐句雙語稿，成功寫出後刪除暫存 WAV
- 關掉視窗，或在終端機按 Ctrl+C 可結束

## 專有名詞對照表（glossary.json）

容易被聽錯/翻錯的人名、節目名，可以編輯 `glossary.json` 增減，格式：

```json
"日文詞": "希望顯示的中文"
```

不想被翻譯、想保留原文的話，右邊填得跟左邊一樣就好。改完**下次重新啟動**才會生效。

## 辨識與翻譯品質

- **延遲播放**：翻譯有時間在畫面播到那一刻之前先算好（見上方「運作原理」）
- **斷句停頓門檻**（700ms）：避免說話中間換氣被過早切開
- **Whisper 模型**：GPU 預設使用 `large-v3-turbo`，CPU 則使用 `small`
- **專有名詞提示**：`glossary.json` 的詞彙會附加到本機模型提示，協助維持專有名詞原文

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
- **翻譯呼叫失敗／逾時**：確認獨立啟動的 `llama-server` 正在監聽
  `LOCAL_LLM_BASE_URL`，再調高 `LOCAL_LLM_TIMEOUT` 或檢查服務主控台記錄
