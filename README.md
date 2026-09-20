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
4. 程式會自動把來源視窗的系統音量調低（不是靜音），這樣你主要聽到的是延遲播放的
   聲音；還是會有一點點很小聲的即時背景音，這是技術上無法避免的取捨（完全靜音的話
   程式會抓不到任何聲音），習慣了通常不會太干擾

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

### 2. 確認 Ontime Riva 已安裝

翻譯直接使用 `/project/Ontime-Translator` 的 Riva 模型與 relay，不需要另外
安裝聊天模型，也不需要 API 金鑰。原本用於畫面、音訊擷取與 Whisper
的 Windows Python 環境仍然需要；Riva 由 WSL 提供翻譯服務。

若想手動啟動翻譯專用 relay，在 WSL 執行：

```bash
cd /project/Ontime-Translator
./start-relay.sh --no-preload --port 8765
```

`--no-preload` 會跳過 Ontime 的 ASR 預載，避免與這個專案的 Whisper 重複佔用顯示記憶體；
Riva NMT 仍會載入。一般情況不用先手動啟動，程式會先沿用已就緒或正在啟動的
relay，確認本機連接埠沒有服務後才會自動啟動，結束字幕程式時不會停掉 relay。

### 3. 安裝其餘 Python 套件

```bash
pip install -r requirements.txt
```

### Ontime Riva 設定

`live_caption_gemini.py` 與 `transcribe_audio_file.py` 現在都直接使用 Riva
`POST /v1/translate`；檔名中的 `gemini` 僅為了相容既有捷徑與 Python 匯入路徑。
預設值如下：

| 環境變數 | 預設值 | 用途 |
| --- | --- | --- |
| `ONTIME_RELAY_URL` | `http://127.0.0.1:8765` | relay 服務根網址 |
| `ONTIME_REPO_PATH` | `/project/Ontime-Translator` | WSL 內的 Ontime 專案絕對路徑 |
| `ONTIME_WSL_DISTRO` | `Ubuntu-26.04` | Windows 啟動 relay 時使用的 WSL 發行版 |
| `ONTIME_AUTO_START` | `1` | 本機服務不存在時是否自動啟動 |
| `ONTIME_START_TIMEOUT` | `180` | 等待 Riva NMT 就緒的秒數 |
| `ONTIME_TRANSLATE_TIMEOUT` | `120` | 單次翻譯請求的秒數 |

Windows Python 會用參數陣列直接執行等效的指令：

```powershell
wsl.exe -d Ubuntu-26.04 -e bash /project/Ontime-Translator/start-relay.sh --no-preload --port 8765
```

如果你的 WSL 發行版或專案路徑不同，可在 PowerShell 執行程式前覆寫：

```powershell
$env:ONTIME_WSL_DISTRO = "Ubuntu-26.04"
$env:ONTIME_REPO_PATH = "/project/Ontime-Translator"
python live_caption_gemini.py
python transcribe_audio_file.py recording.wav --title "節目名稱"
```

relay 的 stdout/stderr 持續寫入專案根目錄的 `ontime-riva.log`。啟動失敗時，錯誤訊息會附上
記錄檔路徑與有界限的末尾內容。程式不會中止沿用或自動啟動的 relay。

可先用以下 smoke test 確認 NMT 已就緒並完成一句真實翻譯：

```bash
python3 -c 'from ontime_riva import OntimeRivaClient; c=OntimeRivaClient(); c.ensure_ready(); print(c.translate("今日はいい天気ですね。"))'
```

Riva 是專用翻譯模型，只接收目前的日文句子與受保護的專有名詞。與先前規劃的
指令模型方案相比，它不會參考前句、不會校正 Whisper 辨識結果，也不做全篇編輯潤飾。
這是只使用現有 Ontime Riva 就能直接執行的明確取捨。當某批翻譯被守門拒絕或連線失敗時，
程式會保留句數與順序，並輸出精確標記 `（翻譯失敗，保留日文原文）<原文>`。

## 執行

雙擊桌面捷徑「即時中日字幕」，或手動執行：

```bash
python live_caption_gemini.py
```

流程：
1. 跳出視窗清單，選你要擷取的視窗（通常是瀏覽器）
2. 終端機問節目/集數名稱（可留空）
3. 終端機問要延遲幾秒鐘播放（直接 Enter 用預設 6 秒）
4. 開始擷取，畫面上會看到：左邊延遲播放畫面、右邊可滾輪翻頁的完整逐字稿、
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
- **Riva 守門與專有名詞保護**：拒絕數字等高風險改寫，並在翻譯前後檢查專有名詞標記完整性

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
- **連接埠 8765 已被其他服務佔用**：程式不會強制停掉它或啟動第二個 relay；請停止佔用者，或將
  relay 與 `ONTIME_RELAY_URL` 一起改用其他連接埠
- **relay 有回應但模型未就緒**：檢查 `http://127.0.0.1:8765/health`；必須看到
  `service` 為 `ontime-translator-relay`、`nmt.loaded` 為 `true`、`nmt.state` 為 `ready`，且 `nmt.error` 為空
- **翻譯呼叫失敗／逾時**：先執行上方 smoke test，再檢查 `ontime-riva.log`；模型首次載入較慢時，
  可調高 `ONTIME_START_TIMEOUT`，單次翻譯太慢則調高 `ONTIME_TRANSLATE_TIMEOUT`
