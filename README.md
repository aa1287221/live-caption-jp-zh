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

### 2. 安裝 Ollama + 翻譯模型

到 https://ollama.com 下載安裝，裝完在終端機執行：

```bash
ollama pull qwen2.5:14b
```

翻譯的時候程式會即時呼叫本機的 Ollama 服務，執行 `live_caption.py` 前請確認
Ollama 有在背景跑著（開始選單搜尋 Ollama 開一次，或終端機另外跑 `ollama serve`）。

### 3. 安裝其餘 Python 套件

```bash
pip install -r requirements.txt
```

## 執行

雙擊桌面捷徑「即時中日字幕」，或手動執行：

```bash
python live_caption.py
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
- 關掉視窗，或在終端機按 Ctrl+C 可結束

## 專有名詞對照表（glossary.json）

容易被聽錯/翻錯的人名、節目名，可以編輯 `glossary.json` 增減，格式：

```json
"日文詞": "希望顯示的中文"
```

不想被翻譯、想保留原文的話，右邊填得跟左邊一樣就好。改完**下次重新啟動**才會生效。

## 準確度是怎麼拉高的

- **延遲播放**：翻譯有時間在畫面播到那一刻之前先算好（見上方「運作原理」）
- **上下文翻譯**：每句話翻譯時會參考前一句，代名詞/省略主詞這類日文常見狀況準確度好很多
- **拉長斷句停頓門檻**（900ms）：避免說話中間換氣被錯誤切成兩句
- **GPU 上開較大的 Whisper 模型**（medium，16GB 顯卡可自行改成 `large-v3`）
- **本機 LLM 翻譯**（Ollama + Qwen2.5）取代傳統翻譯模型：對口語、停頓、省略主詞這類
  真人講話的狀況處理得好很多，翻起來自然很多

如果覺得字幕還是常常「等太久才跳出來」，可以把 `live_caption.py` 裡的
`SILENCE_END_MS`（目前 900）調小一點，抓「準確度」跟「即時性」之間你想要的平衡點；
或是啟動時把延遲秒數設久一點，給翻譯更多緩衝時間。

## 常見問題

- **找不到 loopback 裝置**：確認 Windows 音效設定裡有正常輸出裝置在播放聲音，音量沒靜音
- **顯卡沒被吃到**：重新確認 torch 是否裝成 CUDA 版（`pip show torch` 版本號結尾
  應該是 `+cu130` 之類，而不是純 CPU 版）
- **想換更準的 Whisper 模型**：把 `live_caption.py` 裡 `WHISPER_MODEL_SIZE` 改成
  `"large-v3"`（需要顯存夠大，8GB 顯卡可能會爆顯存，16GB 應該沒問題）
- **視窗清單找不到你要的視窗**：確認那個視窗沒有被最小化、且視窗大小夠大
  （太小的視窗會被過濾掉，避免清單塞滿一堆工具列小圖示）
- **畫面黑屏**：理論上跟 OBS 視窗擷取用同一套技術，先前測試過對這類會員影音平台
  不會黑屏；如果真的遇到黑屏，把情況告訴我再調整
- **翻譯呼叫失敗/逾時**：確認 Ollama 服務有在背景執行、`ollama list` 裡有
  `qwen2.5:14b`；第一次呼叫要把模型讀進顯存，可能要等 30~90 秒
