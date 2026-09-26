# 驗證與使用流程：Gemini / 本機語言模型雙引擎

這份文件給 upstream 維護者（以及你的 Claude Code）驗證這個 PR 用。
前半是給人看的操作流程，後半「給 Claude 的驗證指示」可以直接貼給 Claude 執行。

## 這個 PR 做了什麼（一句話）

Gemini 維持預設、行為跟 master 完全一樣；另外新增「本機語言模型」引擎（llama-server），
在啟動畫面「⑥ 翻譯引擎」切換。本機引擎送出跟 Gemini **一字不差**的提示詞，
並加上小模型需要的保護（逾時快速放棄、回覆有問題重翻、批次截斷自動切小、服務停掉快速失敗）。
同時合併了 #2 的 `stop()` 修正。取代 #1。

## 一、自動測試（任何電腦都可以跑，不需要 GPU、音訊裝置或網路）

```bash
python -m unittest discover -s tests
```

預期：`OK (skipped=4)`（4 個是需要真實 llama-server 的測試，預設跳過）。重點測試：

| 測試 | 驗證什麼 |
| --- | --- |
| `GeminiPathTests` | Gemini 的提示詞、temperature 0.3、額度跳過、60/40 句批次、20 秒重試、4.5 秒節奏都跟 master 相同 |
| `PromptParityTests` | 本機引擎送出的 system/user 提示詞跟 Gemini 逐字相同（即時字幕、事後整理、預先轉錄），預先轉錄的輸出檔也相同 |
| `LocalBackendTests` | 不會載入 Google SDK、服務連不上時的提示、截斷自動切批、服務中途停掉保留 WAV、編號缺漏補翻 |
| `test_local_llm.py` | HTTP 協定：chat/completion 兩種格式、`/props` 自動判斷、`/health` 等待、截斷與 context 溢出分類、不走 proxy、拒絕 redirect |

作者端另外做過的驗證（供參考）：
- 把 master 版與這個 PR 版的 `live_caption_gemini.py` 用同一組假 Gemini 回應並排執行，
  送出的每個 prompt、參數、呼叫順序、sleep 序列、所有輸出檔都相同；唯一差異是
  `transcribe_audio_file.py` 改成先建立翻譯引擎再載入 Whisper（金鑰有問題不用等辨識完才發現）。
- 用 llama.cpp 原始碼編出的真實 `llama-server` 跑過即時字幕、預先轉錄、事後整理、比較工具，
  包含 instruct / riva 自動判斷與 context 太小時的真實 400 錯誤。**當時沒有真實模型，翻譯品質未驗證**。

## 二、Windows 實機驗證（需要維護者本人跑）

### A. Gemini 回歸測試（確認跟以前一樣）

1. `pip install -r requirements.txt`，確認 `gemini_api_key.txt` 或 `GEMINI_API_KEY` 還在
2. `python live_caption_gemini.py`（或雙擊原本的捷徑）
3. 啟動畫面最下面多了「⑥ 翻譯引擎」，選 **Gemini API**（預設就是它）
4. 播放一段節目，確認：字幕、延遲播放、逐字稿視窗都跟以前一樣
5. 按「⏺ 開始錄製」錄 2～3 分鐘，關掉視窗，確認有產生 `_polished.md` 與 `_notebooklm_style.md`，WAV 被刪除
6. 再開一次，確認「⑥」記住了上次的選擇
7. （#2 的修正）暫停/繼續、結束時沒有殘留的音訊背景執行緒問題

### B. 本機語言模型

> Gemini 模式不需要下面任何東西；只有要測本機模式才需要準備。

#### B-0. 準備本機模型服務（照順序做，做完一條就可以跳到 B-1）

**步驟 1：先看有沒有現成的 Ollama（有的話最省事，不用下載 llama-server）**

```powershell
ollama list
```

如果清單裡有 `qwen2.5:14b`（本專案 `live_caption.py` 原本就是用它），直接設定環境變數，跳到 B-1：

```powershell
$env:LOCAL_LLM_BASE_URL = "http://127.0.0.1:11434"
$env:LOCAL_LLM_MODEL = "qwen2.5:14b"
$env:LLAMA_SERVER_URL = "http://127.0.0.1:11434"   # 給 B-1 的介面檢查用
```

有 Ollama 但沒有這個模型：`ollama pull qwen2.5:14b`（約 9 GB）。沒有 Ollama 就繼續步驟 2。

**步驟 2：選模型**（看你要驗證什麼）

| 目的 | 模型 | 顯存（約） | 會跑成的模式 |
| --- | --- | --- | --- |
| **驗證「跟 Gemini 一樣看上下文、校正、潤稿」**（這個 PR 的重點，建議） | Qwen2.5-14B-Instruct，Q4_K_M | 9 GB | instruct（與 Gemini 相同提示詞） |
| 顯存不夠（14B 跟 Whisper 一起放不下）時的替代 | Qwen2.5-7B-Instruct，Q4_K_M | 5 GB | instruct |
| 低顯存、只要逐句翻譯 | Riva-Translate-4B-Instruct-v2，Q4_K_M | 3 GB | riva（逐句、不看上下文、不潤稿，**不適合拿來驗證「不低於 Gemini」**） |

- 16 GB 顯卡要跟 Whisper large-v3-turbo（約 2～3 GB）共用；14B + `-c 8192` 大約用掉 13 GB，放得下。
- 程式會自動判斷模式：檔名或模型名稱含 `riva` 就是 riva 模式，其他是 instruct 模式，不用另外設定。

**步驟 3：下載 GGUF 模型檔**

到 Hugging Face 搜尋 `Qwen2.5-14B-Instruct-GGUF`（例如 `bartowski/Qwen2.5-14B-Instruct-GGUF`，
或 Qwen 官方的 `Qwen/Qwen2.5-14B-Instruct-GGUF`），下載 **Q4_K_M** 那個檔案，檔名以頁面為準。
官方版如果把檔案切成 `-00001-of-0000N.gguf` 好幾份，全部下載放同一個資料夾，啟動時 `-m` 指向第 1 份即可。
建議放在固定位置，例如 `C:\llm\models\`。

**步驟 4：下載 llama-server**

1. 到 llama.cpp 的 GitHub Releases（https://github.com/ggml-org/llama.cpp/releases）最新版本
2. 下載檔名含 **`win-cuda`** 的 zip；RTX 50 系列選 CUDA 版本最新的那組（12.8 以上）。
   同一頁如果有 `cudart-...-win-cuda-....zip`，也一起下載
3. 全部解壓到同一個資料夾，例如 `C:\llm\llama.cpp\`，裡面有 `llama-server.exe`，不需要安裝

**步驟 5：啟動服務**（另開一個 PowerShell 視窗，測試期間不要關）

```powershell
cd C:\llm\llama.cpp
.\llama-server.exe -m C:\llm\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 8766 -c 8192 -np 1 -ngl 99
```

- 視窗訊息裡應該看到模型層數被放上 GPU（`offloaded ... layers to GPU` 之類）；如果都在 CPU 上，
  通常是下載成非 CUDA 版，或 cudart 沒有一起解壓
- 用瀏覽器打開 http://127.0.0.1:8766/health ，看到 `{"status":"ok"}` 就是準備好了
  （剛啟動時顯示 `Loading model` 是正常的，等一下）
- 顯存不足（out of memory）：改用 7B 模型，或把 `-c 8192` 降到 `-c 4096`

#### B-1. 驗證

1. 先檢查服務介面（用 Ollama 的話網址換成 `http://127.0.0.1:11434`）：
   ```powershell
   $env:LLAMA_SERVER_URL = "http://127.0.0.1:8766"
   python -m unittest discover -s tests -p test_llama_server_live.py -v
   ```
   預期 4 個測試 OK
2. `python live_caption_gemini.py`，「⑥ 翻譯引擎」選 **本機語言模型**，終端機應該印出
   `本機模型模式：instruct（與 Gemini 相同的提示詞與流程）` 與暖機結果
3. 重做 A 的步驟 4、5，確認字幕與兩份整理稿都有產生
4. 失敗情境：
   - 關掉 llama-server 再啟動程式 → 應該在動到音訊設定前就顯示「無法使用本機語言模型」並結束
   - 錄製中途關掉 llama-server → 字幕變空白但不會卡住；結束後顯示「已保留完整錄音」，WAV 沒被刪
5. 預先轉錄：`python transcribe_audio_file.py 音檔.mp3 --backend local`，再用「讀取預先轉錄字幕檔」播放

### C. 品質是否「不低於 Gemini」（最重要的判斷依據）

拿一份真實逐字稿（A 或 B 產生的 `transcripts/transcript_*.txt` 都可以）：

```bash
python compare_translation_backends.py transcripts/transcript_XXXXXXXX_XXXXXX.txt --judge --limit 60
python compare_translation_backends.py transcripts/transcript_XXXXXXXX_XXXXXX.txt --mode offline --limit 60
```

- 會產生 `..._backend_report.md`：兩個引擎的問題率、術語命中率、延遲、逐句並排
- `--judge` 請 Gemini 盲評（A/B 隨機），會用到 Gemini 額度（60 句約 120 次呼叫，預設每次間隔 4.5 秒）
- 終端機最後印 **PASS** 才代表本機不低於 Gemini；FAIL 的話換更大的模型再測，或繼續用 Gemini（本來就是預設）

## 三、給 Claude 的驗證指示（可直接貼給 Claude Code）

> 你正在幫 upstream 維護者審查這個 PR。請依序做以下事情，並用繁體中文回報結果：
>
> 1. `git fetch` 並 checkout 這個 PR 的分支。執行 `python -m unittest discover -s tests`，回報結果（預期 `OK (skipped=4)`）。
>    這個環境沒有 tkinter/torch 也沒關係，整合測試會自己 stub。
> 2. 確認 Gemini 路徑沒有行為改變：`git diff master -- live_caption_gemini.py transcribe_audio_file.py`。
>    被移動的程式碼（`_polish_with_gemini`、`_translate_cues_gemini`）內容應該跟 master 原本的迴圈相同，
>    只有 user prompt 改成呼叫共用的 `_rebuild_user_prompt` / `_offline_user_prompt`；請確認這兩個函式產生的字串跟 master 原本的 inline 字串完全一致。
>    `GeminiPathTests` 與 `PromptParityTests` 就是在鎖這件事。
> 3. 確認 #2 的修正（刪除重複的 `DelayedAudioPlayer.stop()`）在四個 `live_caption*.py` 都在。
> 4. 檢查 `local_llm.py`、`local_translation.py`：只用標準函式庫、不會啟動或下載任何東西、錯誤訊息不含 prompt 或模型輸出。
> 5. 本機模型環境：先檢查 `ollama list`、`where.exe llama-server`、`nvidia-smi`（顯存）。
>    缺什麼就依「B-0. 準備本機模型服務」的順序告訴維護者要準備什麼，並依顯存從步驟 2 的表格推薦模型；
>    要驗證「不低於 Gemini」請推薦通用指令模型（Qwen2.5-14B-Instruct），不要推薦 Riva-Translate。
>    下載數 GB 的模型或執行檔前先徵求維護者同意；不要修改這個 repo 的程式來遷就環境。
>    如果這台電腦有正在跑的 llama-server（或 Ollama）：設定 `LLAMA_SERVER_URL` 跑 `tests/test_llama_server_live.py`，
>    再用 `compare_translation_backends.py --backends local` 跑一份樣本當 smoke test。
>    如果也有 Gemini 金鑰，跑 `compare_translation_backends.py 樣本 --judge`，把報告的結論（PASS/FAIL 與各項數字）回報。
> 6. 你無法代替維護者驗證 Windows 的畫面擷取、延遲播放與音訊；請把「二、Windows 實機驗證」列成清單請維護者勾選，
>    不要宣稱已驗證。
> 7. 發現問題時，請在 PR 上留言說明重現步驟；不要直接改作者的分支。

## 已知限制

- 本機模式的實際翻譯品質取決於載入的模型；「不低於 Gemini」要以 C 的報告為準。
- Riva-Translate 這類翻譯專用模型會被自動偵測成 `riva` 模式：逐句翻譯、不看上下文、不校正潤稿，品質可能低於 Gemini（啟動時與整理稿檔頭都會註明）。
- `glossary.json` 右邊的對應值（例如 ゆみやひな → 羊宮妃那）在 Gemini 與本機通用模型的提示詞裡沒有被使用（master 原本就是如此，這個 PR 沒有改動 Gemini 提示詞）；只有 `riva` 模式會真的換成右邊的文字。
