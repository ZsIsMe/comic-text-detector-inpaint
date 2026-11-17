# 準備可選的 ComfyUI 修復環境

以下命令從本倉庫根目錄執行。Windows 用 `.\.venv\Scripts\python.exe`，macOS/Linux 改為 `.venv/bin/python`。`<ComfyUI>` 是含 `main.py` 的實際目錄，`<ComfyPython>` 是該 ComfyUI 使用的 Python；不要填作者電腦路徑，也不要把本工程的 Python 當成 Comfy 的 Python。

沒有 ComfyUI 時，原有 LaMa 導出仍可用。要啟用額外方法，先參照 [ComfyUI 官方安裝說明](https://docs.comfy.org/installation/desktop) 安裝適合使用者系統的版本，再定位實際根目錄、Python 和服務地址；本工具不代替官方 GPU／PyTorch 安裝器。

## 檢查與配置

```powershell
.\.venv\Scripts\python.exe scripts/setup_repair_workflows.py plan --methods klein qwen
.\.venv\Scripts\python.exe scripts/setup_repair_workflows.py check --methods klein qwen --comfy-root "<ComfyUI>" --python "<ComfyPython>" --server http://127.0.0.1:8188
```

`plan` 產生方案，不寫環境；`check` 檢查本機檔案／節點並可查服務，快速存在檢查不是權重雜湊驗證。`verify` 讀取完整模型計算 SHA-256，會花時間。模型、節點固定來源在 `repair_workflows/setup_manifest.json`，兩方法的 API JSON 位於各方法子目錄，所有檔案都隨本工程交付。

在使用者已授權配置環境的範圍內執行：

```powershell
.\.venv\Scripts\python.exe scripts/setup_repair_workflows.py download-models --methods klein qwen --comfy-root "<ComfyUI>"
.\.venv\Scripts\python.exe scripts/setup_repair_workflows.py install-nodes --methods klein qwen --comfy-root "<ComfyUI>" --python "<ComfyPython>" --install-requirements
.\.venv\Scripts\python.exe scripts/setup_repair_workflows.py verify --methods klein qwen --comfy-root "<ComfyUI>" --server http://127.0.0.1:8188
```

下載統一使用 Hugging Face 固定 revision，支援續傳並驗證大小及 SHA-256 後才完成落盤。Klein 主模型可能要求在 HF 接受授權並設定 `HF_TOKEN`；存取失敗要告知使用者，不能換成近似模型。不要在訊息、報告或 Git 中輸出 token。

節點來自公開固定 Git 提交，僅在缺少時安裝；既有不同版本不會 reset、覆蓋或默默升級，應先報告差異。只有明確使用 `--install-requirements` 才會以指定 Comfy Python 安裝節點依賴；工具以既有 torch／torchvision／torchaudio 版本作約束，不自動選 GPU 框架。重新啟動 ComfyUI 後再預檢。已存在模型若損壞或身份不同會保留並報錯，不覆蓋。

配置腳本的 `--model-map` 是 JSON **模型 ID 或檔名 → 實際檔案路徑**，用來複用其他磁碟的權重。這與導出 MCP 的 **方法 ID → unet/clip/vae → Comfy 下拉檔名** 映射不同；後者必須填服務可見名稱。

## 單頁試跑

先用 MCP `check_repair_environment` 檢查選定方法，確認 queue 沒有與使用者工作衝突，再按要求導出一頁。可讀取 `workflow_results` 完整結果及透明 PNG、PSD 圖層進行核對。記錄方法、耗時、服務／節點版本及實測顯存；沒有量測便不要聲稱某張卡一定足夠。首次啟動包含載入和快取建立，與後續熱跑分開記錄。
