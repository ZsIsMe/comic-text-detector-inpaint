# Solid Inpaint MCP 控制入口

MCP 控制正在執行的 Solid Inpaint GUI。開啟專案、啟動偵測、跳頁與保存都交由 GUI 既有流程執行，改動會顯示在同一個視窗。

先按原本方式啟動 GUI，另安裝選用依賴：

```bash
.venv/bin/python -m pip install -r requirements-mcp.txt
```

在本機 Codex 的專案設定 `.codex/config.toml`，或其他 MCP 客戶端，加入 STDIO server。以下將 `<repo>` 改為本專案絕對路徑：

```toml
[mcp_servers.solid_inpaint]
command = "<repo>/.venv/bin/python"
args = ["<repo>/solid_inpaint_mcp.py"]
cwd = "<repo>"
tool_timeout_sec = 60
```

重新載入 MCP 連線後，先呼叫 `get_app_status`。如果 GUI 尚未執行，工具會回報 unavailable；MCP 不會自行開啟另一個 GUI 或直接修改專案檔。UI 與 MCP 預設選項共用全局 QSettings，已處理頁面重偵測需明確傳入 `allow_redetect=true`。

可用工具：

| 工具 | 用途 |
| --- | --- |
| `get_app_status` | 資料夾、目前頁、頁數、儲存／任務狀態 |
| `open_project(path)` | 直接打開絕對資料夾路徑 |
| `get_detection_settings` | 目前全局模型與氣泡填充預設值 |
| `start_detection(...)` | 指定頁面範圍及選用設定，立即取得 job id |
| `get_job_status(job_id)` | 查詢進度、完成與失敗頁 |
| `list_pages` | 依 GUI 順序列出頁面與校對狀態 |
| `go_to_page(index/name)` | 切至頁碼或完整檔名，先保存目前修改 |
| `get_page_view(kind,max_side,crop)` | 取得原圖／合成／選區預覽 PNG；可用原圖座標 `[x,y,width,height]` 局部放大 |
| `save_project` | 等待目前頁正式保存並回報結果 |

MCP 透過同使用者可存取的 Qt 本地 socket 與視窗通訊。耗時的偵測在 GUI 背景 worker 執行；MCP 工具用 job id 查進度。第一批工具只管理流程與讀取頁面，精確選區編輯仍使用 Computer Use。`get_page_view` 的縮圖可幫助定位，最終校正要回到完整圖與 GUI 預覽確認。正式資料仍由 `project_store.py` 原子保存。

同一份程式路徑由一個 GUI 持有 MCP 控制入口；第二個視窗不會搶走或刪除已運行的端點。要切換控制視窗，先正常關閉原視窗，再重新啟動需要控制的 GUI。

`get_job_status` 查詢由 MCP 啟動的任務；狀態含範圍、完成頁數與失敗檔名。`get_page_view` 回傳原圖尺寸、裁切起點、實際圖像尺寸與縮放倍率，座標不依賴桌面視窗大小。局部視窗或選色等模態操作尚未結束時，控制工具回報 busy，不會自動關閉對話框。

Codex 也可用 CLI 加入服務（將兩個路徑換為實際位置）：

```bash
codex mcp add solid_inpaint -- /absolute/repo/.venv/bin/python /absolute/repo/solid_inpaint_mcp.py
```

安裝後在客戶端重新載入 MCP 服務；當前聊天尚未取得新工具時，不代表程式端連線失敗。配置方式參考 [OpenAI 官方 MCP 說明](https://developers.openai.com/codex/mcp)，SDK 來源為 [MCP 官方 Python SDK](https://github.com/modelcontextprotocol/python-sdk)。
