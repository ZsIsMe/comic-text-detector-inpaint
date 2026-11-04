# Solid Inpaint MCP 控制入口

MCP 控制正在執行的 Solid Inpaint GUI。開啟專案、啟動偵測、跳頁、原圖座標選區及保存沿用 GUI 的兩類資料與撤銷流程，改動顯示在同一個視窗。影像內容與去字品質仍由使用者或 AI 逐頁視覺確認。

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

重新載入 MCP 連線後，先呼叫 `list_app_instances` 查看正在執行的視窗及其 `instance_id`、PID、資料夾與目前頁面。只有一個視窗時，原有工具仍可省略 `instance_id`；多個視窗時，每次操作都須傳入目標 `instance_id`，否則回報 `ambiguous_instance`，不會猜測目標。如果 GUI 尚未執行，列表為空，控制工具回報 `unavailable`；MCP 不會自行開啟另一個 GUI 或直接修改專案檔。UI 與 MCP 預設選項共用全局 QSettings，已處理頁面重偵測需明確傳入 `allow_redetect=true`。

可用工具：

| 工具 | 用途 |
| --- | --- |
| `list_app_instances` | 列出本份程式所有 GUI 視窗的 ID、PID 與目前狀態 |
| `get_app_status` | 資料夾、目前頁、頁數、儲存／任務狀態 |
| `open_project(path)` | 直接打開絕對資料夾路徑 |
| `get_detection_settings` | 目前全局模型與氣泡填充預設值 |
| `start_detection(...)` | 指定頁面範圍及選用設定，立即取得 job id |
| `get_job_status(job_id)` | 查詢進度、完成與失敗頁 |
| `list_pages` | 依 GUI 順序列出頁面與校對狀態 |
| `go_to_page(index/name)` | 切至頁碼或完整檔名，先保存目前修改 |
| `get_page_view(kind,max_side,crop)` | 取得原圖／合成／選區預覽 PNG；可用原圖座標 `[x,y,width,height]` 局部放大 |
| `get_review_bundle(max_side,crop,include_mask)` | 一次取得同版原圖及合成圖、可選 Mask，以及頁面 token／座標映射／保存狀態 |
| `apply_page_edits(page_token,operations,...)` | 原圖座標的同頁矩形／多邊形編輯，整批可撤銷；保存後回傳新預覽 |
| `undo_page_edit(page_token,...)` | 撤銷仍對應目前頁面內容的最近一批 MCP 編輯，保存後回傳新預覽 |
| `mark_page_reviewed(page_token,reviewed,advance,...)` | 視覺驗收後保存校對標記，可選擇進入下一頁 |
| `save_project` | 等待目前頁正式保存並回報結果 |

MCP 透過同使用者可存取的 Qt 本地 socket 與視窗通訊。耗時的偵測在 GUI 背景 worker 執行；MCP 工具用 job id 查進度。矩形／套索的精確編輯可透過 MCP 執行；筆刷、魔法棒及局部副本等使用 Computer Use。整頁圖用來掃描漏檢與誤判，小字和邊緣需以足夠解析度的局部圖或 GUI 放大核對；完成修改後仍要核對局部及整頁預覽。正式資料由 `project_store.py` 原子保存，MCP 不接受任意 Mask 陣列覆寫。

每個 GUI 視窗都有獨立的本地控制端點及 `instance_id`，可以同時控制多個視窗。除了 `list_app_instances`，上述每個工具都接受可選的 `instance_id`。例如先從列表依資料夾辨認兩個視窗，再分別呼叫：

```text
get_app_status(instance_id="視窗 A 的 ID")
open_project(path="/absolute/path/to/chapter-b", instance_id="視窗 B 的 ID")
start_detection(instance_id="視窗 B 的 ID")
get_job_status(job_id="剛取得的任務 ID", instance_id="視窗 B 的 ID")
```

從舊版更新後，先保存並重啟 GUI，再於 MCP 客戶端重新載入服務，讓視窗使用新的獨立端點並取得更新後的工具參數。

`instance_id` 在視窗關閉後失效，重開須重新取得；指定已關閉或不存在的 ID 會回報 `unavailable`，不會轉到其他視窗。翻頁、取圖、編輯、保存與任務進度查詢都應沿用同一視窗 ID，頁面 token 及 job id 不能跨視窗使用。單開時省略 ID 的方式保留相容，但需要持續操作同一視窗時，建議仍傳入明確 ID。

多視窗應各自處理不同漫畫資料夾；此功能不提供同一專案的跨程序編輯鎖，同時修改同一資料夾仍可能互相覆蓋。多開也會共用全局偏好設定，並各自使用偵測所需的記憶體及顯存。

`get_job_status` 查詢由 MCP 啟動的任務；狀態含範圍、完成頁數與失敗檔名。`get_page_view` 回傳原圖尺寸、裁切起點、實際圖像尺寸與縮放倍率，座標不依賴桌面視窗大小。局部視窗或選色等模態操作尚未結束時，控制工具回報 busy，不會自動關閉對話框。

## 全話範圍與完成核對

全話偵測使用 `start_detection()`，省略 `end_page` 由 GUI 實際圖片列表取最後頁；明確子範圍才傳起止頁。任務回傳 `project_page_count`、`selected_page_count` 及 `coverage_missing_pages`，最後一欄表示**這次任務未涵蓋**的頁面，不表示它們之前一定沒偵測過。`state=completed` 只代表本次選定範圍完成。整話完成另核對 `list_pages` 的 `page_count`、`processed_count`、`failed_pages`、`unreviewed_pages`，並確認保存無錯誤。

## 同版影像與精確編輯

`get_review_bundle` 回傳 metadata 及原圖、合成圖兩張 PNG；`include_mask=true` 再附選區圖，圖像順序見 `image_order`。所有圖共用同一頁面狀態，座標映射見 `origin`、`scale`、原圖及輸出尺寸。`crop=[x,y,width,height]` 使用原圖像素。`page_token` 是暫時的頁面憑證，不能跨頁、重載或內容變更後沿用；每次重新取得 bundle 後使用最新 token。

先視覺確認目標和底色，再以 `apply_page_edits` 提交同頁的一批操作，例如：

```json
{
  "page_token": "剛看過的頁面 token",
  "operations": [
    {"kind": "restore", "shape": {"rect": [10, 20, 30, 40]}},
    {"kind": "add", "target": "other", "shape": {"polygon": [[100, 100], [130, 105], [120, 140], [95, 135]]}},
    {"kind": "add", "target": "solid", "color_rgb": [0, 0, 0], "shape": {"rect": [200, 200, 30, 40]}}
  ]
}
```

矩形 `[x,y,width,height]` 覆蓋 `x` 至 `x+width-1`、`y` 至 `y+height-1`；多邊形包含邊界像素。`restore` 清除兩類，`add` 加入目標類並移除同位置另一類；`transfer` 只將範圍內另一類的**既有**選區轉入目標類，不補漏選。`solid` 操作明確指定 `color_rgb=[R,G,B]`，新增填色沿用 GUI 核心對既有顏色的保留規則；`other` 不指定顏色。

全批先驗證形狀、原圖邊界及頁面 token，準備成功才更新 GUI。一批對應一筆撤銷；`undo_page_edit` 只撤銷仍與目前內容對應的最近一批 MCP 編輯。保存失敗會回報失敗，尚未保存的修改仍可在 GUI 核對、撤銷或重試保存，不能當成已完成；不要盲目重送同一批。過期 token 必須重新取圖判斷。

每批回傳新的原圖／合成預覽、token、保存狀態及 `changed_bbox`（`[x1,y1,x2,y2]`，右／下邊界不含）。先放大核對修改區域，離頁前再看整頁，才以 `mark_page_reviewed` 保存驗收標記。選區編輯不自動標記已校對，版本與保存檢查也不能代替視覺驗收。`advance=true` 的下一頁仍須重新檢查。

翻頁後回傳下一頁整圖，不沿用上一頁的 `crop`。若本頁標記已保存、下一頁卻無法載入，回應會分開交代 `marked_reviewed`／`advanced` 與 `next_page_unavailable` 或 `next_bundle_error`；此時沒有可編輯的新頁 token，先排除下一頁讀取問題，不把上一頁的影像當成下一頁。

Codex 也可用 CLI 加入服務（將兩個路徑換為實際位置）：

```bash
codex mcp add solid_inpaint -- /absolute/repo/.venv/bin/python /absolute/repo/solid_inpaint_mcp.py
```

安裝後在客戶端重新載入 MCP 服務；當前聊天尚未取得新工具時，不代表程式端連線失敗。配置方式參考 [OpenAI 官方 MCP 說明](https://developers.openai.com/codex/mcp)，SDK 來源為 [MCP 官方 Python SDK](https://github.com/modelcontextprotocol/python-sdk)。
