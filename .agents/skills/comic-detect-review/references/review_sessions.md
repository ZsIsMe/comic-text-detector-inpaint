# 校對會話 CLI

從專案根目錄執行下列腳本。需要 MCP 時先確認 GUI 已啟動，並用 `instances` 讀取實際 ID、PID 和資料夾。`review_session.py` 由自身位置找到 repo 的 MCP server，不依賴臨時目錄裡的舊 client。以下命令中的 `python` 使用本專案 `.venv/bin/python`。

`batches` 每次取 10 頁，剩餘少於 20 頁時把剩餘頁一次交給最後一批；39 頁為 10、10、19，20 頁仍是 10、10。`--indices` 可輸入實際待校對頁碼，順序應與 GUI 列表一致。

```bash
python .agents/skills/comic-detect-review/scripts/review_session.py batches --pages 201
python .agents/skills/comic-detect-review/scripts/review_session.py batches --indices 1,2,5,8
python .agents/skills/comic-detect-review/scripts/review_session.py instances
python .agents/skills/comic-detect-review/scripts/review_session.py bind --session chapter_a --instance-id <32位ID> --pid <GUI_PID> --folder /absolute/image/folder
python .agents/skills/comic-detect-review/scripts/review_session.py call --binding <bind輸出的binding路徑> list_pages
python .agents/skills/comic-detect-review/scripts/review_session.py call --binding <binding路徑> get_review_bundle /tmp/bundle-params.json
python .agents/skills/comic-detect-review/scripts/review_session.py switch --binding <binding路徑> --folder /absolute/other/folder
```

參數 JSON 檔應為單一物件，例如 `{"max_side": 1600, "crop": [10, 20, 100, 80]}`。與舊 MCP client 一樣，位置參數是工具名和可選 JSON 檔。`instance_id` 只能來自 binding，不可寫在參數檔；`open_project` 只能透過 `switch`。每次普通呼叫前後都核對 bound ID、PID、資料夾；不符即失敗，不保存新的圖片或 token。編輯、撤銷與校對標記須使用本會話最新 bundle 的 token。`switch` 先核對舊資料夾，顯式開啟新資料夾後更新 binding。GUI 重開時不要改舊 binding，重新 `bind` 到新的 ID。

輸出 JSON 含 `metadata`、`images` 絕對路徑及 `output_dir`。每次取圖建立獨立快照目錄，檔名維持 `original.png`、`composite.png`、`mask.png` 或 `view.png`；同版 `latest-bundle.json` 存 metadata、token 和 `image_paths`。快照寫齊後才原子更新此指標，失敗時舊指標仍對應舊快照。輸出路徑依 session、instance、資料夾分開。每次取新 bundle 後使用最新 token；跨頁、跨 GUI 或跨資料夾不可重用。工具仍由 GUI 核心驗證 token 和保存。

Computer Use 的**實際桌面輸入**前，在可保持執行的 exec TTY 會話執行：

```bash
python .agents/skills/comic-detect-review/scripts/desktop_lock.py
```

讀到 `{"locked": true, ...}` 才能操作實體滑鼠／鍵盤。其他會話同時取得鎖會以退出碼 2 失敗；需要等候可加 `--wait`。完成輸入後向該 TTY 送出換行或 EOF，或結束進程，POSIX `flock` 即釋放。MCP 及無桌面輸入的圖像判讀不需此鎖。本助手目前只支援 macOS／其他 POSIX 系統；它是合作式操作約定，不會攔截繞過它的 Computer Use 或其他代理。
