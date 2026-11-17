# 校對會話 CLI

從專案根目錄執行下列腳本。需要 MCP 時先確認 GUI 已啟動，並用 `instances` 讀取實際 ID、PID 和資料夾。`review_session.py` 由自身位置找到 repo 的 MCP server，不依賴臨時目錄裡的舊 client。以下命令中的 `python` 使用本專案 `.venv/bin/python`。

`batches` 預設每次取 20 頁，剩餘少於 40 頁時把剩餘頁一次交給最後一批；39 頁直接一批，59 頁為 20、39，40 頁仍為 20、20。自訂 `--size N` 時，尾批門檻同步為少於 `2*N` 頁。`--indices` 可輸入實際待校對頁碼，順序應與 GUI 列表一致。

```bash
python .agents/skills/comic-detect-review/scripts/review_session.py batches --pages 201
python .agents/skills/comic-detect-review/scripts/review_session.py batches --indices 1,2,5,8
python .agents/skills/comic-detect-review/scripts/review_session.py instances
python .agents/skills/comic-detect-review/scripts/review_session.py bind --session chapter_a --instance-id <32位ID> --pid <GUI_PID> --folder /absolute/image/folder
python .agents/skills/comic-detect-review/scripts/review_session.py bind --session chapter_b --instance-id <另一個ID> --pid <另一個PID> --folder /absolute/other/folder --open-project
python .agents/skills/comic-detect-review/scripts/review_session.py call --binding <bind輸出的binding路徑> list_pages
python .agents/skills/comic-detect-review/scripts/review_session.py call --binding <binding路徑> get_review_bundle /tmp/bundle-params.json
python .agents/skills/comic-detect-review/scripts/review_session.py handoff --binding <binding路徑> --start-page 1 --end-page 20
python .agents/skills/comic-detect-review/scripts/review_session.py switch --binding <binding路徑> --folder /absolute/other/folder
python .agents/skills/comic-detect-review/scripts/review_session.py release --binding <binding路徑>
```

參數 JSON 檔應為單一物件，例如 `{"max_side": 1600, "crop": [10, 20, 100, 80]}`。與舊 MCP client 一樣，位置參數是工具名和可選 JSON 檔。`bind` 會在固定用戶級目錄認領 GUI；不同會話或 binding 路徑認領同一視窗會被拒絕，原會話重複 `bind` 可沿用認領。新 GUI 尚未開資料夾時，明確加 `--open-project`，先認領再載入目標資料夾；不加時須與 GUI 目前資料夾一致。存量舊 binding 沒有認領憑證，須明確重新執行 `bind`。`instance_id` 只能來自 binding，不可寫在參數檔；已綁定後的 `open_project` 只能透過 `switch`。`call`／`switch` 持同 GUI 的短事務鎖，每次普通呼叫前後都核對 bound ID、PID、資料夾；不符即失敗，不保存新的圖片或 token。不同 GUI 可同時呼叫。編輯、撤銷與校對標記須使用本會話最新 bundle 的 token。`switch` 先核對舊資料夾，顯式開啟新資料夾後更新 binding；若開啟回應逾時，重新查詢同 ID／PID，GUI 已到目標資料夾時修復 binding。GUI 重開後重新 `bind` 到新 ID，舊 GUI 確認消失後可用 `reap --instance-id <舊ID>` 清理舊認領；認領已寫但 binding 遺失或憑證不符的孤兒記錄也可清理，有效 binding 加活視窗不可強行認領。完成時 `release` 使舊 binding 失效。

`handoff` 在同一事務內執行 `save_project`、`list_pages`，核對指定閉區間各頁的 `reviewed=true` 和保存狀態，輸出 `verified`、`unreviewed_in_range`、`next_batch` 與 `next_unreviewed_index`，不取圖、不自動標記。未指定範圍時核對整個專案；未通過時 CLI 退出碼 1。

輸出 JSON 含 `metadata`、`images` 絕對路徑及 `output_dir`。每次取圖建立獨立快照目錄，檔名維持 `original.png`、`composite.png`、`mask.png` 或 `view.png`；同版 `latest-bundle.json` 存 metadata、token 和 `image_paths`。快照寫齊後才原子更新此指標，失敗時舊指標仍對應舊快照。輸出路徑依 session、instance、資料夾分開。每次取新 bundle 後使用最新 token；跨頁、跨 GUI 或跨資料夾不可重用。工具仍由 GUI 核心驗證 token 和保存。

Computer Use 的**實際桌面輸入**前，在可保持執行的 exec TTY 會話執行：

```bash
python .agents/skills/comic-detect-review/scripts/desktop_lock.py
```

讀到 `{"locked": true, ...}` 才能操作實體滑鼠／鍵盤。其他會話同時取得鎖會以退出碼 2 失敗；需要等候可加 `--wait`。完成輸入後向該 TTY 送出換行或 EOF，或結束進程，POSIX `flock` 即釋放。MCP 及無桌面輸入的圖像判讀不需此鎖。預設鎖位於固定的 `/tmp/solid-inpaint-<uid>/coordination/`，不受 `TMPDIR` 影響。

多個資料夾的 GPU CLI 偵測用以下包裝器排隊；它轉發子進程退出碼，Ctrl-C 時轉送中斷，子進程繼承鎖描述符以避免包裝器提前退出後鎖失效。僅包住偵測命令，校對及各 GUI 不共用 GPU 鎖。包裝器的 `--device` 只決定是否排隊，不代替偵測程式自身的裝置參數；明確使用 CPU 時將包裝器設為 `--device cpu`，不排隊。

```bash
python .agents/skills/comic-detect-review/scripts/detect_queue.py --device gpu -- .venv/bin/python detect_solid_inpaint_folder.py /absolute/image/folder
```

上述協調目前只支援 macOS／其他 POSIX 系統。它不能攔截繞過這些腳本的 MCP、Computer Use 或偵測進程；專案正式資料仍沒有跨程序編輯鎖。
