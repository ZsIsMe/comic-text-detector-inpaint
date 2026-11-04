# Agent 工作備忘

這個項目是 Solid Inpaint，一個漫畫文字處理輔助工具。它的定位不是完整翻譯器，也不是通用 inpainting 工具，而是：

```text
偵測漫畫圖片中的文字 mask，
對可靠純色背景生成透明塗白 overlay，
對不可靠背景生成 other_mask，方便人工或其他工具後續處理。
```

## 項目入口

主要入口：

```text
bootstrap.py
solid_inpaint_ui.py
detect_solid_inpaint_folder.py
```

說明：

```text
bootstrap.py
  首次啟動入口。負責建立 .venv、安裝 requirements.txt、下載模型、啟動 GUI。

solid_inpaint_ui.py
  PySide6 圖形界面。提供資料夾選擇、偵測生成、mask 預覽與手動編輯、PDF 預覽等功能。

solid_inpaint_control.py / solid_inpaint_mcp.py
  GUI 內的本機 Qt socket 控制端與可選的 MCP stdio 橋接。開專案、偵測、導航和保存沿用 GUI；MCP SDK 只在橋接程序需要，安裝與工具契約見 docs/mcp.md。

workflow_compare_ui.py
  獨立的多工作流比較與合成視窗。讀取 export_pair 與 inpaint_workflows，按各工作流結果和底圖的差異生成專屬 Mask，以同步比較滑桿、矩形或筆刷指定來源，最左側顯示當頁成品並輸出 result；不依賴 other_mask。

detect_solid_inpaint_folder.py
  核心批量處理腳本。負責偵測文字、生成 mask、生成 inpainted overlay、生成 other_mask、寫入報告。
```

## 運行方式

推薦使用 Python 3.10-3.12。

macOS：

```bash
python3 bootstrap.py
```

Windows：

```bat
py -3 bootstrap.py
```

無界面批量偵測（不啟動 GUI，不需要 MCP）：

```bash
.venv/bin/python detect_solid_inpaint_folder.py /path/to/image_folder
```

預設跳過已有成功結果及人工資料，只處理未完成頁；成功但沒有選區的頁面也算已有結果。`--start-page 5 --end-page 8` 指定與 GUI 相同自然排序、從 1 開始的頁碼（包含首尾）；只有使用者明確要求重跑才加 `--redetect`。可用 `--detector`、`--device`、`--bubble`／`--no-bubble`；未指定模型與氣泡設定時，舊專案沿用其已保存參數，新專案使用預設值，不讀寫 GUI 全局偏好。結果仍是相同的 v2 專案，之後可用 GUI 繼續逐頁校正。退出碼 0 為成功、1 為頁面失敗、2 為參數或啟動錯誤；完整參數見 README。不要同時用 CLI 和 GUI 修改同一專案。

## 依賴注意

`requirements.txt` 有兩個重要點：

```text
numpy<2
PySide6-Essentials
```

`numpy<2` 是為了避免舊 detector 程式遇到 NumPy 2.x 移除舊別名後啟動失敗。

`PySide6-Essentials` 是為了避免安裝完整 `PySide6` 時拉入大型 Qt Addons。當前 GUI 主要使用 `QtCore`、`QtGui`、`QtWidgets`，Essentials 應該足夠。

如果後續加入 QtCharts、QtWebEngine、QtMultimedia 等功能，需要重新檢查是否仍能只依賴 `PySide6-Essentials`。

## 模型文件

模型不提交到 git。

首次啟動時 `bootstrap.py` 會下載：

```text
models/comictextdetector.pt
```

模型來源：

```text
https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt
```

模型授權與歸屬屬於原項目。

## 輸入與輸出

輸入是一個圖片資料夾，支援：

```text
.bmp
.jpg
.jpeg
.png
```

輸出位置：

```text
<image_folder>/ctd_inpainted/
```

主要資料（v2，不讀取舊格式）：

```text
ctd_inpainted/raw/project.json
ctd_inpainted/raw/pages/<原圖檔名>.npz
ctd_inpainted/raw/cache/<原圖檔名>.npz
```

`project_store.py` 管理唯一正式頁面資料（填色 overlay、OTHER、人工修改保護範圍）與原子保存。編輯只有純色填充和圖像修補兩類；偵測 Mask 是內部快取，普通編輯不重新分類。重新偵測保留人工修改與擦除。

`export_pair`、`psd_assets`、`preview_report.pdf` 按需生成在 `raw` 外。格式和行為見 `docs/two_class_project.md`。

## 核心處理邏輯

核心流程在 `detect_solid_inpaint_folder.py`：

```text
1. 使用 TextDetector 偵測文字 mask
2. 將 mask 連通區合併成文字區塊
3. 對每個區塊建立 repair area
4. 在 repair area 外建立 sample ring
5. 分析 sample ring 的顏色分布
6. 若背景足夠純色，生成透明 overlay
7. 若背景不可靠，寫入 other_mask
```

啟用 MangaLens 時，它只提供氣泡輪廓，不再次分類背景。氣泡填充預設開啟「直接對氣泡塗白」：對每個偵測到的氣泡內縮後塗白，即使沒有文字或只有 OTHER。取消此子選項則沿用已判純色文字的底色擴張：氣泡內須有純色文字才填色，只有 OTHER 不觸發。兩種模式都會清除填色範圍內的自動 OTHER；重新偵測後仍逐像素還原人工分類、填色與擦除。GUI「偵測並生成」的氣泡填充選擇存於全局 QSettings，跨專案及重開沿用；當次實際處理設定也寫入 `raw/project.json` 的 `settings.solid_fill`，供專案處理及 CLI 使用。舊專案缺 `direct_white` 時預設開啟。

「偵測並生成」可選起始頁與結束頁，依左側圖片列表順序、包含首尾，預設第一頁至最後一頁；只重跑所選頁面。
此視窗的「保留原圖（不去字）」模式不載入模型或執行偵測，將所選頁的填色、OTHER、人工修改與擦除全部清空，保存為已處理的空白頁；既有選區會先要求確認，校對標記不會因此成立。原圖與範圍外頁面不變。

判斷純色時主要看：

```text
sample pixels 是否足夠
RGB histogram 的 p90-p10 spread
主色峰值比例
是否接近白色主導背景
```

## GUI 功能

GUI 提供：

```text
選擇圖片資料夾
最近資料夾
偵測並生成
使用傳入 Mask 運行
圖片列表
mask / 原圖疊加預覽
inpainted 合成預覽
other_mask 顯示
矩形工具
筆刷工具
魔法棒工具
多邊形套索工具
局部 ROI 獨立編輯視窗，含獨立的四邊調整工具，所有編輯工具硬限制於 ROI 內
撤銷 / 重做
自動保存 mask
自動重新生成當前頁 overlay
生成 PDF 預覽
打開輸出資料夾
獨立工作流比較視窗：最左側當頁成品、同步比較、Mask 內矩形／筆刷選擇、result 輸出
```

手動編輯只更新正式的兩類選區、顏色與修改記錄，預覽不重新分類。

AI 監修介面獨立顯示目前頁、保存狀態與編輯模式，支援路徑輸入、頁碼／完整檔名跳轉、F3 左鍵恢復原圖及「完成本頁並到下一頁」。校對標記存於專案頁面 metadata，與偵測成功及 OTHER 分開；以正式頁面內容摘要判斷標記是否仍有效。切工具保留相容模式，文字輸入不觸發 A／D 翻頁。

開專案、切頁、開始偵測及同版原圖／合成圖讀取可優先使用已接入的 MCP。全話偵測省略結束頁，由程式取實際圖片數；任務完成與整話覆蓋分開核對。影像判斷仍逐頁進行；視覺確認後，可攜帶頁面 token 透過 MCP 執行原圖座標的矩形／多邊形選區，沿用 GUI 核心、撤銷及正式保存，再看修改後預覽驗收。純色新增明確指定 RGB，複雜背景標記 OTHER。Computer Use 負責實際介面確認及尚未暴露的筆刷、魔法棒、局部副本等操作。未接入時使用原有 GUI。MCP 遇到模態視窗、筆畫或局部副本編輯時回報 busy；過期頁面 token、保存失敗不得假報成功或切頁，不繞過 GUI 直接改寫專案檔。完整操作契約見 docs/mcp.md。

多開 GUI 時，每個視窗有獨立 MCP `instance_id`。先以 `list_app_instances` 依資料夾及目前頁辨認目標，再在該視窗的每個工具呼叫傳入同一 `instance_id`；多視窗省略 ID 會回報 `ambiguous_instance`，單視窗保留省略 ID 的用法。視窗重開後重新取得 ID。多視窗各自處理不同漫畫資料夾，正式專案資料沒有跨程序編輯鎖。

若要用另一個文件夾的 mask 取代目前 mask，可在 GUI 的「使用傳入 Mask 運行」中選「取代目前 Mask」，再選擇傳入 Mask 文件夾。同名 PNG 用於重新分類，保留人工修改與擦除；缺少同名 PNG 的頁面保持原狀。

若需要用另一組 mask 收窄目前 mask，可在「使用傳入 Mask 運行」中選「取兩者交集」，再選擇傳入 Mask 文件夾。它會將同名 PNG 與兩類選區取交集，保存原色及擦除記錄；缺少同名 PNG 的頁面保持原狀。

## 開發原則

保持項目定位清楚：

```text
不要把它改成完整翻譯器。
不要讓它假裝能完美修補複雜背景。
對不可靠背景，優先標記 other_mask，而不是硬修。
```

修改依賴時要注意普通用戶首次啟動成本。完整 PySide6 會顯著增加下載量和磁碟佔用。

修改 detector/vendor 相關代碼時要小心。`vendor/` 是外部 detector 程式的拷貝版本，不會自動跟上游同步。

## 驗證建議

修改後至少做：

```bash
.venv/bin/python -c "from PySide6.QtCore import Qt; from PySide6.QtGui import QImage; from PySide6.QtWidgets import QApplication; print('qt ok')"
```

```bash
.venv/bin/python -c "import sys; sys.path.insert(0, 'vendor'); from detect_solid_inpaint_folder import TextDetector; print('detector import ok')"
```

如果改動核心處理邏輯，應該用一個小圖片資料夾跑：

```bash
.venv/bin/python detect_solid_inpaint_folder.py /path/to/test_images
```

檢查：

```text
raw/project.json
raw/pages/
raw/cache/
export_pair/
preview_report.pdf（按需生成）
```

## 不應提交

不要提交：

```text
.venv/
__pycache__/
.DS_Store
ctd_inpainted/
models/comictextdetector.pt
```
