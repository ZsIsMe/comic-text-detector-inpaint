# Agent 工作備忘

這個項目是 Solid Inpaint，一個漫畫文字處理輔助工具。它的定位不是完整翻譯器，也不是通用 inpainting 工具，而是：

```text
偵測漫畫圖片中的文字 mask，
對可靠純色背景生成透明塗白 overlay，
對不可靠背景生成 other_mask，方便人工或其他工具後續處理。
```

核心偵測與編輯流程仍保存 OTHER 分類，沒有自動修補複雜背景。明確導出 PSD 時提供獨立的 LaMa 修補擴展：由 OTHER 擴張並扣除人工保護後生成可隱藏、可繼續精修的圖層，不回寫原圖或正式頁面資料，也不代表修補品質已經人工驗收。

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

psd_export.py / lama_inpainter.py / lama_config.py
  按需導出透明修補 PNG 與分層 PSD。psd_export.py 先完成本批全部 LaMa PNG，再讀取原圖、solid 及修補 PNG，以 psd-tools 組裝 PSD；第二階段不重新推論。lama_inpainter.py 懶載入、校驗並序列化模型推論，lama_config.py 管理專用權重設定。只在明確 PSD 導出時使用，不屬於漫畫偵測或正式頁面保存流程。
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

預設跳過已有成功結果及人工資料，只處理未完成頁；成功但沒有選區的頁面也算已有結果。`--start-page 5 --end-page 8` 指定與 GUI 相同自然排序、從 1 開始的頁碼（包含首尾）；只有使用者明確要求重跑才加 `--redetect`。本分支唯一文字偵測器為 Koharu Layout HighRes，`--detector` 僅接受 `rfdetr`，可省略；另可用 `--device`、`--bubble`／`--no-bubble`。有效的 RF-DETR 與氣泡參數沿用專案設定，新專案使用預設值，不讀寫 GUI 全局偏好。舊 v2 專案若記錄了已移除的模型，仍可讀取其保存結果；新偵測或重新分類改用 HighRes 安全預設，不沿用舊模型專屬參數，也不因此自動重跑既有頁面。結果仍是相同的 v2 專案，之後可用 GUI 繼續逐頁校正。退出碼 0 為成功、1 為頁面失敗、2 為參數或啟動錯誤；完整參數見 README。不要同時用 CLI 和 GUI 修改同一專案。命令列偵測不執行 LaMa 或 PSD 導出。

## 依賴注意

`requirements.txt` 有兩個重要點：

```text
numpy<2（Python 3.13 之前）
numpy>=2,<3（Python 3.13 以上）
PySide6-Essentials
```

NumPy 限制依 Python 版本選擇相容系列，修改時須同時核對 OpenCV、PyTorch 與 RF-DETR 的相容性。

分層 PSD 導出使用 `psd-tools==1.26.1`。LaMa 沿用 PyTorch 推論，模型架構包含頻域運算；不要只因 GPU 可見就假定該後端能正確執行，沿用 adapter 的裝置檢查及回退處理。

`PySide6-Essentials` 是為了避免安裝完整 `PySide6` 時拉入大型 Qt Addons。當前 GUI 主要使用 `QtCore`、`QtGui`、`QtWidgets`，Essentials 應該足夠。

如果後續加入 QtCharts、QtWebEngine、QtMultimedia 等功能，需要重新檢查是否仍能只依賴 `PySide6-Essentials`。

## 模型文件

依使用者要求，模型權重不提交到 Git，也不使用 Git LFS 分發。`models/` 僅保存本機下載檔案，除 `.gitkeep` 外全部忽略；保留來源、下載與 SHA-256 校驗設定於程式碼。首次啟動由 `bootstrap.py` 自動下載缺少的 HighRes 與 LaMa，已有且校驗相符的檔案直接複用。直接執行 CLI 前也須先備妥模型；可執行 bootstrap 或按 README 手動下載。移除 Git 跟蹤時保留本機模型；既有歷史提交中的 LFS 指標不因停止跟蹤而消失。

兩個模型統一使用 Hugging Face 原發布來源，不提供 HF／ModelScope 選擇。首次啟動時 `bootstrap.py` 會下載：

```text
models/koharu-layout-highres/model.safetensors
models/lama_large_512px.ckpt
```

HighRes 模型來源：

```text
https://huggingface.co/mayocream/koharu-layout-highres
```

模型授權與歸屬屬於原項目。`rfdetr_config.py` 統一 HighRes 權重來源、路徑、SHA-256、1920 輸入尺寸與分類門檻；文字及氣泡填充共用此權重。

PSD 匯出專用 LaMa 權重來源為 `https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt`，大小 204,544,673 bytes，SHA-256 為 `11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935`。`lama_config.py` 管理來源與校驗值；bootstrap 下載並校驗上述兩個模型，已存在且校驗相符時不重複下載。LaMa 不是第二個文字偵測器，只在明確 PSD 導出且 OTHER 非空時推論。

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

「導出 PSD 素材」先保存目前編輯，再由背景 worker 做兩階段整批導出。第一階段「LaMa 修補」完成全部 `psd_assets/solid`、`other_mask`、`repair_mask` 素材與 `psd_assets/lama_large_512px/<stem>.png` 透明修補圖；第二階段「PSD 組裝」才從原圖、solid 和落盤修補 PNG 生成 `psd_assets/psds/<stem>.psd`，不重跑模型。視窗顯示進度條、階段、該階段頁碼與檔名；整批以 2N 個頁面工作單位更新進度。

PSD 自下而上為原圖 `bg`、有純色 Alpha 時才建立的 `overlay-manual`、OTHER 非空時才建立並預設顯示的 `lama_large_512px` 普通 RGBA 圖層；修補層內容必須與中間 PNG 同源。空 OTHER 不執行推論、不生成修補 PNG，也不建立 LaMa 圖層；同頁已有尺寸相符且 RGBA 全零的舊修補 PNG 在重新導出時移除，非空或損壞舊檔保留且不採用。LaMa 的 8-bit 修補輸出只在 `repair_mask > 0` 處有內容與 Alpha 255，區外 RGBA 全為 0；寫入 16-bit PSD 時按位深轉存，保持區內完全不透明。不把黑白 OTHER 放成 PSD 圖層或 `OTHER_CHANNEL`，旁邊的黑白 `other_mask` PNG 仍保留。

本分支試用使用者已測 LaMa-Manga-Local-Crop：FP32/max768；OTHER 先十字核 GrowMask8（tapered），25×25 方核膨脹分組（merge_distance24），按原選區∩label取框，面積比1.7外擴。上下文重疊不合併，每組只推論自己的label。局部寬高按scale→round→ceil64後LINEAR調整，square REFLECT補邊，模型模型輸出先在網路尺寸做mask*prediction+(1-mask)*image，再roundclip uint8、去補邊及LINEAR恢復尺寸。不混入舊版16px合併/min32/thin-area縮放。對本工程僅適配：原圖＋solid合成作底圖；Grow後扣除已有solid及edited>0且原OTHER==0的人工擦除保護；原OTHER及raw不改。每頁派生 repair_mask PNG記錄實際範圍（空頁全黑），PNG與PSD Alpha按它驗證，第二階段只讀同一mask與修補PNG。assets-only優先自動讀對應工程raw的edited；無raw或edited_loader時只能保護可知solid，須明確說明人工擦除資訊缺失。前版1536策略保留於codex/koharu-layout-highres的509b2c9。推論方法改動須同步升級lama_config.REPAIR_POLICY，舊策略PSD保留並標stale；批量N頁/K個實際非空修補頁有3N+K份PNG、4N+K個雜湊輸出。

新 PSD 保留原圖尺寸與位深；RGB 來源的既有 ICC 直接寫入，8-bit CMYK 或帶灰階 ICC 的來源先轉為 sRGB 並使用轉換後的 ICC。16-bit 灰階 ICC 或其他不能套用到 RGB PSD 的描述檔明確報錯，不能透過 8-bit 轉換降低精度。16-bit 背景保深，LaMa 輸入及修補內容使用 8-bit，不把整張背景降為 8-bit。每次明確導出依當次資料更新中間 PNG；已存在的同名 PSD 預設跳過，不覆寫 Photoshop 手改。舊 `create_psds_from_outputs.jsx` 可繼續選用，輸出仍為 `ctd_inpainted/psd/` 並沿用 `OTHER_CHANNEL` 流程，與程式直接生成的 `psd_assets/psds/` 分開。

## 核心處理邏輯

核心流程在 `detect_solid_inpaint_folder.py`：

```text
1. 使用 RfDetrSegDetector（Koharu Layout HighRes）偵測文字 mask
2. 將 mask 連通區合併成文字區塊
3. 對每個區塊建立 repair area
4. 在 repair area 外建立 sample ring
5. 分析 sample ring 的顏色分布
6. 若背景足夠純色，生成透明 overlay
7. 若背景不可靠，寫入 other_mask
```

啟用氣泡填充時，使用 Koharu Layout HighRes 的 `bubble` 類（信心門檻 0.50）提供逐實例輪廓，不再次分類背景。一般頁共用同一次推論的獨立氣泡結果；按來源像素及模型簽章核對後才使用，不從文字 Mask 推導氣泡。外部 Mask 需要氣泡填充時會懶載入 HighRes 氣泡模型，自動使用 CUDA、MPS 或 CPU；長圖仍用重疊裁窗並優先沿用已載入的偵測器。關閉氣泡填充不讀氣泡快取或另載入氣泡模型。新版快取與舊模型快取隔離，既有正式頁面不自動重跑。

氣泡填充預設開啟「直接對氣泡塗白」：對每個偵測到的氣泡內縮後塗白，即使沒有文字或只有 OTHER。取消此子選項則沿用已判純色文字的底色擴張：氣泡內須有純色文字才填色，只有 OTHER 不觸發。兩種模式都會清除填色範圍內的自動 OTHER；重新偵測後仍逐像素還原人工分類、填色與擦除。GUI「偵測並生成」的氣泡填充選擇存於全局 QSettings，跨專案及重開沿用；當次實際處理設定也寫入 `raw/project.json` 的 `settings.solid_fill`，供專案處理及 CLI 使用。舊專案缺 `direct_white` 時預設開啟。

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

PSD 修補擴展只處理使用者明確要求的導出，不改 OTHER 分類或自動完成校對。測試時按授權頁數處理，保留既有 PSD 樣本；不得為單頁驗證自行展開整話批量匯出。

修改依賴時要注意普通用戶首次啟動成本。完整 PySide6 會顯著增加下載量和磁碟佔用。

文字偵測器只有 `rfdetr_detector.py`；通用影像 I/O 與 Mask 幾何分別位於 `image_io.py`、`mask_utils.py`。舊 detector 的 vendor 程式已移除；PSD 匯出專用 LaMa 架構在 `vendor/lama_large.py`。氣泡填充與 LaMa 的衍生設計仍須保留來源標註及相關授權，包括 `vendor/LICENSE-BallonsTranslator`。

## 驗證建議

修改後至少做：

```bash
.venv/bin/python -c "from PySide6.QtCore import Qt; from PySide6.QtGui import QImage; from PySide6.QtWidgets import QApplication; print('qt ok')"
```

```bash
.venv/bin/python -c "from rfdetr_detector import RfDetrSegDetector; from detect_solid_inpaint_folder import create_detector; print('detector import ok')"
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

修改 PSD 導出時，以授權的單頁副本驗證 `psd_assets/lama_large_512px/` 及 `psd_assets/psds/`：確認整批 PNG 先完成、PSD 組裝不重新推論，重讀圖層順序、可見性、ICC、尺寸與位深；核對 LaMa 層與中間 PNG 同源、repair_mask 區外 RGBA 全零、區內 Alpha 為該位深最大值（8-bit 為 255），且非空 OTHER 才執行推論。同時核對原圖與 `raw/` 資料未改動，以及 PNG 按當次資料更新、同名既有 PSD 預設跳過。

## 不應提交

不要提交：

```text
.venv/
__pycache__/
.DS_Store
ctd_inpainted/
```
