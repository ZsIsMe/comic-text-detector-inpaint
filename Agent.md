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

命令行批量處理：

```bash
.venv/bin/python detect_solid_inpaint_folder.py /path/to/image_folder
```

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

主要輸出：

```text
mask/<name>.png
  文字偵測 mask。

inpainted/<name>.png
  與原圖同尺寸的透明 BGRA overlay。這不是完整修好的成品圖。

other_mask/<name>.png
  不適合自動純色塗白的區域。

solid_inpaint_report.json
  每頁統計與 debug 資訊。

preview_report.pdf
  檢查用 PDF。
```

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
圖片列表
mask / 原圖疊加預覽
inpainted 合成預覽
other_mask 顯示
矩形工具
筆刷工具
撤銷 / 重做
自動保存 mask
自動重新生成當前頁 overlay
生成 PDF 預覽
打開輸出資料夾
```

手動修改 mask 後，會基於新 mask 重新生成當前頁的 overlay 和 other_mask。

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
mask/
inpainted/
other_mask/
solid_inpaint_report.json
preview_report.pdf
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
