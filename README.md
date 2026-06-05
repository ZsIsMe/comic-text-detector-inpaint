# Solid Inpaint

Solid Inpaint is a Python tool for comic text mask detection, solid-background whiteout overlay generation, and `other_mask` review.

![Solid Inpaint preview](docs/preview.png)

This repository is designed to run from source on macOS and Windows. It does not package a native app.

The model file is not included in this repository. On first launch, `bootstrap.py` downloads `comictextdetector.pt` from the original public release and stores it at:

```text
models/comictextdetector.pt
```

Model source:

```text
https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt
```

The model license and ownership belong to the original project.

## 功能

```text
1. 偵測圖片中的文字，輸出文字 mask
2. 判斷文字周圍背景是否為可靠純色
3. 對可靠區域生成透明塗白 overlay
4. 對不可靠區域生成 other_mask
5. 可手動生成完整 PDF 預覽報告
6. 可選用 Photoshop JSX 生成 PSD
```

## Quick Start

Install Python 3.10-3.12 first. Python 3.13+ may work, but is not the recommended runtime for first-time users.

macOS:

```text
Double-click launch.command
```

If macOS says the command file is not executable, run once:

```bash
chmod +x launch.command
```

Windows:

```text
Double-click launch.bat
```

First launch will:

```text
1. create .venv
2. install requirements.txt
3. download models/comictextdetector.pt if missing
4. start the UI
```

The GUI depends on `PySide6-Essentials` instead of the full `PySide6` package, so first launch avoids downloading the large optional Qt Addons package.

This tool runs on CPU by default and does not provide a CUDA/GPU option.

## Manual Launch

macOS:

```bash
python3 bootstrap.py
```

Windows:

```bat
py -3 bootstrap.py
```

If you prefer to manage dependencies yourself:

macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python solid_inpaint_ui.py
```

Windows:

```bat
py -3 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python solid_inpaint_ui.py
```

## 命令行批量處理

命令行批量處理：

```bash
.venv/bin/python detect_solid_inpaint_folder.py /path/to/image_folder
```

Windows:

```bat
.venv\Scripts\python detect_solid_inpaint_folder.py D:\path\to\image_folder
```

## 圖形界面功能

```text
選擇圖片文件夾
打開最近列表
偵測並生成
顯示進度
瀏覽圖片列表
Mask / 原圖白色疊加預覽與手動編輯
矩形工具：左鍵添加 mask，右鍵去掉 mask
筆刷工具：左鍵添加 mask，右鍵去掉 mask
撤銷 / 重做
編輯後自動保存 mask，並自動重新生成當前頁預覽
Inpainted 合成預覽，可選淡紅色顯示 other_mask
打開輸出資料夾
生成 PDF 預覽
打開 PDF 預覽
說明頁，包含版本號和快捷鍵
```

紅色的「偵測並生成」會重新跑 detector，覆蓋已有 mask。如果輸出資料夾內已有 mask，UI 會要求確認。

圖形界面默認不生成 PDF。需要檢查整套圖片時，點擊「生成 PDF」手動生成。

快捷鍵：

```text
B：筆刷
R：矩形
[：縮小筆刷
]：放大筆刷
← / PageUp：上一頁
→ / PageDown：下一頁
Ctrl+Z：撤銷
Ctrl+Shift+Z：重做
```

模型固定讀取：

```text
models/comictextdetector.pt
```

如果缺少模型，命令行和圖形界面都會提示找不到模型檔。

## Python 輸出

輸入資料夾：

```text
/path/to/image_folder
```

輸出資料夾：

```text
/path/to/image_folder/ctd_inpainted
```

輸出檔案：

```text
ctd_inpainted/mask/<name>.png
ctd_inpainted/other_mask/<name>.png
ctd_inpainted/inpainted/<name>.png
ctd_inpainted/solid_inpaint_report.json
ctd_inpainted/preview_report.pdf  # 圖形界面需手動生成
```

說明：

```text
mask
  偵測後的文字 mask。

inpainted
  與原圖同尺寸的透明 BGRA overlay。
  只包含自動判斷為可塗白的區域。

other_mask
  背景不可靠、非純色、或取樣不足的區域。
  這些區域需要人工檢查或交給其他修補流程。

solid_inpaint_report.json
  每頁統計和 debug 資訊。

preview_report.pdf
  檢查用 PDF。每頁包含 original / preview / mask / other_mask。
  命令行批量處理會自動生成；圖形界面需點擊「生成 PDF」。
```

## 純色判斷

每個文字區塊會建立兩個區域：

```text
repair area
  需要被 overlay 覆蓋的文字修補區。

sample ring
  repair area 外側的背景取樣環。
```

腳本會分析 sample ring 的 RGB histogram。

目前有兩類可通過條件：

```text
strict solid
  顏色分布足夠集中，且主色比例足夠高。

white dominant
  主色接近白色，且白色主峰比例足夠高。
  用於處理白底氣泡、旁白框附近混入少量黑邊的情況。
```

如果完整 sample ring 不可靠，腳本會嘗試上、下、左、右方向取樣，選擇品質最高的方向作為 fallback。

常用參數在 [detect_solid_inpaint_folder.py](detect_solid_inpaint_folder.py) 前段：

```text
REPAIR_EXPAND_PX
SAMPLE_RING_PX
GROUP_MERGE_PX
SOLID_P90_P10_MAX
SOLID_PEAK_RATIO_MIN
WHITE_DOMINANT_MIN
WHITE_PEAK_RATIO_MIN
```

## Photoshop PSD 配套

Python 輸出完成後，可在 Photoshop 中執行：

```text
create_psds_from_outputs.jsx
```

它會生成：

```text
ctd_inpainted/psd/<name>.psd
```

每個 PSD 包含兩個圖層：

```text
bg
overlay-manual
```

以及兩個 alpha channels：

```text
TEXT_CHANNEL
OTHER_CHANNEL
```

PSD 腳本可選擇「有 `OTHER_CHANNEL` 時執行 Photoshop Action」。

執行順序：

```text
打開原圖
-> 建 bg
-> 貼入 overlay-manual
-> 建 TEXT_CHANNEL
-> 建 OTHER_CHANNEL
-> 可選執行 Photoshop Action
-> 保存 PSD
-> 關閉
```

## Repository Contents

此工具的 detector 相關程式放在：

```text
vendor/
```

Keep these files in the repository:

```text
README.md
requirements.txt
bootstrap.py
launch.command
launch.bat
detect_solid_inpaint_folder.py
solid_inpaint_ui.py
create_psds_from_outputs.jsx
models/.gitkeep
icons/
vendor/
```

The model is downloaded on first launch and is intentionally ignored by git:

```text
models/comictextdetector.pt
```

Do not commit:

```text
__pycache__/
.DS_Store
ctd_inpainted/
models/comictextdetector.pt
.venv/
```

## 開發注意

- `vendor/` 是 detector 程式的拷貝版本，不會自動跟外部程式同步。
- 發佈給普通用戶時建議使用 Python 3.10-3.12 測試；`requirements.txt` 會鎖住 `numpy<2`，避免舊 detector 程式遇到 NumPy 2.x 移除舊別名的兼容問題。
- `inpainted` 是完整畫布尺寸的透明 PNG，不需要 Photoshop 圖層用的四角 anchor pixel。
- `other_mask` 只代表不能自動純色填補的 repair area。
- 每次調整純色判斷參數後，建議手動生成並查看 `preview_report.pdf`。
