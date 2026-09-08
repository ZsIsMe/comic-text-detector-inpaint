# Solid Inpaint

日文漫畫批量去字 PSD 生成工具。

網站：[https://zsisme.github.io/comic-text-detector-inpaint/](https://zsisme.github.io/comic-text-detector-inpaint/)

Solid Inpaint 會先偵測漫畫圖片中的文字，對純色背景文字生成透明去字 overlay；對非純色背景、框外字、網點、線稿或複雜背景文字，保存為 `OTHER_CHANNEL`，方便在 Photoshop 中用動作批量執行「生成式移去」。

![Solid Inpaint preview](docs/preview.png)

本項目從源碼運行，支援 macOS 和 Windows，暫不打包原生 App。

## 主要用途

```text
1. 批量偵測漫畫文字 mask
2. 自動處理可靠純色背景文字
3. 生成可疊加的透明去字 overlay
4. 標記非純色背景文字為 other_mask
5. 用項目內 Photoshop JSX 生成 PSD
6. 在 PSD 中保存 TEXT_CHANNEL 和 OTHER_CHANNEL
7. 可綁定 Photoshop 動作，對 OTHER_CHANNEL 批量執行生成式移去
```

一句話：

```text
不止是生成框外去字圖，而是轉換為可繼續精修的 Photoshop PSD。
```

## 快速開始

建議使用 Python 3.10-3.12。Python 3.13+ 可能可用，但不建議普通用戶首次安裝時使用。

macOS：

```text
雙擊 launch.command
```

如果 macOS 提示命令文件不可執行，先執行一次：

```bash
chmod +x launch.command
```

Windows：

```text
雙擊 launch.bat
```

首次啟動會自動：

```text
1. 建立 .venv
2. 安裝 requirements.txt
3. 下載 CTBD、CTD 與 YSGYOLO 2.0 模型
4. 啟動圖形界面
```

CTBD 與 CTD 使用 CPU；YSGYOLO 可選自動、MPS、CPU 或 CUDA，RF-DETR 可選 MPS 或 CPU。

## 手動啟動

macOS：

```bash
python3 bootstrap.py
```

Windows：

```bat
py -3 bootstrap.py
```

如果想手動管理依賴：

macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python solid_inpaint_ui.py
```

Windows：

```bat
py -3 -m venv .venv
.venv\Scripts\python -m pip install -U pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python solid_inpaint_ui.py
```

`requirements.txt` 使用 `PySide6-Essentials`，避免安裝完整 `PySide6` 時下載大型 Qt Addons。

## 圖形界面功能

```text
選擇圖片資料夾
打開最近列表
偵測並生成
使用傳入 Mask 運行
顯示進度
瀏覽圖片列表
Mask / 原圖疊加預覽
手動編輯 mask
矩形、筆刷、魔法棒工具：配合「添加 / 減去」編輯目前 mask
套索工具：左鍵逐點建立多邊形，雙擊或 Enter 閉合並套用
套索支援添加、減去、局部交集、從其他轉入和選區 CTD 檢測
局部視窗：矩形框選 ROI，在獨立放大視窗編輯副本；選擇頂部「調整邊框」工具後，可拖動上下左右四條邊界線，也可輸入 X、Y、寬、高；筆刷、矩形、套索和魔法棒的計算與 mask 修改都強制限制在邊框內，套用後才回寫主頁
右鍵拖曳矩形：清除範圍內所有 mask 層
撤銷 / 重做
編輯後自動保存 mask
自動重新生成當前頁預覽
Inpainted 合成預覽
可顯示 other_mask
導出右圖
導出待精修配對圖片
打開輸出資料夾
生成 PDF 預覽
打開 PDF 預覽
工作流比較與 Mask 分區合成
```

紅色的「偵測並生成」會重新跑 detector，並覆蓋已有的 `mask`、`other_mask` 和 `inpainted` 輸出。如果輸出資料夾內已有 mask，UI 會要求確認。

「使用傳入 Mask 運行」提供兩種方式。「取代目前 Mask」會讓你選擇傳入 Mask 文件夾；若裡面存在同名 PNG，會覆蓋 `ctd_inpainted/raw/mask/<name>.png`，再重新運行。缺少同名 PNG 的頁面會保留原 mask。

「取兩者交集」會讓你選擇傳入 Mask 文件夾。若裡面存在同名 PNG，會用 `目前 mask ∩ 傳入 mask` 覆蓋目前 mask，然後重新生成 `other_mask`、`inpainted` 和 `solid_inpaint_report.json`；缺少同名 PNG 的頁面會保留原 mask。

「導出待精修」會將全部頁面直接輸出到 `ctd_inpainted/export_pair/`，不生成壓縮包。該目錄根層是原圖與 `inpainted` 的去字合成 PNG，`other_mask/` 放同名 mask，`colored/` 放與「導出右圖」一致的標記預覽。

「工作流比較」會開啟獨立視窗。選擇一個 `export_pair` 文件夾後，工具會讀取根目錄底圖，並自動識別 `inpaint_workflows/` 內任意數量的工作流子文件夾；它會逐組比較工作流圖片與底圖來生成各自的差異 Mask，不再依賴 `other_mask/`。頂部可調整「差異閾值」、「最小區域」及「Mask 擴大」（預設 `5 px`），差異 Mask 會緩存在 `.workflow_compare/diff_masks/`，也可按「重算 Mask」刷新當頁。比較區最左側固定顯示當頁實際合成效果，右側可同步比較最多三組結果；所有圖片同步縮放和移動，每張工作流圖片頂部的原圖比較滑桿也會同步移動，初始值為 `0`，即完整顯示工作流結果。每個工作流名稱旁的色塊代表該工作流的 Mask 顏色；在對應面板中，已採用區域使用較高不透明度，未採用區域以相同顏色淡化顯示。首次載入的頁面預設採用第一組工作流；在工作流圖片上左鍵拖矩形或使用筆刷，可重新指定該工作流差異 Mask 內的局部來源，右鍵拖矩形則直接指定該範圍保留原圖，中鍵拖動用於平移。圖片下方按鈕可將該工作流的整個差異 Mask 指定給該組。`M` 顯示或隱藏選區，`[`、`]` 調整筆刷大小。選擇狀態保存在 `.workflow_compare/`，最終圖片和來源摘要 `selection.json` 輸出到 `result/`。

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

## 命令行批量處理

macOS：

```bash
.venv/bin/python detect_solid_inpaint_folder.py /path/to/image_folder
```

Windows：

```bat
.venv\Scripts\python detect_solid_inpaint_folder.py D:\path\to\image_folder
```

命令行模式會處理整個圖片資料夾，並生成 PDF 預覽報告。

## 輸出結構

輸入資料夾：

```text
/path/to/image_folder
```

輸出資料夾：

```text
/path/to/image_folder/ctd_inpainted
```

主要輸出：

```text
ctd_inpainted/raw/mask/<name>.png
ctd_inpainted/raw/other_mask/<name>.png
ctd_inpainted/raw/inpainted/<name>.png
ctd_inpainted/raw/solid_inpaint_report.json
ctd_inpainted/raw/preview_report.pdf
ctd_inpainted/export_pair/<name>.png
ctd_inpainted/export_pair/other_mask/<name>.png
ctd_inpainted/export_pair/colored/<name>.png
ctd_inpainted/export_pair/inpaint_workflows/<workflow>/<name>.png
ctd_inpainted/export_pair/result/<name>.png
```

說明：

```text
mask
  偵測後的文字 mask。

inpainted
  與原圖同尺寸的透明 BGRA overlay。
  只包含自動判斷為可純色覆蓋的區域。

other_mask
  非純色背景、框外字、取樣不足或不適合自動覆蓋的區域。
  這些區域可在 Photoshop 中進一步生成式消除。

solid_inpaint_report.json
  每頁統計和 debug 資訊。

preview_report.pdf
  檢查用 PDF。每頁包含 original / preview / mask / other_mask。
```

## Photoshop PSD 配套

Python 輸出完成後，可在 Photoshop 中執行：

```text
create_psds_from_outputs.jsx
```

Photoshop 路徑：

```text
File > Scripts > Browse...
```

腳本會讀取：

```text
<image folder>/ctd_inpainted/raw/mask/<name>.png
<image folder>/ctd_inpainted/raw/other_mask/<name>.png
<image folder>/ctd_inpainted/raw/inpainted/<name>.png
```

並生成：

```text
<image folder>/ctd_inpainted/raw/psd/<name>.psd
```

每個 PSD 包含：

```text
圖層：
bg
overlay-manual

通道：
TEXT_CHANNEL
OTHER_CHANNEL
```

`overlay-manual` 是已自動去字的透明覆蓋圖層。

`OTHER_CHANNEL` 保存識別到的非純色背景文字，可用 Photoshop 動作轉成選區並批量執行「生成式移去」。

腳本窗口中可選：

```text
有 OTHER_CHANNEL 時執行動作
```

勾選後，選擇已錄好的 Photoshop 動作組和動作。腳本會在有 `OTHER_CHANNEL` 的 PSD 上自動執行該動作。

## 模型文件

模型不包含在 git 倉庫中。首次啟動時 `bootstrap.py` 會下載：

```text
models/comic-text-and-bubble-detector.onnx
models/comictextdetector.pt
models/ysgyolo_yolo26_2.0.pt
```

CTBD 模型來源：

```text
https://huggingface.co/ogkalu/comic-text-and-bubble-detector
```

CTD 模型來源：

```text
https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.2.1/comictextdetector.pt
```

模型授權與歸屬屬於原項目。

點擊「偵測並生成」後可以選擇：

```text
1. YSGYOLO 2.0
2. CTBD（文字＋氣泡偵測）
3. RF-DETR
4. CTD
```

選擇 CTBD 時，可以在執行前設定 Mask 膨脹尺寸、Mask 合併方式和文字區域篩選。
選擇與 CTBD 設定會自動保存，下一次打開對話框時沿用。

YSGYOLO 2.0 也支援「添加偵測」。它沿用 BallonsTranslator 的推理設定（信心閾值 0.3、IoU 0.5、偵測尺寸 1024），由偵測框生成 Mask，再交給現有純色塗白流程。
模型與文本行合併程式已移入本專案，執行時不依賴 BallonsTranslator 的安裝目錄。啟動腳本會檢查模型 SHA-256，缺少時從 `dreMaz/mit_models` 下載。

YSGYOLO 設定會自動保存：

- 運算裝置預設自動，優先 GPU（本機 Apple Silicon 使用 MPS），無可用 GPU 時使用 CPU。
- 合併文本行預設開啟。它只整理文字區塊資料，不會填滿行間空白，不影響塗白 Mask。
- 豎排文本預設關閉（自動判斷方向）；開啟後以豎排方向整理區塊與行順序，不會篩掉橫排文字。
- Mask 擴張尺寸預設 0。數值為擴張半徑，使用與 BallonsTranslator 相同的橢圓核心。
- Mask 圓角半徑預設 0（關閉），可設為 1–1000 px；12 px 為 BallonsTranslator 圓角工具的預設值。先擴張，再按每個連通區的外接矩形削掉四角，只縮減遮罩。半徑會限制在區域寬、高的一半以內；已相連的偵測框視為同一區域。「添加偵測」只對新偵測的 Mask 做圓角，再加入所選層，保留既有遮罩。
- 標籤預設勾選前五項，不勾選 `other`。全部取消時產生空白 Mask；添加偵測時不會加入新區域。

| 標籤 | 說明 |
| --- | --- |
| `balloon` | 氣泡外的文字 |
| `qipao` | 氣泡內的文字 |
| `shuqing` | 豎斜：豎著的氣泡內和氣泡外的傾斜文字 |
| `changfangtiao` | 長方條：全部橫向文字，不區分氣泡或矩形框內外 |
| `hengxie` | 橫斜：長方條的上位版，所有橫著的傾斜文字 |
| `other` | 框體：氣泡，以及任意包含文字的垂直、水平框體 |

這份 `ysgyolo_yolo26_2.0.pt` 的實際類別為 `balloon/qipao/fangkuai/changfangtiao/kuangwai/other`。
依使用者選擇，介面保留 BallonsTranslator 的六個選項：`shuqing`、`hengxie` 標明「此模型無效」；不將 `fangkuai`、`kuangwai` 擅自對應到它們，這兩個原始類別會被略過。
滑鼠停在各標籤上可查看完整說明。

命令行使用預設 YSGYOLO 設定：

```bash
.venv/bin/python detect_solid_inpaint_folder.py /path/to/image_folder --detector ysgyolo
```

新增依賴為 `ultralytics>=8.4.14` 與 `networkx`。Python 3.13 以上使用 NumPy 2，以兼容新 Python 的套件；較舊 Python 沿用 NumPy 1 的限制。

如果缺少所選模型，命令行和圖形界面都會提示找不到模型文件。

## GitHub Pages

本倉庫的介紹頁放在：

```text
docs/index.html
```

啟用 GitHub Pages：

```text
1. 打開 GitHub 倉庫頁面
2. 進入 Settings
3. 左側選 Pages
4. Source 選 Deploy from a branch
5. Branch 選 main
6. Folder 選 /docs
7. Save
```

啟用後網址通常是：

```text
https://zsisme.github.io/comic-text-detector-inpaint/
```

如果 GitHub 顯示的 Pages 地址不同，以 GitHub Settings > Pages 中顯示的地址為準。

## 倉庫內容

需要保留在倉庫中的主要文件：

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
ctbd_detector.py
docs/
icons/
vendor/
```

不要提交：

```text
.venv/
__pycache__/
.DS_Store
ctd_inpainted/
models/comictextdetector.pt
models/comic-text-and-bubble-detector.onnx
```

## 開發注意

- `vendor/` 是 detector 程式的拷貝版本，不會自動跟外部程式同步。
- 建議用 Python 3.10-3.12 測試發佈流程。
- `requirements.txt` 鎖定 `numpy<2`，避免舊 detector 程式遇到 NumPy 2.x 移除舊別名的兼容問題。
- `inpainted` 是完整畫布尺寸的透明 PNG，不需要 Photoshop 圖層用的四角 anchor pixel。
- `other_mask` 表示不能自動純色填補、需要後續處理的 repair area。
- 每次調整純色判斷參數後，建議手動生成並查看 `preview_report.pdf`。
