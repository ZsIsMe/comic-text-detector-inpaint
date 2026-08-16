# RF-DETR 支援工作交接文檔

最後更新：2026-08-16（模型下載、依賴、最小推理、adapter 已全部完成）

## 目前 Git 狀態

- 分支：`codex/koharu-rfdetr-seg`
- 已提交：`08a1fad Persist other mask preview appearance settings`
- 未提交的新增/修改（尚未 commit，依指示先不入 Git）：
  - 新增：`load_model.py`、`min_infer_rfdetr.py`、`rfdetr_detector.py`、`HANDOFF_RFDETR.md`
  - 修改：`detect_solid_inpaint_folder.py`、`solid_inpaint_ui.py`
- `models/koharu-layout-rfdetr-seg-2xl-1152/model.safetensors` 已下載完成
  （161,292,684 bytes，SHA-256 已驗證）；`models/` 在 .gitignore 內，不會誤入 Git。

## 目標模型

Hugging Face repository：

```text
mayocream/koharu-layout-rfdetr-seg-2xl-1152
```

權重地址：

```text
https://huggingface.co/mayocream/koharu-layout-rfdetr-seg-2xl-1152/resolve/main/model.safetensors
```

模型資料：

- 架構：`RFDETRSeg2XLarge`
- RF-DETR 版本：`1.7.0`
- 輸入解析度：`1152 x 1152`
- 類別：`text`、`onomatopoeia`、`bubble`、`panel`
- 權重檔：`model.safetensors`
- 遠端大小：約 161,292,684 bytes（約 153 MiB）
- SHA-256：
  `9bf6d2cbd7793c956d8c857bb1672a396eb7f100eb0682f86830d05e31168efb`

模型官方 loader：

```python
from rfdetr import RFDETRSeg2XLarge
from safetensors.torch import load_file

model = RFDETRSeg2XLarge(
    pretrain_weights=None,
    resolution=1152,
    num_select=160,
    num_classes=4,
)
model.model.model.load_state_dict(load_file("model.safetensors", device="cpu"), strict=True)
```

官方推理方式使用：

```python
detections = model.predict(
    image,
    threshold=0.20,
    shape=(1152, 1152),
    include_source_image=False,
)
```

建議類別 threshold：text 0.25、onomatopoeia 0.20、bubble 0.50、panel 0.50。

## 本機環境檢查

- macOS Darwin arm64
- Apple M1 Pro，Metal 支援
- 專案虛擬環境：`.venv`
- Python：3.14.6
- PyTorch：2.13.0
- torchvision：0.28.0
- `torch.backends.mps.is_available()`：`True`
- 可用磁碟：約 160 GiB
- 測試圖片：`/Users/zhongsheng/Downloads/87/14.jpg`
- 測試圖片尺寸：1121 x 1600，RGB JPEG

目前專案已有：NumPy、OpenCV、Pillow、ONNX Runtime、PyTorch、torchvision。

## 依賴安裝結果（已完成）

已安裝並驗證可正常 import 與推理：

- `rfdetr==1.7.0` wheel 已下載至 `/private/tmp/rfdetr-install/`
- `huggingface_hub==1.27.0`
- `transformers==5.15.0`
- `safetensors==0.8.0`
- `supervision==0.30.0`
- `pydantic==2.13.4`
- `requests==2.34.2`
- `pydeprecate==0.11.0`（重點：rfdetr 宣告要 `<0.8`，但實測 0.11.0 完全相容；supervision 則需要 `>=0.9` 的 `TargetMode`，因此保留 0.11.0 才能讓 `predict()` 正常運作）
- `scipy`、`regex`、`urllib3`、`PyYAML`、`tokenizers==0.22.2`、`typer`、`certifi`、
  `charset-normalizer`、`idna`、`click`、`hf-xet`、`httpx`、`pydantic-core==2.46.4`、
  `annotated-types`、`typing-inspection`、`av`、`defusedxml`、`matplotlib`
- `tokenizers` 不可用 0.23.1（transformers 5.15 要求 `<=0.23.0`）；`pydantic-core` 必須為 2.46.4（pydantic 2.13.4 綁定）

網路：pip 的 TLS 問題可透過本機代理解決（`http://127.0.0.1:7890`）。
模型下載使用 `curl -C -` Range 續傳，成功後 SHA-256 與預期一致。

## 最小推理結果（已完成）

`min_infer_rfdetr.py` 對 `/Users/zhongsheng/Downloads/87/14.jpg`（1121×1600）實測：

- CPU：3.76s；MPS：3.55s；兩者偵測結果一致
- 逐類別（建議 threshold）：text 13 個（0.25）、onomatopoeia 4 個（0.20）、
  bubble 9 個（0.50）、panel 7 個（0.50）
- CPU 與 MPS 的 mask 完全一致（IoU = 1.0）

## 已完成的程式碼工作

- `load_model.py`：權重 SHA-256 驗證 + `RFDETRSeg2XLarge` 官方 loader 封裝
- `rfdetr_detector.py`：`RfDetrSegDetector` adapter，介面與 `ComicTextAndBubbleDetector`
  相同 `(raw_mask, refined_mask, blocks)`；text + onomatopoeia 聯集輸出 0/255 mask；
  支援 `device='auto'|'cpu'|'mps'`、`inpaint_mask_dilate`、`mask_unification_method`、
  `mask_mode`（`text_onomatopoeia`/`text`/`onomatopoeia`/`all` 塗白範圍）、
  `class_thresholds`（逐類別 threshold 覆寫）
- `detect_solid_inpaint_folder.py`：新增 `DETECTOR_RFDETR`、模型路徑、`create_detector`
  分支（延遲 import rfdetr，未安裝時給出明確錯誤）、`build_report(..., device=...)`、
  CLI `--detector rfdetr`
- `solid_inpaint_ui.py`：偵測模型選擇對話框新增「RF-DETR」選項與設定面板
  （運算裝置、Mask 膨脹尺寸、塗白範圍），設定持久化於 `detector/rfdetr/*`；批次 worker
  把實際裝置寫入 report；另新增「添加偵測」按鈕與 `AddDetectionDialog`（選 detector +
  加入層 自動/強制純色/需要修改 + 範圍 當前頁/全部頁面），與 `FolderWorker` 的
  `add_detection` 模式；detector 選擇面板重構為共用的 `DetectorSelectorWidget`
- `detect_solid_inpaint_folder.py`：新增 `add_detection_to_mask()`——偵測後與指定層
  （`mask` / `manual_solid` / `manual_other`）做 union 合併，不覆蓋既有內容，再重新生成

## 建議下一步

1. 在 UI 或 CLI 實際跑一個資料夾批次，確認 mask/other_mask/inpainted/report 流程無誤。
2. 可評估 `model.optimize_for_inference()` 對 MPS 延遲的影響（目前單頁約 3.6s）。
3. 確認無誤後再決定是否 commit 上述程式碼（模型權重與 wheel 暫存一律不入 Git）。
