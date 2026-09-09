# 氣泡凹口分割與中心預覽（不使用模型）

入口：`preview_split_centers.py`（轉入 `preview_original_centers.py`）。
凹口幾何：`bubble_neck_split.py`。

## 方法

1. 讀取來源 JSON 的 `transMap`、原圖，以及同名 `inpainted/*.png`。
2. 沿用 `layout_core.get_best_component_mask` 取得白色連通區；共用連通區的文字項目歸為一組。
3. 用多尺度腐蝕、保留種子所在區域、有限膨脹來關閉細小漏口。選取漏到整頁時，另外用實際水平格線限制上下範圍。膨脹結果限制在原選區內。
4. 在封閉輪廓找局部凹點，再回到未平滑的輪廓及原圖墨線微調端點。候選切線須位於氣泡內、分開不同種子，且不能穿過種子的近鄰。
5. 在有凹點證據的候選中選擇較短的配對，將切線實際寫入計算遮罩。切線角度不由文字中心的垂直平分線決定。
6. 對每個新區域呼叫原 `layout_core` 的最大內接矩形、外接矩形投影修正、形狀判斷及中心選擇函式。平滑限制於各自分區內，避免跨回切線另一側。

每頁預覽只畫藍色切線、綠色中心和文字項目序號。JSON 保存矩形幾何、切線端點、清理尺度、原未分割候選中心及新候選中心；這是診斷輸出，不是正式排版 JSON。中心未通過「在自身分區內」检查時不顯示綠點；未找到有效凹點對時不偽造切線。

## 重跑

```bash
.venv/bin/python preview_split_centers.py \
  '/Users/zhongsheng/Documents/comic_data/居酒屋/居酒屋12_13/居酒屋12_13_meo_bt.json' \
  --module-dir '/Users/zhongsheng/Documents/comic_data/comic_translator_playwright/labelplus_bt_meo_convert/建立对齐方框' \
  --output-dir artifacts/original_neck_all_pages
```

加入 `--pages 12-19 12-20` 可只測重點頁。`--center-mode` 支援原本的 `auto/outer/inner/average`。省略 `--pages` 跑全部頁面。

## 驗證與限制

- 每組檢查分區互斥、遮罩不超出原選區、每個種子仍屬於對應分區。
- 用報告內的切線端點重建分割，逐像素比較計算區域，避免預覽與算法使用不同的切線。
- `12-19` 的 1/2、4/5，以及 `12-20` 的右上三泡、左上兩泡皆有凹口切線。
- 42 頁測試產生 51 條切線、275 個文字項目中有 265 個區域內的新候選中心。其餘 10 個保留原因；包含無氣泡文字、同一平滑氣泡內的兩段文字，以及輪廓漏選造成的候選中心失效。
- 同一個氣泡內有多個文字項目並不一定表示應分割。沒有凹口證據時保留未解決狀態。
- 不讀取 MangaLens 遮罩、快取、權重或輸出。先前模型實驗輸出僅留在 artifacts 供歷史比較。
- 未改動來源圖、去字圖或來源翻譯 JSON；未把候選中心直接寫回正式排版。
