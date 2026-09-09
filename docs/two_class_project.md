# 兩類編輯與專案格式 v2

## 操作

- F1「純色填充」：指定範圍直接填色。既有區域保持原色，延伸時沿用鄰接顏色；獨立新增區域取樣，取樣不足時由使用者選色。
- F2「圖像修補」：保留原圖像素，導出為精修 Mask。
- 兩類互斥；加入一類會移除另一類。擦除後兩類皆空，保留原圖。
- 首次偵測自動分類，但沒有可編輯的「自動」層。
- 普通編輯與重開只讀取保存的顏色，不重新分類。
- 「偵測並生成」或「純色填充設定 → 套用並重新分類」更新未經人工修改的範圍；人工分類和擦除受保護。
- 「添加偵測」加入所選類，並從另一類移除同一區域。
- 傳入 Mask 的「取代」重新分類、保留人工決定；「交集」同時裁切兩類，保留顏色並記錄被擦除部分。
- 撤銷／重做同時恢復選區、顏色和人工修改記錄。

## 目錄

```text
ctd_inpainted/
  raw/
    project.json
    pages/01.jpg.npz
    cache/01.jpg.npz
  preview_report.pdf           # 按需生成
  export_pair/                 # 導出待精修／右圖時生成
    01.png
    other_mask/01.png
    colored/01.png
  psd_assets/                  # 導出 PSD 素材時生成
    solid/01.png
    other_mask/01.png
  psd/                        # Photoshop 腳本輸出
```

正式資料只有 `project.json` 和 `pages/`。`project.json` 保存版本、來源頁面清單、設定、偵測器參數與必要統計。
每頁 NPZ 保存 `overlay`（BGRA，Alpha 即純色填充選區）、`other`（圖像修補選區）、`edited`（含擦除的人工修改保護範圍）、格式版本及原圖簽名。
`edited` 是內部修改記錄，不是第三種編輯類型。顏色直接存在 overlay，無需合併相鄰氣泡或重新推測顏色。

頁面使用完整原圖檔名加 `.npz`，所以 `01.jpg` 和 `01.png` 的內部資料不會互蓋。導出同名 PNG 時會明確拒絕同 stem 衝突。

完全空白的已處理頁只保存專案條目，不建立空白頁面或快取檔。若曾經擦除，仍需保存修改記錄，避免文字在重新分類時出現。

`cache/` 的每頁 NPZ 合併文字偵測 Mask、氣泡輪廓、背景取樣及詳細診斷，使用無 pickle 的 NumPy archive。不同計算更新快取時保留其他內容。
清除快取不影響顏色、分類、重開或導出；重新分類缺少文字偵測快取時，會使用專案保存的偵測器設定重新偵測。

不再自動生成 `raw/mask`、`manual_solid`、`manual_other`、`other_mask`、`inpainted`、獨立取樣／氣泡快取目錄，以及獨立設定／報告 JSON。
PDF、標記圖與 Photoshop 可讀的 PNG 只在導出時產生。

## 保存與錯誤處理

正式資料與快取使用臨時檔加原子替換。正式頁面遺失、損壞、尺寸不符或原圖已變更時，不會默默以空白覆寫。
明確重新偵測可為變更後的原圖建立新結果。手動強制填色取樣失敗時，不會悄悄留下未填區域。

v2 不讀取或遷移舊格式。遇到舊 `ctd_inpainted/raw` 會提示先將 `ctd_inpainted` 改名保留，再重新偵測；不會自動刪除原成果。

## Photoshop

在程式中按「導出 PSD 素材」，再於 Photoshop 執行 `create_psds_from_outputs.jsx`。
腳本讀取 `ctd_inpainted/psd_assets/solid` 與 `other_mask`，輸出到 `ctd_inpainted/psd`。

「Mask / 原圖」滑桿直接控制原圖與黑底彩色 Mask 的混合：0% 為原圖，100% 為純 Mask，預設 28% 為淡黃色透明疊加；背景取樣提示不在正式選區上重複染色。新顯示方式首次使用 28%，之後記住使用者調整值。
