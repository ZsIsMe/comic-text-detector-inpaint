# 兩類編輯與專案格式 v2

## 操作

- F1「純色填充」：指定範圍直接填色。既有區域保持原色，延伸時沿用鄰接顏色；獨立新增區域取樣，取樣不足時由使用者選色。
- F2「圖像修補」：保留原圖像素，導出為精修 Mask。
- 兩類互斥；加入一類會移除另一類。擦除後兩類皆空，保留原圖。
- 首次偵測自動分類，但沒有可編輯的「自動」層。
- 普通編輯與重開只讀取保存的顏色，不重新分類。
- 「偵測並生成」或「純色填充設定 → 套用並重新分類」更新未經人工修改的範圍；人工分類和擦除受保護。
- 本分支只使用 RF-DETR（Koharu Layout HighRes）。GUI「偵測並生成」的氣泡填充選擇與 RF-DETR 參數存於全局 QSettings，成功開始偵測後更新；關閉重開或切換專案會沿用。取消對話框不保存選擇。
- 「添加偵測」加入所選類，並從另一類移除同一區域。
- 「添加RF-DETR選區」只將選區內的偵測結果加入目前類型，沿用本機上次保存的裝置、Mask 膨脹尺寸與塗白範圍設定。
- 傳入 Mask 的「取代」重新分類、保留人工決定；「交集」同時裁切兩類，保留顏色並記錄被擦除部分。
- 撤銷／重做同時恢復選區、顏色和人工修改記錄。
- F3「恢復原圖」：以矩形或套索左鍵選區清除兩類 Mask，沿用撤銷及人工擦除保護。切換工具時保留相容的選區模式。
- 目前頁碼、檔名和保存狀態獨立顯示；可輸入頁碼或完整檔名跳轉。圖片列表的「已校對／待回看」與偵測狀態及 OTHER 分開；確認保存後才能標記已校對。
- MCP 精確選區與 GUI 使用相同的兩類資料、撤銷及保存流程；原圖座標的同頁批次需攜帶有效頁面 token，純色新增指定 RGB。執行選區不會自動標記已校對，仍須看修改後預覽再驗收；詳見 [MCP 契約](mcp.md)。
- 局部視窗可用 F1／F2 切換兩類編輯，魔法棒有容差與擴展像素滑桿；類型與滑桿設定即時同步主視窗，兩類選區修改在「套用」時一起保存，取消則丟棄局部副本。

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
  psd_assets/                  # 按需導出 PNG 與分層 PSD
    solid/01.png
    other_mask/01.png
    lama_large_512px/01.png    # 先整批生成的 RGBA 透明修補圖
    psds/01.psd                # psd-tools 直接生成，含 LaMa 修補層
  psd/                        # Photoshop 腳本輸出
```

正式資料只有 `project.json` 和 `pages/`。`project.json` 保存版本、來源頁面清單、當次處理的氣泡填充設定、偵測器參數與必要統計；GUI 的下次預設選擇讀取全局 QSettings。
每頁 NPZ 保存 `overlay`（BGRA，Alpha 即純色填充選區）、`other`（圖像修補選區）、`edited`（含擦除的人工修改保護範圍）、格式版本及原圖簽名。
`edited` 是內部修改記錄，不是第三種編輯類型。顏色直接存在 overlay，無需合併相鄰氣泡或重新推測顏色。

每頁的 `state_digest` 與 `reviewed_digest` 存在 `project.json`，不增加 NPZ 圖層。校對摘要涵蓋來源簽名、兩類選區、填色與人工修改記錄；兩摘要相同才算已校對。實質修改後需回看，相同內容重存或撤銷回已校對內容則仍有效。沒有摘要的既有 v2 頁面可直接建立校對標記，無需重新偵測。

頁面使用完整原圖檔名加 `.npz`，所以 `01.jpg` 和 `01.png` 的內部資料不會互蓋。導出同名 PNG 時會明確拒絕同 stem 衝突。

完全空白的已處理頁只保存專案條目，不建立空白頁面或快取檔。若曾經擦除，仍需保存修改記錄，避免文字在重新分類時出現。

`cache/` 的每頁 NPZ 合併氣泡輪廓、背景取樣及詳細診斷，使用無 pickle 的 NumPy archive。文字偵測 Mask 只在偵測與分類過程中使用，不另存為選區或快取欄位。
清除快取不影響顏色、分類、重開或導出；重新分類會使用 HighRes 重新偵測文字，沿用有效的 RF-DETR 參數。舊 v2 專案若記錄了已移除的模型，改用 HighRes 安全預設，不沿用舊模型專屬參數；單純重開或跳過已處理頁面不會因此重新偵測，也不改寫其人工修改。

不再自動生成 `raw/mask`、`manual_solid`、`manual_other`、`other_mask`、`inpainted`、獨立取樣／氣泡快取目錄，以及獨立設定／報告 JSON。
PDF、標記圖、Photoshop 可讀的 PNG 與分層 PSD 只在導出時產生。普通保存、重開或偵測不執行 LaMa 修補。

## 保存與錯誤處理

正式資料與快取使用臨時檔加原子替換。正式頁面遺失、損壞、尺寸不符或原圖已變更時，不會默默以空白覆寫。
明確重新偵測可為變更後的原圖建立新結果。手動強制填色取樣失敗時，不會悄悄留下未填區域。

v2 不讀取或遷移舊格式。遇到舊 `ctd_inpainted/raw` 會提示先將 `ctd_inpainted` 改名保留，再重新偵測；不會自動刪除原成果。

## 矩形內類型互換

主視窗或局部編輯視窗中，按住 Cmd（macOS）或 Ctrl（Windows/Linux）並用右鍵拖出矩形，可把框內**已有**的「純色填充」與「圖像修補」逐像素互換。兩類混在同一矩形也會各自互換；空白與框外不變，與目前工具及 F1/F2 編輯類型無關。普通右鍵拖曳仍清除兩類。主視窗的互換可一次撤銷／重做並自動保存；局部視窗限於目前 ROI，按「套用」才寫回主頁，按「取消」則捨棄。

## PSD 導出與 Photoshop

在程式中按「導出 PSD 素材」。程式先保存目前編輯，再由背景 worker 分兩階段整批導出：

1. 「LaMa 修補」：先完成全部 `psd_assets/solid`、`psd_assets/other_mask` PNG 與 `psd_assets/lama_large_512px/<stem>.png` 透明修補圖。
2. 「PSD 組裝」：全部修補 PNG 完成後，使用 `psd-tools==1.26.1` 從原圖、solid 和落盤修補 PNG 生成 `ctd_inpainted/psd_assets/psds/<stem>.psd`；這一階段不再推論模型。

UI 以進度條和目前階段、該階段頁碼、檔名顯示進度，完成後顯示 PSD 所在路徑。PNG 素材與修補 PNG 每次明確導出都依當次資料更新；同名既有 PSD 預設跳過，避免覆蓋 Photoshop 手動修改。

新 PSD 自下而上為：

- `bg`：原圖背景，保留來源尺寸與位深；RGB 來源的既有 ICC 直接寫入。
- `overlay-manual`：正式保存的純色 overlay；Alpha 非零時才建立。
- `lama_large_512px`：讀取同名透明修補 PNG，放置 LaMa 修補原圖 `other_mask > 0` 區域的普通 RGBA 圖層，預設隱藏。

LaMa 的 8-bit 修補 PNG 在 OTHER 內 Alpha 為 255，區外 RGBA 全為 0；PSD 修補層與此 PNG 同源，寫入 16-bit PSD 時按位深轉存，區內仍為完全不透明。新 PSD 不建立黑白 `other_mask` 圖層或 `OTHER_CHANNEL`。原有 `other_mask` PNG 仍維持黑白 Mask，`solid` PNG 仍維持透明填色 overlay。OTHER 為空時仍輸出全透明修補 PNG，但不載入或執行模型，PSD 可留下空透明的 LaMa 圖層。

8-bit CMYK 或帶灰階 ICC 的來源會先轉為 sRGB，並寫入轉換後的 ICC。16-bit 灰階 ICC 或其他不能套用到 RGB PSD 的描述檔會明確報錯。

LaMa 只在明確導出 PSD 時推論，不更新兩類分類、人工修改記錄、原圖或 `raw/` 正式頁面。16-bit 背景保留原位深；LaMa 使用 8-bit 輸入與修補內容，不將整張背景降為 8-bit。修補結果需在 Photoshop 中自行開啟圖層檢查和精修，不代表已完成頁面校對。

舊 `create_psds_from_outputs.jsx` 仍可選用。腳本讀取 `ctd_inpainted/psd_assets/solid` 與 `other_mask`，維持輸出到 `ctd_inpainted/psd`，並保留原本的 `OTHER_CHANNEL` 與 Photoshop 動作流程；它與新 `psd_assets/psds/` 輸出分開。

左側 F1「純色填充」和 F2「圖像修補」預覽中的白色均為正式 overlay 的完整塗白選區，包含氣泡內縮填色。「轉為圖像修補」會搬移所選白區。黃色背景取樣提示以固定 28% 不透明度疊加，不覆蓋已填色或圖像修補的像素；由「顯示背景選區」獨立控制。擦除後不再顯示白色。滑桿 100% 為黑底選區，0% 為原圖；預設 70%，沿用舊版保存的滑桿比例。筆刷及局部視窗使用相同顯示規則。
