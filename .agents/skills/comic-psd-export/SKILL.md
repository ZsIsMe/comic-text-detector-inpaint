---
name: comic-psd-export
description: 支援「生成去字 PSD」及「使用 MeoJSON 生成預排版 PSD」。去字可使用 LaMa、Klein 和 Qwen；預排版用每話的 Meo.json 和可選的作品共用 .ini，將譯文加入已有的去字 PSD。
---

# 去字 PSD 與 MeoJSON 預排版 PSD

讀 [Agent.md](../../../Agent.md) 的資料與保存規則；完整參數及輸出契約見 [AI PSD 導出](../../../docs/ai_psd_export.md)。所有工作流、模型清單、工具都隨本倉庫提供；相對路徑以本倉庫為準，不依賴作者其他工程或固定盤符。

兩項功能可獨立執行。只要求去字時，交付去字 PSD 即完成；只要求預排版時，直接使用已有的去字 PSD。兩項都要求時，先將去字成品放回正式目錄，再開始預排版。

## 生成去字 PSD

- 「生成去字 PSD」沿用 LaMa、Klein、Qwen 三種修復；若使用者指定方法或頁碼，按指定範圍執行。呼叫下列入口時明確傳入所選方法；只需 LaMa 時省略額外方法。
- 額外方法 ID 為 `klein9b_fp8_lanpaint`、`qwen21_int8_convrot`。每套方法使用自己的完整已測工作流，不互相串聯或替換權重／量化版本。
- 額外方法需要可用的 ComfyUI；先用 `check_repair_environment` 檢查所選方法。節點或文件名可用不等於權重雜湊已驗證。
- 缺環境或模型時讀 [環境準備](references/environment.md)，使用本倉庫安裝工具生成方案，按使用者的配置授權執行。保留對方既有環境；已安裝的不同節點版本或不同權重需清楚報告，不能偷偷更新或換成近似模型。

### 執行

- 多頁、多方法的匯出按工作流分批執行：先完成整批頁面的 LaMa，再跑完一個 ComfyUI 方法的所有待修頁，才切換下一個方法，最後統一組裝 PSD。續跑時先接回已提交的任務，複用通過校驗的素材。
- 開始前確認入口的實際執行順序。若既有入口會逐頁交替模型，先按整批用 `export_repair_png` 完成 LaMa，並用 `ComfyRepairClient.run` 按方法分批生成到同一版本，沿用正式頁面選區與人工保護，再以該版本的素材組裝 PSD。

優先用現有 Solid Inpaint MCP。先 `list_app_instances`，按工程識別目標視窗，後續同一視窗工具傳相同 `instance_id`。`open_project`／保存沿用程序，正在編輯、模態視窗或其他任務時等待狀態釋放。保存失敗不繼續讀舊資料導出。

呼叫 `start_psd_export(extra_methods=[...], start_page=..., end_page=..., comfy_url=..., instance_id=...)`。返回 `job_id` 和 `export_name` 後，用 `get_job_status` 跟進至 completed、incomplete 或 failed。它一次完成所選方法素材及 PSD；不把「素材已生成」說成「PSD 已完成」。保留失敗任務的 export_name 以便同一版本續跑。

沒有新 MCP 工具時，先保存並確認沒有其他程序修改該工程，再使用本倉庫 `scripts/batch_export_psd.py --help` 所述 CLI；MCP SDK 是可選依賴，不應為普通 LaMa 使用安裝 ComfyUI。

首次配置可在使用者授權範圍內試跑一頁並檢查輸出，再繼續既定批量。檢查素材與圖層是否完整、尺寸、透明度和報告；未要求品質審查時不自行展開整話視覺校對。不要因單頁慢或連線短暫中斷重複提交；提交狀態不明時先查原任務／保留現場報告。

### 輸出與驗收

- 所選方法都要生成對應透明修復 PNG，並同步加入同頁 PSD。每套方法都從同一已填色底圖獨立生成。
- 空 OTHER 跳過生成，空差分不產生透明 PNG 或空圖層。LaMa 以自己的有效修補 Mask 出圖；Klein／Qwen 使用已驗證的差分規則並保護填色與人工擦除。
- 增強導出先在版本資料夾生成並驗證本批所有 PSD，保留各方法素材及任務記錄。整批驗證通過後，將所選頁面的成品放回工程的 `ctd_inpainted/psd_assets/psds/`，直接替換同名 PSD，不另建備份。保留原圖、正式 raw 及本批以外的頁面。
- 逐頁核對正式目錄中的 PSD 與已驗證成品的 SHA-256；全部一致才回報完成，並提供正式目錄路徑。
- LaMa 修補層預設顯示，額外候選預設隱藏。若所選方法失敗或舊 PSD 未包含新素材，應報未完成，不把缺層 PSD 算成功。
- 核對任務狀態、失敗頁、PSD 與 PNG 的實際存在及後端回讀驗證。回報方法、完成／失敗頁數、輸出路徑和必要限制。生成完成不表示漫畫校對完成。

## 使用 MeoJSON 生成預排版 PSD

1. 按使用者提供的對應關係，為每話配對章節目錄及一份 Meo JSON。從該話的 `ctd_inpainted/psd_assets/psds/` 讀取去字 PSD；按 `transMap` 的原頁名換副檔名匹配（例如 `page01.png` 對應 `page01.psd`）。先核對頁名，遇到缺頁或重名時列出具體檔名。
2. 整部作品可共用一份 `.ini`（LabelPlus 儲存的 UTF-8 JSON）。提供時沿用字體、字級、直排、置中、標點替換及 Meo 樣式選項；省略時採用腳本預設，啟用 `useMeoFontSize` 以套用譯文提供的樣式。每話只在本次執行的選項中設定 `lpTextFilePath`、`psdSource` 與 `target`，避免沿用預設裡記錄的上一話路徑。
3. 用 Photoshop 執行隨附的 [LabelPlus JSX](../../../LabelPlus_Ps_Script_ZS.jsx)。批次呼叫沿用其 Meo 解析器及 `LabelPlus.importFiles(opts)`：用解析出的分組填入 `groupSelected`，用原圖檔名與對應 PSD 檔名填入 `imageSelected`。隨附 JSX 預設開啟手動視窗；自動執行可在臨時副本中替換啟動視窗的入口，保留原 JSX 和作品預設。
4. 將含可編輯文字層的 PSD 存到**各話自己的** `output/`。逐頁回讀，比對文字層數及經預設替換後的譯文內容，核對畫布尺寸、原有修復圖層及可見性。回報每話的成品目錄、PSD 頁數及文字層數。
