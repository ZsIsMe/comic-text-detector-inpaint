# AI 導出修復素材與 PSD

原有「導出 PSD 素材」仍使用本機 LaMa，不需要 ComfyUI。AI 可透過隨倉庫提供的 [comic-psd-export](../.agents/skills/comic-psd-export/SKILL.md) 或 MCP，額外選擇以下方法；沒有新增 GUI 選單。

| 方法 ID | 工作流及模型 |
| --- | --- |
| `klein9b_fp8_lanpaint` | FLUX.2 Klein **9B FP8 distilled**、Qwen3 8B FP8 Mixed、FLUX.2 VAE；LanPaint，4 steps、CFG 1、Euler/simple、NumSteps 2 |
| `qwen21_int8_convrot` | Qwen Image 2.1 INT8 ConvRot、Qwen3-VL 8B INT8 ConvRot、專用 BF16 VAE；25 steps、CFG 1、Euler/simple |

使用本機已測 API 工作流。Klein 使用實際批量試跑的 LanPaint_KSampler 版本；不宣稱與另一份使用 CustomAdvanced sampler 的 UI 工作流完全相同。兩方法各自從原圖＋solid 合成底圖生成，不互相串接；輸入的 OTHER 由工作流擴張 8px。額外透明素材使用完整結果與底圖的 RGB 最大差值 ≥12、3px 橢圓閉運算、保留至少 16px 的八連通區、向外擴張 8px，再扣除 solid 與人工擦除保護。差分不再與 OTHER 相交。

## 交付內容與環境

`comfy_repair.py`、`repair_workflows/` 的兩套 API JSON／方法清單／安裝清單、`scripts/setup_repair_workflows.py`、批量導出腳本和 Skill 全部隨倉庫交付，不需要作者其他工程。工作流只使用 ComfyUI 和公開節點；第三方節點由安裝工具從固定公開提交取得，不把整個 ComfyUI 複製進本工程。

模型權重不提交 Git。六個權重的固定 Hugging Face revision、大小與 SHA-256，以及四個節點套件的固定 Git 提交，記錄於 `repair_workflows/setup_manifest.json`。配置方式見 [環境準備](../.agents/skills/comic-psd-export/references/environment.md)。本機驗證的 ComfyUI 為 0.45.3；其他環境以實際節點 schema、模型與單頁試跑結果判斷相容性。

## MCP

先啟動 GUI 並載入 [MCP 橋接](mcp.md)，然後：

```text
list_app_instances()
check_repair_environment(extra_methods=["klein9b_fp8_lanpaint", "qwen21_int8_convrot"])
start_psd_export(extra_methods=["klein9b_fp8_lanpaint", "qwen21_int8_convrot"],
                 start_page=1, end_page=1, instance_id="目標視窗 ID")
get_job_status(job_id="返回的 job_id", instance_id="同一視窗 ID")
```

`check_repair_environment` 為獨立只讀檢查，不要求 GUI 視窗；檢查節點 schema 和模型下拉選項，**不代表已計算本機權重 SHA-256**。其他視窗操作須沿用 `instance_id`。導出前保存 GUI 編輯；保存失敗、正在編輯或其他任務未結束時不啟動。頁碼從 1 開始，包含首尾；省略範圍為全部頁面。

`start_psd_export` 立即回傳任務，背景依次生成素材並組裝 PSD。狀態為 `running`、`completed`、`incomplete` 或 `failed`；檢查 `result`、`failed_pages`、`errors`、`preserved_stale`，不能只看進度百分比。未傳額外方法沿用本機 LaMa。

可選 `comfy_url` 指定服務；`model_map` 格式為 `{方法ID: {"unet": "子目錄/模型.safetensors", "clip": "...", "vae": "..."}}`，使用服務下拉選項中的實際名稱。映射僅調整位置，不應換成別的權重或量化版本。

## 命令列

先保存工程並停止其他程序修改相同資料夾。此命令遞迴查找 root 下的 v2 工程，逐工程、逐頁執行：

```powershell
.\.venv\Scripts\python.exe scripts/batch_export_psd.py --root "漫畫根目錄" --report-dir "報告目錄" --extra-methods klein9b_fp8_lanpaint qwen21_int8_convrot --comfy-url http://127.0.0.1:8188
```

`--preflight-only` 只檢查工程資料；Comfy 環境另用配置工具或 MCP 預檢。可加 `--model-map mapping.json` 和 `--export-name trial_001`。省略額外方法及 export-name 時，沿用既有 LaMa 輸出契約。

## 輸出、續跑與保留

增強導出位於 `ctd_inpainted/psd_assets/ai_exports/<export_name>/`，預設新建唯一版本；MCP 使用任務 ID 作版本名。指定相同 export_name 可續跑，名稱為 1–80 個英數字、底線或連字號，首字元須為英數字。

每個版本包含 solid、other_mask、LaMa、所選方法的透明 PNG，`repair_masks/<方法>/`、`workflow_results/<方法>/` 完整生成結果、`workflow_jobs/<方法>/` 任務記錄及 `psds/`。PSD 底層為 bg，接著是有效 solid、預設顯示的 LaMa、預設隱藏的 Klein／Qwen 候選。空 OTHER 不呼叫模型；空結果不生成透明修復 PNG 或空圖層。

先完成所有頁的素材，再從落盤 PNG 組裝 PSD，組裝不重新推論。任一所選方法失敗，該頁不算完成。素材、輸入、工作流與配置摘要及檔案雜湊用於續跑；已有或人工修改的 PSD 保留，摘要不符時報 stale／未完成。原圖與正式 raw 不回寫。

Comfy 提交前保存 prompt/client ID；連線超時不盲目重送。以相同版本續跑時先查原 queue/history，仍不能確認則報提交狀態不明，保留記錄供排查。每頁有鎖避免重複提交；確認原程序已停止後才人工處理殘留鎖。服務歷史被清除可能無法恢復未取回的結果。

## 本機驗證記錄

2026-11-17 在 RTX 4070 Ti SUPER 16GB、ComfyUI 0.45.3 上，以一頁 847×1200 圖像實測最終版本：LaMa 約 2.8 秒、Klein 流程約 20.5 秒、Qwen 流程約 22.9 秒，含 PSD 組裝總計約 47.6 秒。這是當時環境的單頁耗時，包含請求／模型切換，並非通用速度保證。每 2 秒採樣的整張 GPU 總佔用峰值為 15,426 MiB，包含其他程序及原生 LaMa，不能當作單一模型的顯存需求或精確峰值。

實際 PSD 回讀確認五層順序、尺寸與可見性，三個修復圖層的 RGB／Alpha 與各自 PNG 逐像素一致；原圖和正式 raw 雜湊未變。相同版本再次導出時禁止所有 Comfy POST，仍成功複用素材並保留既有 PSD。這些核對驗證的是生成與分層導出流程，候選修復品質仍需人工選擇和精修。
