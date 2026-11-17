# Codex 專案指示

開始處理任務前，先閱讀 [Agent.md](Agent.md) 中的專案架構、資料格式與執行方式。

漫畫去字的「開始偵測 → 逐頁校正 → 保存」工作流使用專案 Skill：[comic-detect-review](.agents/skills/comic-detect-review/SKILL.md)。

AI 導出修復 PNG 及分層 PSD、配置可選的 Klein／Qwen ComfyUI 方法，使用 [comic-psd-export](.agents/skills/comic-psd-export/SKILL.md)。所有必要工作流和工具須在本倉庫交付，不能依賴作者其他工程或固定盤符；模型權重不提交 Git。

## 子代理委派

本專案已在 `.codex/agents/` 配置 `explorer` 和 `worker`。對需要追查程式碼路徑、跨檔分析或修改程式的任務，主代理應主動委派適合的子代理，不必等使用者每次點名模型。

- `explorer`：先進行唯讀探索，定位入口、相關函式、資料流、測試與潛在約束；回報具體檔案和行號。
- `worker`：在修改範圍明確後負責實作與必要驗證；回報改動、測試結果及剩餘風險。

需要探索後才能實作時，先等 `explorer` 回報，再將明確範圍交給 `worker`。若任務只有單一、明確且很小的修改，主代理可直接完成。避免多個代理同時編輯同一批檔案。主代理負責整合結果並向使用者報告。
