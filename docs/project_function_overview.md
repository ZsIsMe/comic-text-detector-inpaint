# Solid Inpaint 功能定位說明

Solid Inpaint 是一個漫畫文字處理輔助工具。它的核心目標不是完整自動修圖，也不是翻譯工具，而是：

```text
偵測漫畫圖片中的文字區域，
自動處理背景可靠的純色區域，
並把不適合自動處理的區域標記出來。
```

換句話說，它負責漫畫修圖流程中「文字 mask 偵測」和「純色背景安全塗白」這一段。

## 適合解決的問題

在漫畫翻譯、嵌字或修圖流程中，很多文字位於白底、黑底、灰底或其他接近純色的區域。這類文字通常不需要複雜 inpainting，只要用周圍背景色覆蓋即可。

但也有一些文字位於網點、漸變、圖案、人物、線稿或複雜背景上。這些區域如果強行用純色覆蓋，容易破壞畫面。

Solid Inpaint 的設計重點就是區分這兩類區域：

```text
可靠純色背景
  -> 自動生成透明塗白 overlay

不可靠或複雜背景
  -> 生成 other_mask，提醒人工檢查或交給其他修補流程
```

## 基本工作流

```text
輸入一個圖片資料夾
  ↓
使用 comic text detector 偵測文字 mask
  ↓
把文字 mask 分成多個文字區塊
  ↓
分析每個區塊周圍背景是否接近純色
  ↓
可靠區域生成 inpainted overlay
  ↓
不可靠區域生成 other_mask
  ↓
輸出報告與可選 PDF 預覽
```

## 輸入

輸入是一個圖片資料夾，支援常見圖片格式，例如：

```text
.bmp
.jpg
.jpeg
.png
```

工具會按檔名排序處理資料夾內的圖片。

## 輸出

所有輸出會放在輸入資料夾下的：

```text
ctd_inpainted/
```

主要輸出包括：

```text
mask/<name>.png
```

文字偵測 mask。白色代表偵測到的文字區域。

```text
inpainted/<name>.png
```

透明 BGRA overlay。它不是完整修好的原圖，而是一張和原圖同尺寸的透明圖層，只在可安全純色覆蓋的區域有內容。

```text
other_mask/<name>.png
```

需要人工注意的區域。這些區域可能背景不夠純、取樣不足、色彩分布不穩定，或不適合用純色直接覆蓋。

```text
solid_inpaint_report.json
```

每頁處理統計和 debug 資訊。

```text
preview_report.pdf
```

檢查用 PDF。每頁包含 original、preview、mask、other_mask 四個視圖。命令行批量處理會自動生成；圖形界面中可以手動生成。

## 圖形界面功能

GUI 主要提供以下功能：

```text
選擇圖片資料夾
打開最近資料夾
偵測並生成
查看圖片列表
查看 mask / 原圖疊加預覽
查看 inpainted 合成預覽
顯示 other_mask
手動編輯 mask
撤銷 / 重做
自動重新生成當前頁 overlay
打開輸出資料夾
生成 PDF 預覽
打開 PDF 預覽
```

手動編輯 mask 後，工具會保存新的 mask，並基於修改後的 mask 重新生成當前頁的 overlay 和 other_mask。

## 命令行功能

命令行模式適合批量處理整個資料夾：

```bash
.venv/bin/python detect_solid_inpaint_folder.py /path/to/image_folder
```

它會自動完成偵測、生成 overlay、生成 other_mask、寫入 JSON 報告，並生成 PDF 預覽。

## 不是什麼

Solid Inpaint 不是完整漫畫翻譯器。

它不負責：

```text
OCR
翻譯
重新嵌字
完整複雜背景修補
生成最終合成成品圖
```

它也不是通用 inpainting 模型。對複雜背景，它的策略不是硬修，而是標記到 `other_mask`，讓用戶進一步人工處理或交給其他工具。

## 一句話定位

```text
Solid Inpaint 是一個漫畫文字 mask 偵測、純色背景安全塗白、問題區域標記工具。
```

它最適合放在漫畫翻譯與修圖流程的前半段，用來快速處理簡單背景文字，並把複雜背景文字整理出來供後續處理。
