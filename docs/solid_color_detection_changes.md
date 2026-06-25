# 純色判斷修改說明

本文記錄本次對 Solid Inpaint「純色背景判斷」和「背景選區可視化」的主要修改。

## 修改背景

原本的純色判斷主要依賴文字 mask 外圍的一圈 `sample_ring`。流程大致是：

```text
文字 mask
  -> 膨脹得到 repair_area
  -> repair_area 再往外膨脹
  -> 扣掉 repair_area 得到 sample_ring
  -> 用 sample_ring 的顏色分布判斷背景是否純色
```

這個方法在簡單白底文字上有效，但遇到以下情況容易誤判：

```text
1. 文字靠近氣泡邊緣，sample_ring 會碰到線稿或框線。
2. 背景區域較大時，一圈 ring 不能代表整個氣泡背景。
3. 非封閉或半封閉區域會讓取樣範圍跑到外部背景。
4. 單個異常文字區塊可能拖累整頁背景選區顯示。
```

所以本次修改的核心方向是：不再只把「文字外面一圈」當背景，而是更接近 Photoshop 魔法棒的思路，從文字外側找一個可靠種子點，取得和該種子連通且顏色接近的背景區域，再用這個背景區域判斷是否純色。

## 核心概念

### repair_area

`repair_area` 是真正要被覆蓋或修補的區域。

它由文字 mask 膨脹得到，用來確保文字邊緣、抗鋸齒和細小殘留能被一起處理。

```text
文字 mask -> 膨脹 -> repair_area
```

純色判斷通過後，`repair_area` 會被填入估算出的背景色，寫入透明 overlay。

### sample_ring

`sample_ring` 仍然保留，但角色變了。

它不再是唯一的背景取樣來源，而是作為：

```text
1. 魔法棒種子點搜尋的備援區域。
2. 背景選區和文字區域是否有接觸的參考。
3. 魔法棒失敗時的 fallback。
```

也就是說，`sample_ring` 現在更像保底取樣，不是最終判斷的主體。

### background_sample

`background_sample` 是本次新增的主要取樣區域。

它由每個文字區塊獨立計算：

```text
repair_area 外側搜尋候選點
  -> 過濾太暗、梯度太高、太靠邊的點
  -> 選出多個候選種子
  -> 對種子執行 floodFill / 魔法棒式連通選區
  -> 排除文字 mask 和 repair_area
  -> 過濾碰到 ROI 邊界或面積異常的結果
  -> 選擇分數最高的背景選區
```

如果找不到可靠的 `background_sample`，該區塊會退回使用 `sample_ring`。

## 魔法棒式背景選區

新邏輯位於 `detect_solid_inpaint_folder.py`，主要由以下函式組成：

```text
_candidate_seed_points()
_wand_selection_from_seed()
_background_wand_sample()
iter_background_samples_from_mask()
background_sample_from_mask()
```

候選種子點會優先選擇：

```text
1. 在 repair_area 外側一定距離之外。
2. 不在文字 mask 內。
3. 灰階值不太暗。
4. 梯度較低，避免落在線稿或邊界上。
5. 顏色接近候選區域 dominant color。
6. 不太靠近局部 ROI 邊緣。
```

對每個種子點會執行 OpenCV flood fill。這裡的行為類似 Photoshop 魔法棒：

```text
從種子點出發，
只選取與種子顏色差異在容差內，
且空間上連通的區域。
```

為了避免選到外部背景，會拒絕以下結果：

```text
1. 選區碰到局部 ROI 邊界。
2. 選區面積超過 ROI 的固定比例。
3. 選區太小，沒有足夠取樣像素。
4. 選區和 repair_area / sample_ring 幾乎沒有接觸。
```

這能降低「文字靠近氣泡邊緣時選到氣泡外面」的機率。

## 純色判斷方式

取得 `background_sample` 後，純色判斷仍然使用顏色分布統計，而不是只看單點顏色。

主要指標包括：

```text
p90 - p10 色彩跨度
dominant color peak ratio
與 dominant color 接近的像素比例
p95 color delta
有效取樣像素數
```

若整個 `background_sample` 不通過，仍會嘗試方向性取樣：

```text
top
bottom
left
right
```

方向性取樣會使用更嚴格的門檻。這是為了處理某些區域只有某一側背景可靠的情況，同時避免把複雜背景誤判成純色。

## 每個文字區塊獨立計算

本次修改後，背景選區按文字區塊獨立計算，而不是整頁一次性成功或失敗。

這點很重要，因為之前會出現：

```text
某一個麻煩區塊計算失敗或很慢
  -> 整頁背景選區都不顯示
```

現在改為：

```text
每個區塊獨立產生 background_sample
  -> 成功的區塊先顯示
  -> 失敗的區塊退回 sample_ring
  -> 不影響其他區塊
```

對應函式是：

```text
iter_background_samples_from_mask()
```

它會逐個區塊 `yield` 結果，供 UI 做 partial emit。

## UI 可視化與 partial emit

UI 左側新增了背景選區可視化，使用淡黃色半透明層顯示 `background_sample`。

為了避免切頁或手動刷 mask 時卡頓，UI 採用延遲和 worker thread：

```text
切換頁面
  -> 先立即顯示原圖 + mask
  -> 優先讀取 npz cache
  -> 沒有 cache 才延遲啟動背景選區計算

手動修改 mask
  -> 先更新 mask 顯示
  -> 延遲重算 background_sample
  -> 舊任務會被取消
```

worker 計算時不是等整頁完成才更新，而是每完成一個區塊就發出 partial result：

```text
BackgroundSampleWorker.partial
BackgroundSampleWorker.finished
```

因此頁面上會逐步出現黃色背景選區。這樣某個慢區塊不會阻塞已經成功的區塊顯示。

## npz cache

背景選區會保存為：

```text
ctd_inpainted/background_sample_cache/<name>.npz
```

cache 內包含：

```text
mask_hash
sample
```

`mask_hash` 用來確認 cache 是否對應當前 mask。當使用者手動修改 mask 後，舊 cache 的 hash 不匹配，UI 會重新計算並覆蓋該頁 `.npz`。

目前以下流程都會產生或更新 `.npz`：

```text
偵測並生成
不修改 mask 重生成
使用 ysgyolo 更新 mask 圖
單頁手動修改 mask 後自動重生成
切換頁面時補算缺失 cache
```

批量生成時會在 `regenerate_image_from_mask()` 裡同步寫入 `.npz`。這樣切換到新頁時，UI 可以直接讀 cache，不需要臨時計算整頁背景選區。

## 本次修改後的輸出關係

每頁主要輸出變成：

```text
mask/<name>.png
  文字 mask

other_mask/<name>.png
  不適合純色自動處理的 repair_area

inpainted/<name>.png
  通過純色判斷後生成的透明 overlay

background_sample_cache/<name>.npz
  UI 顯示和 debug 用的背景取樣選區 cache
```

其中 `background_sample_cache` 不是最終修圖輸出，但它能幫助檢查純色判斷到底採樣了哪片背景。

## 已知限制

新方法更接近 Photoshop 魔法棒，但仍不是 Photoshop 選區演算法的完整複製。

仍可能出錯的情況包括：

```text
1. 氣泡沒有封閉，且內外背景顏色非常接近。
2. 氣泡邊界很淡或被壓縮噪聲破壞。
3. 文字緊貼線稿，候選種子點不足。
4. 大面積漸變背景局部看起來接近純色。
5. 多個文字區塊被 merge 成過大的局部區域。
```

這些情況下，工具會盡量保守：不可靠的區域應進入 `other_mask`，由人工或其他修補流程處理。

## 相關檔案

主要修改集中在：

```text
detect_solid_inpaint_folder.py
solid_inpaint_ui.py
```

核心判斷和 cache 寫入在 `detect_solid_inpaint_folder.py`。

UI 的黃色半透明背景選區、延遲計算、worker thread、partial emit 和 cache 讀取在 `solid_inpaint_ui.py`。
