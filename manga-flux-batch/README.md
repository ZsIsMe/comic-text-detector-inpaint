# Manga FLUX 批量修复

使用 FLUX.1 Fill，按同名文件批量处理漫画原图和 Mask。

## 文件夹结构

```text
漫画目录\
├── 01.jpg
├── 02.jpg
└── other_mask\
    ├── 01.png
    └── 02.png
```

配对规则：相对子目录和文件名主干相同，扩展名可以不同。

```text
01.jpg  ↔ other_mask\01.png
02.webp ↔ other_mask\02.png
```

## 检查配对

```powershell
python manga-flux-batch/batch_run.py "C:\漫画目录\other_mask" --dry-run
```

`--dry-run` 不生成文件。显示 `配对 0 张` 时不要执行正式任务。

## 执行全部图片

```powershell
python manga-flux-batch/batch_run.py "C:\漫画目录\other_mask"
```

只传入 Mask 文件夹时自动推断：

```text
原图文件夹 = Mask 文件夹的上一级
输出文件夹 = 原图文件夹\flux_inpainted
```

路径包含空格、中文或日文时必须使用双引号。

## 明确指定路径

```powershell
python manga-flux-batch/batch_run.py `
  --original "D:\manga" `
  --mask "D:\manga\other_mask" `
  --output "D:\manga\flux_inpainted"
```

## 常用参数

| 参数 | 作用 |
|---|---|
| `--dry-run` | 只检查配对。 |
| `--limit 1` | 只处理排序后的第一对图片。 |
| `--no-pdf` | 不生成对比 PDF。 |
| `--overwrite` | 重新生成已有结果。 |
| `--output PATH` | 指定输出文件夹。 |

默认跳过已有结果。中断后重新运行相同命令即可续跑。

## 输出

```text
漫画目录\flux_inpainted\
├── 01.jpg
├── 02.jpg
├── batch_report.json
└── comparison.pdf
```

- 输出图片保持原图的相对路径、文件名和扩展名。
- `batch_report.json` 记录成功、跳过、失败和配对异常。
- `comparison.pdf` 每页左侧为紫色 Mask 标记的原图，右侧为修复结果。

## 给 AI 助手

1. 用户给原图目录时，使用其下的 `other_mask`；用户给 Mask 目录时直接使用。
2. 先执行 `--dry-run`。配对为 0 时停止并报告。
3. 正式执行前确认 `http://127.0.0.1:8188/system_stats` 可访问。
4. 用户已经要求执行全部图片时，dry-run 通过后直接运行，不要重复询问。
5. 使用长时间运行的终端任务并持续等待同一个进程，不要因短超时重复启动批处理。
6. 完成后读取 `batch_report.json`，确认失败数量和 `comparison.pdf` 是否存在。
7. 除非用户要求，不分析图片质量，只报告进度和完成情况。

推荐调用：

```powershell
& "C:\Users\zs\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\.venv\Scripts\python.exe" `
  "C:\Users\zs\Documents\ComfyUI 2\manga-flux-batch\batch_run.py" `
  "C:\漫画目录\other_mask"
```

## 运行条件

ComfyUI 需要安装：

- `flux1-Fill-Dev_FP8.safetensors`
- `clip_l.safetensors`
- `t5xxl_fp8_e4m3fn.safetensors`
- `ae.safetensors`

使用其他 Python 时如果缺少 Pillow：

```powershell
python -m pip install -r manga-flux-batch/requirements.txt
```

## 常见错误

| 错误 | 处理 |
|---|---|
| 无法连接 `127.0.0.1:8188` | 启动 ComfyUI。 |
| `配对 0 张` | 检查是否传入了正确的 Mask 文件夹及文件名主干。 |
| 拒绝访问输出目录 | 给运行终端该目录的写入权限。 |
| 工作流校验提示缺少模型 | 检查上述四个模型。 |
