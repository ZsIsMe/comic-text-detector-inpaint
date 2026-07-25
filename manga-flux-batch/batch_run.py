import argparse
import copy
import hashlib
import io
import json
import mimetypes
from pathlib import Path
import secrets
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid

try:
    from PIL import Image, ImageFilter, ImageOps
except ImportError:
    print("缺少 Pillow。请使用 ComfyUI 的 Python 运行，或执行: python -m pip install Pillow")
    raise SystemExit(2)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def request_bytes(url, method="GET", data=None, headers=None, timeout=30):
    request = Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read()
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"无法连接 ComfyUI: {error.reason}") from error


def request_json(url, method="GET", value=None, timeout=30):
    data = None
    headers = {}
    if value is not None:
        data = json.dumps(value).encode("utf-8")
        headers["Content-Type"] = "application/json"
    return json.loads(request_bytes(url, method, data, headers, timeout).decode("utf-8"))


def multipart_body(fields, file_field, file_path):
    boundary = f"----manga-flux-batch-{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode())
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def upload_image(server, file_path, subfolder, timeout):
    body, content_type = multipart_body(
        {"type": "input", "overwrite": "true", "subfolder": subfolder},
        "image",
        file_path,
    )
    response = request_bytes(
        f"{server}/upload/image",
        "POST",
        body,
        {"Content-Type": content_type},
        timeout,
    )
    uploaded = json.loads(response.decode("utf-8"))
    relative = Path(uploaded.get("subfolder", "")) / uploaded["name"]
    return relative.as_posix()


def image_files(root, recursive, excluded_roots=()):
    iterator = root.rglob("*") if recursive else root.iterdir()
    files = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if any(path == excluded or excluded in path.parents for excluded in excluded_roots):
            continue
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().lower())


def pair_images(original_root, mask_root, output_root, recursive):
    excluded = []
    if original_root in mask_root.parents:
        excluded.append(mask_root)
    if original_root in output_root.parents:
        excluded.append(output_root)
    originals = image_files(original_root, recursive, excluded)
    masks = image_files(mask_root, recursive)
    mask_lookup = {}
    for path in masks:
        relative = path.relative_to(mask_root)
        key = (relative.parent / relative.stem).as_posix().casefold()
        mask_lookup.setdefault(key, []).append(path)
    pairs = []
    missing_masks = []
    ambiguous_masks = []
    invalid_sizes = []
    for original in originals:
        relative = original.relative_to(original_root)
        key = (relative.parent / relative.stem).as_posix().casefold()
        candidates = mask_lookup.get(key, [])
        if not candidates:
            missing_masks.append(relative.as_posix())
            continue
        if len(candidates) > 1:
            ambiguous_masks.append(
                {
                    "file": relative.as_posix(),
                    "masks": [path.relative_to(mask_root).as_posix() for path in candidates],
                }
            )
            continue
        mask = candidates[0]
        try:
            with Image.open(original) as original_image, Image.open(mask) as mask_image:
                original_size = original_image.size
                mask_size = mask_image.size
        except OSError as error:
            invalid_sizes.append({"file": relative.as_posix(), "reason": str(error)})
            continue
        if original_size != mask_size:
            invalid_sizes.append(
                {"file": relative.as_posix(), "original": original_size, "mask": mask_size}
            )
            continue
        pairs.append((relative, original, mask))
    return pairs, missing_masks, ambiguous_masks, invalid_sizes


def queue_prompt(server, workflow, client_id, timeout):
    response = request_json(
        f"{server}/prompt",
        "POST",
        {"prompt": workflow, "client_id": client_id},
        timeout,
    )
    node_errors = response.get("node_errors") or {}
    if node_errors:
        raise RuntimeError(f"工作流校验失败: {json.dumps(node_errors, ensure_ascii=False)}")
    return response["prompt_id"]


def wait_for_prompt(server, prompt_id, timeout_seconds, poll_seconds, progress_label, progress_interval):
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    last_progress = 0
    while time.monotonic() < deadline:
        history = request_json(f"{server}/history/{prompt_id}", timeout=30)
        if prompt_id in history:
            item = history[prompt_id]
            status = item.get("status", {})
            if status.get("status_str") != "success" or not status.get("completed"):
                raise RuntimeError(f"ComfyUI 执行失败: {json.dumps(status, ensure_ascii=False)}")
            return item
        elapsed = int(time.monotonic() - started)
        if elapsed - last_progress >= progress_interval:
            print(f"{progress_label} 处理中，已用 {elapsed} 秒", flush=True)
            last_progress = elapsed
        time.sleep(poll_seconds)
    raise TimeoutError(f"等待任务超时: {prompt_id}")


def output_image_info(history_item, save_node_id):
    outputs = history_item.get("outputs", {})
    node_output = outputs.get(str(save_node_id), {})
    images = node_output.get("images") or []
    if not images:
        for value in outputs.values():
            images.extend(value.get("images") or [])
    if not images:
        raise RuntimeError("执行成功，但历史记录中没有输出图片")
    return images[-1]


def download_output(server, info, timeout):
    query = urlencode(
        {
            "filename": info["filename"],
            "subfolder": info.get("subfolder", ""),
            "type": info.get("type", "output"),
        }
    )
    return request_bytes(f"{server}/view?{query}", timeout=timeout)


def save_result(image_bytes, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".png":
        destination.write_bytes(image_bytes)
        return
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        output_format = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".webp": "WEBP",
            ".bmp": "BMP",
            ".tif": "TIFF",
            ".tiff": "TIFF",
        }.get(destination.suffix.lower())
        if output_format is None:
            raise ValueError(f"不支持的输出格式: {destination.suffix}")
        options = {"quality": 95, "subsampling": 0} if output_format == "JPEG" else {}
        if output_format == "WEBP":
            options = {"lossless": True, "quality": 100}
        image.save(destination, output_format, **options)


def build_pdf(pairs, output_root, pdf_name, opacity, mask_expand, invert_mask):
    pages = []
    missing = []
    for relative, original_path, mask_path in pairs:
        result_path = output_root / relative
        if not result_path.is_file():
            missing.append(relative.as_posix())
            continue
        with Image.open(original_path) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(mask_path) as source:
            mask = ImageOps.exif_transpose(source).convert("L")
        with Image.open(result_path) as source:
            result = ImageOps.exif_transpose(source).convert("RGB")
        if original.size != mask.size or original.size != result.size:
            missing.append(relative.as_posix())
            continue
        binary_mask = mask.point(lambda value: 255 if value >= 128 else 0)
        if invert_mask:
            binary_mask = ImageOps.invert(binary_mask)
        if mask_expand > 0:
            binary_mask = binary_mask.filter(ImageFilter.MaxFilter(mask_expand * 2 + 1))
        alpha = binary_mask.point(lambda value: round(value * opacity))
        overlay = Image.new("RGBA", original.size, (157, 52, 235, 0))
        overlay.putalpha(alpha)
        marked = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
        gutter = 24
        page = Image.new("RGB", (original.width * 2 + gutter, original.height), "white")
        page.paste(marked, (0, 0))
        page.paste(result, (original.width + gutter, 0))
        pages.append(page)
    if not pages:
        return None, missing
    pdf_path = output_root / pdf_name
    pages[0].save(
        pdf_path,
        "PDF",
        save_all=True,
        append_images=pages[1:],
        resolution=200.0,
        quality=92,
    )
    for page in pages:
        page.close()
    return pdf_path, missing


def workflow_for_pair(base_workflow, config, original_input, mask_input, relative):
    workflow = copy.deepcopy(base_workflow)
    original_node = str(config["original_node_id"])
    mask_node = str(config["mask_node_id"])
    save_node = str(config["save_node_id"])
    sampler_node = str(config.get("sampler_node_id", "14"))
    workflow[original_node]["inputs"]["image"] = original_input
    workflow[mask_node]["inputs"]["image"] = mask_input
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:10]
    workflow[save_node]["inputs"]["filename_prefix"] = f"manga_flux_batch/{digest}_{relative.stem}"
    if sampler_node in workflow and "seed" in workflow[sampler_node].get("inputs", {}):
        workflow[sampler_node]["inputs"]["seed"] = secrets.randbits(63)
    return workflow


def parse_arguments():
    parser = argparse.ArgumentParser(description="FLUX.1 Fill 漫画双文件夹批处理")
    parser.add_argument(
        "mask_folder",
        nargs="?",
        help="Mask 文件夹；只传这一个路径时，原图文件夹自动取其上一级",
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--original")
    parser.add_argument("--mask", dest="mask_option")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_arguments()
    script_dir = Path(__file__).resolve().parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = script_dir / config_path
    config = load_json(config_path)
    server = config.get("server", "http://127.0.0.1:8188").rstrip("/")
    if args.mask_folder and args.mask_option:
        raise ValueError("位置参数和 --mask 不能同时使用")
    mask_argument = args.mask_folder or args.mask_option
    paths_from_command_line = bool(mask_argument or args.original)
    if mask_argument:
        mask_root = Path(mask_argument).expanduser().resolve()
        original_root = Path(args.original).expanduser().resolve() if args.original else mask_root.parent
    elif args.original:
        original_root = Path(args.original).expanduser().resolve()
        mask_root = original_root / "other_mask"
    else:
        original_root = Path(config["original_folder"]).expanduser().resolve()
        mask_root = Path(config["mask_folder"]).expanduser().resolve()
    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    elif paths_from_command_line:
        output_root = original_root / "flux_inpainted"
    else:
        output_root = Path(config.get("output_folder") or original_root / "flux_inpainted").expanduser().resolve()
    recursive = bool(config.get("recursive", False))
    skip_existing = bool(config.get("skip_existing", True)) and not args.overwrite
    if not original_root.is_dir() or not mask_root.is_dir():
        raise ValueError("原图或 Mask 文件夹不存在")
    if output_root in {original_root, mask_root}:
        raise ValueError("输出文件夹不能与原图或 Mask 文件夹相同")
    workflow_path = Path(config.get("workflow", "workflow_api.json"))
    if not workflow_path.is_absolute():
        workflow_path = script_dir / workflow_path
    base_workflow = load_json(workflow_path)
    pairs, missing_masks, ambiguous_masks, invalid_sizes = pair_images(
        original_root, mask_root, output_root, recursive
    )
    if args.limit > 0:
        pairs = pairs[: args.limit]
    print(
        f"配对 {len(pairs)} 张，缺少 Mask {len(missing_masks)} 张，"
        f"Mask 重名冲突 {len(ambiguous_masks)} 张，尺寸异常 {len(invalid_sizes)} 张"
    )
    if args.dry_run:
        for relative, _, _ in pairs:
            print(relative.as_posix())
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    request_json(f"{server}/system_stats", timeout=10)

    client_id = f"manga-flux-batch-{uuid.uuid4()}"
    report = {
        "success": [],
        "skipped": [],
        "failed": [],
        "missing_masks": missing_masks,
        "ambiguous_masks": ambiguous_masks,
        "invalid_sizes": invalid_sizes,
    }
    timeout = int(config.get("request_timeout_seconds", 60))
    prompt_timeout = int(config.get("prompt_timeout_seconds", 600))
    poll_seconds = float(config.get("poll_seconds", 2))
    progress_interval = max(1, int(config.get("progress_interval_seconds", 5)))
    total = len(pairs)
    for index, (relative, original_path, mask_path) in enumerate(pairs, 1):
        destination = output_root / relative
        if skip_existing and destination.is_file():
            report["skipped"].append(relative.as_posix())
            print(f"[{index}/{total}] 跳过已有: {relative}")
            continue
        print(f"[{index}/{total}] 开始: {relative}")
        started = time.monotonic()
        try:
            original_subfolder = (Path("manga_flux_batch") / "original" / relative.parent).as_posix()
            mask_subfolder = (Path("manga_flux_batch") / "mask" / relative.parent).as_posix()
            original_input = upload_image(server, original_path, original_subfolder, timeout)
            mask_input = upload_image(server, mask_path, mask_subfolder, timeout)
            workflow = workflow_for_pair(base_workflow, config, original_input, mask_input, relative)
            prompt_id = queue_prompt(server, workflow, client_id, timeout)
            history = wait_for_prompt(
                server,
                prompt_id,
                prompt_timeout,
                poll_seconds,
                f"[{index}/{total}] {relative}",
                progress_interval,
            )
            output_info = output_image_info(history, config["save_node_id"])
            image_bytes = download_output(server, output_info, timeout)
            save_result(image_bytes, destination)
            elapsed = round(time.monotonic() - started, 1)
            report["success"].append({"file": relative.as_posix(), "seconds": elapsed})
            print(f"[{index}/{total}] 完成: {destination} ({elapsed}s)")
        except Exception as error:
            report["failed"].append({"file": relative.as_posix(), "error": str(error)})
            print(f"[{index}/{total}] 失败: {relative}: {error}", file=sys.stderr)
        save_json(output_root / "batch_report.json", report)

    if config.get("create_pdf", True) and not args.no_pdf:
        pdf_path, missing_outputs = build_pdf(
            pairs,
            output_root,
            config.get("pdf_name", "comparison.pdf"),
            float(config.get("overlay_opacity", 0.35)),
            int(config.get("mask_expand", 4)),
            bool(config.get("invert_mask", False)),
        )
        report["pdf"] = str(pdf_path) if pdf_path else None
        report["pdf_missing_outputs"] = missing_outputs
        if pdf_path:
            print(f"对比 PDF: {pdf_path}")
    save_json(output_root / "batch_report.json", report)
    print(
        f"结束：成功 {len(report['success'])}，跳过 {len(report['skipped'])}，失败 {len(report['failed'])}"
    )
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("已由用户中断")
        raise SystemExit(130)
    except Exception as error:
        print(f"启动失败: {error}", file=sys.stderr)
        raise SystemExit(1)
