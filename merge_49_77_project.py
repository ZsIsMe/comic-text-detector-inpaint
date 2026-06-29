#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


BASE = Path("/Users/zhongsheng/Documents/comic_data/血刃之花/combined")
SRC_49_77 = BASE / "49_77"
SRC_67_70 = BASE / "67_70"
DST = BASE / "49_77_merged"


def is_page_file(path: Path) -> bool:
    stem = path.stem
    if "_" not in stem:
        return False
    chapter = stem.split("_", 1)[0]
    return chapter.isdigit()


def sorted_page_keys(keys):
    def key(value: str):
        stem = Path(value).stem
        parts = stem.split("_", 1)
        try:
            return (int(parts[0]), int(parts[1]))
        except Exception:
            return (10_000, value)

    return sorted(keys, key=key)


def copytree_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        raise SystemExit(f"Destination already exists: {dst}")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".DS_Store"))


def copy_missing_67_70_files() -> list[Path]:
    copied: list[Path] = []
    skip = {
        Path("imgtrans_67_70.json"),
        Path("ctd_inpainted/solid_inpaint_report.json"),
    }
    for src in SRC_67_70.rglob("*"):
        if not src.is_file() or src.name == ".DS_Store":
            continue
        rel = src.relative_to(SRC_67_70)
        if rel in skip:
            continue
        dst = DST / rel
        if dst.exists():
            raise SystemExit(f"Unexpected collision while copying {rel}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)
    return copied


def copy_yolo_masks() -> list[Path]:
    copied: list[Path] = []
    dst_dir = DST / "ctd_inpainted" / "yolo_mask"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted((SRC_67_70 / "mask").glob("*.png")):
        if not is_page_file(src):
            continue
        dst = dst_dir / src.name
        if dst.exists():
            raise SystemExit(f"Unexpected yolo_mask collision: {dst}")
        shutil.copy2(src, dst)
        copied.append(dst.relative_to(DST))
    return copied


def merge_imgtrans() -> int:
    dst_json = DST / "imgtrans_49_77.json"
    src_json = SRC_67_70 / "imgtrans_67_70.json"
    dst_data = json.loads(dst_json.read_text(encoding="utf-8"))
    src_data = json.loads(src_json.read_text(encoding="utf-8"))

    dst_pages = dst_data.setdefault("pages", {})
    for key, value in src_data.get("pages", {}).items():
        if key in dst_pages:
            raise SystemExit(f"imgtrans page collision: {key}")
        dst_pages[key] = value

    ordered_pages = {key: dst_pages[key] for key in sorted_page_keys(dst_pages)}
    dst_data["directory"] = str(DST)
    dst_data["pages"] = ordered_pages
    dst_json.write_text(
        json.dumps(dst_data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return len(src_data.get("pages", {}))


def merge_solid_report() -> int:
    dst_json = DST / "ctd_inpainted" / "solid_inpaint_report.json"
    src_json = SRC_67_70 / "ctd_inpainted" / "solid_inpaint_report.json"
    dst_data = json.loads(dst_json.read_text(encoding="utf-8"))
    src_data = json.loads(src_json.read_text(encoding="utf-8"))

    dst_pages = dst_data.setdefault("pages", {})
    for key, value in src_data.get("pages", {}).items():
        if key in dst_pages:
            raise SystemExit(f"solid report page collision: {key}")
        dst_pages[key] = value

    dst_data["image_dir"] = str(DST)
    dst_data["output_dir"] = str(DST / "ctd_inpainted")
    dst_data["pages"] = {key: dst_pages[key] for key in sorted_page_keys(dst_pages)}

    summary = dict(dst_data.get("summary", {}))
    src_summary = src_data.get("summary", {})
    for field in ("total", "processed", "with_other_mask", "failed"):
        summary[field] = int(summary.get(field, 0)) + int(src_summary.get(field, 0))
    dst_data["summary"] = summary

    dst_json.write_text(
        json.dumps(dst_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return len(src_data.get("pages", {}))


def summarize() -> dict[str, int]:
    result: dict[str, int] = {}
    checks = {
        "root_jpg": DST,
        "result_png": DST / "result",
        "mask_png": DST / "mask",
        "ctd_inpainted_inpainted_png": DST / "ctd_inpainted" / "inpainted",
        "ctd_inpainted_mask_png": DST / "ctd_inpainted" / "mask",
        "ctd_inpainted_other_mask_png": DST / "ctd_inpainted" / "other_mask",
        "ctd_inpainted_yolo_mask_png": DST / "ctd_inpainted" / "yolo_mask",
    }
    for key, path in checks.items():
        if key == "root_jpg":
            files = list(path.glob("*.jpg"))
        else:
            files = list(path.glob("*.png"))
        result[key] = sum(1 for item in files if is_page_file(item))
    return result


def main() -> None:
    copytree_clean(SRC_49_77, DST)
    copied = copy_missing_67_70_files()
    yolo = copy_yolo_masks()
    imgtrans_pages = merge_imgtrans()
    report_pages = merge_solid_report()

    print(f"destination={DST}")
    print(f"copied_from_67_70={len(copied)}")
    print(f"copied_yolo_masks={len(yolo)}")
    print(f"merged_imgtrans_pages={imgtrans_pages}")
    print(f"merged_solid_report_pages={report_pages}")
    for key, value in summarize().items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
