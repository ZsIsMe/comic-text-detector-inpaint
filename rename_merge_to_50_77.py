#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path


BASE = Path("/Users/zhongsheng/Documents/comic_data/血刃之花/combined")
SRC = BASE / "49_77_merged"
DST = BASE / "50_77_merge"


def is_chapter_49(path: Path) -> bool:
    return path.name.startswith("49_")


def sort_page_key(value: str):
    stem = Path(value).stem
    try:
        chapter, page = stem.split("_", 1)
        return int(chapter), int(page)
    except Exception:
        return 10_000, value


def rename_folder() -> None:
    if not SRC.exists():
        raise SystemExit(f"Missing source folder: {SRC}")
    if DST.exists():
        raise SystemExit(f"Destination already exists: {DST}")
    SRC.rename(DST)


def remove_chapter_49_files() -> int:
    removed = 0
    for path in sorted(DST.rglob("49_*")):
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def rewrite_imgtrans() -> tuple[int, int]:
    old_path = DST / "imgtrans_49_77.json"
    new_path = DST / "imgtrans_50_77.json"
    data = json.loads(old_path.read_text(encoding="utf-8"))
    pages = data.get("pages", {})
    before = len(pages)
    pages = {key: value for key, value in pages.items() if not key.startswith("49_")}
    data["directory"] = str(DST)
    data["pages"] = {key: pages[key] for key in sorted(pages, key=sort_page_key)}
    new_path.write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    old_path.unlink()
    return before, len(pages)


def rewrite_solid_report() -> tuple[int, int]:
    path = DST / "ctd_inpainted" / "solid_inpaint_report.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    all_pages = data.get("pages", {})
    removed_with_other_mask = sum(
        1
        for key, value in all_pages.items()
        if key.startswith("49_")
        and isinstance(value, dict)
        and int(value.get("other_pixels", 0)) > 0
    )
    pages = all_pages
    before = len(pages)
    pages = {key: value for key, value in pages.items() if not key.startswith("49_")}
    data["image_dir"] = str(DST)
    data["output_dir"] = str(DST / "ctd_inpainted")
    data["pages"] = {key: pages[key] for key in sorted(pages, key=sort_page_key)}

    summary = dict(data.get("summary", {}))
    removed_pages = before - len(pages)
    for key in ("total", "processed"):
        summary[key] = int(summary.get(key, 0)) - removed_pages
    summary["with_other_mask"] = (
        int(summary.get("with_other_mask", 0)) - removed_with_other_mask
    )
    data["summary"] = summary

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return before, len(pages)


def main() -> None:
    rename_folder()
    removed = remove_chapter_49_files()
    img_before, img_after = rewrite_imgtrans()
    report_before, report_after = rewrite_solid_report()

    print(f"destination={DST}")
    print(f"removed_49_files={removed}")
    print(f"imgtrans_pages={img_before}->{img_after}")
    print(f"solid_report_pages={report_before}->{report_after}")


if __name__ == "__main__":
    main()
