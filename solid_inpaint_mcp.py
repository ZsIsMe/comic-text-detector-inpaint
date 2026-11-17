"""MCP stdio bridge to an already running Solid Inpaint GUI.

Install the optional requirements-mcp.txt and configure this file as a local
stdio server in the MCP client.  All authoritative operations run in the GUI.
"""

from __future__ import annotations

import json
import sys

try:
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ImageContent, TextContent
except ModuleNotFoundError as exc:
    if exc.name != 'mcp':
        raise
    sys.stderr.write('缺少選用 MCP 依賴；請先執行 python -m pip install -r requirements-mcp.txt\n')
    raise SystemExit(2) from exc

from solid_inpaint_control import ControlError, list_app_instances as discover_app_instances, request_control
from PySide6.QtCore import QCoreApplication


mcp = FastMCP('Solid Inpaint')


def _call(method: str, *, instance_id: str | None = None, **params):
    try:
        if instance_id is None:
            return request_control(method, params)
        return request_control(method, params, instance_id=instance_id)
    except ControlError as exc:
        raise ValueError(f'{exc.code}: {exc}') from exc


@mcp.tool()
def list_app_instances() -> dict:
    """Find GUI windows, their instance IDs, folders and pages.

    When more than one window runs, pass its instance_id to every other tool.
    """
    return discover_app_instances()


@mcp.tool()
def get_app_status(instance_id: str | None = None) -> dict:
    """Read the running GUI's open folder, current page, busy state and save state."""
    return _call('get_app_status', instance_id=instance_id)


@mcp.tool()
def open_project(path: str, instance_id: str | None = None) -> dict:
    """Open an absolute image folder in the running GUI without a folder picker."""
    return _call('open_project', instance_id=instance_id, path=path)


@mcp.tool()
def get_detection_settings(instance_id: str | None = None) -> dict:
    """Read globally saved model, device, mask and bubble defaults."""
    return _call('get_detection_settings', instance_id=instance_id)


@mcp.tool()
def start_detection(
    start_page: int = 1,
    end_page: int | None = None,
    detector: str | None = None,
    detector_params: dict | None = None,
    bubble: dict | None = None,
    allow_redetect: bool = False,
    instance_id: str | None = None,
) -> dict:
    """Start a background detection job; return a job id for get_job_status.

    Defaults come from the same global preferences as the GUI. Existing masks
    require allow_redetect=True. Human edits are preserved by the GUI workflow.
    """
    params = {'start_page': start_page, 'allow_redetect': allow_redetect}
    if end_page is not None:
        params['end_page'] = end_page
    if detector is not None:
        params['detector'] = detector
    if detector_params is not None:
        params['detector_params'] = detector_params
    if bubble is not None:
        params['bubble'] = bubble
    return _call('start_detection', instance_id=instance_id, **params)


@mcp.tool()
def get_job_status(job_id: str | None = None, instance_id: str | None = None) -> dict:
    """Read detection progress, failed pages and final job result."""
    return _call('get_job_status', instance_id=instance_id, job_id=job_id)


@mcp.tool()
def list_pages(instance_id: str | None = None) -> dict:
    """List every page in GUI order with detection, OTHER and review status."""
    return _call('list_pages', instance_id=instance_id)


@mcp.tool()
def go_to_page(index: int | None = None, name: str | None = None,
               instance_id: str | None = None) -> dict:
    """Go to a one-based page index or exact filename; flush edits first."""
    return _call('go_to_page', instance_id=instance_id, index=index, name=name)


@mcp.tool()
def get_page_view(kind: str = 'composite', max_side: int = 1600,
                  crop: list[int] | None = None,
                  instance_id: str | None = None) -> list[TextContent | ImageContent]:
    """Return the visible page as PNG and its original dimensions/revision.

    kind is original, composite (solid fill plus pink OTHER) or mask (both
    selection classes). max_side controls a bounded preview image size.
    """
    page = _call('get_page_view', instance_id=instance_id,
                 kind=kind, max_side=max_side, crop=crop)
    png = page.pop('image_png_base64')
    mime = page.pop('mime_type')
    return [TextContent(type='text', text=json.dumps(page, ensure_ascii=False)),
            ImageContent(type='image', data=png, mimeType=mime)]


def _bundle_contents(bundle: dict) -> list[TextContent | ImageContent]:
    images = bundle.pop('images')
    order = list(images)
    bundle['image_order'] = order
    image_blocks = []
    bundle['image_metadata'] = {}
    for kind in order:
        entry = images[kind]
        bundle['image_metadata'][kind] = {key: value for key, value in entry.items()
                                          if key not in ('image_png_base64', 'mime_type')}
        image_blocks.append(ImageContent(type='image', data=entry['image_png_base64'],
                                         mimeType=entry['mime_type']))
    return [TextContent(type='text', text=json.dumps(bundle, ensure_ascii=False)), *image_blocks]


@mcp.tool()
def get_review_bundle(max_side: int = 1600, crop: list[int] | None = None,
                      include_mask: bool = False,
                      instance_id: str | None = None) -> list[TextContent | ImageContent]:
    """Get one saved page snapshot: original and composite PNGs, optional mask.

    crop uses original-image pixels [x,y,width,height]. The opaque page_token
    is required for edits or review marks and expires when this page changes.
    """
    return _bundle_contents(_call('get_review_bundle', instance_id=instance_id,
                                  max_side=max_side, crop=crop,
                                  include_mask=include_mask))


@mcp.tool()
def apply_page_edits(page_token: str, operations: list[dict], max_side: int = 1600,
                     crop: list[int] | None = None,
                     include_mask: bool = False,
                     instance_id: str | None = None) -> list[TextContent | ImageContent]:
    """Atomically apply one undoable batch on the current saved page.

    Each operation has kind restore/add/transfer and shape
    {"rect":[x,y,width,height]} or {"polygon":[[x,y],...]}, all in original
    pixels. Add/transfer require target solid/other. Operations targeting solid
    require explicit color_rgb [R,G,B]. Transfer moves only selected pixels
    already in the opposite class. Restore clears both classes.
    """
    return _bundle_contents(_call('apply_page_edits', instance_id=instance_id,
                                  page_token=page_token,
                                  operations=operations, max_side=max_side, crop=crop,
                                  include_mask=include_mask))


@mcp.tool()
def undo_page_edit(page_token: str, max_side: int = 1600,
                   crop: list[int] | None = None,
                   include_mask: bool = False,
                   instance_id: str | None = None) -> list[TextContent | ImageContent]:
    """Undo only the latest MCP batch when its page token is still current."""
    return _bundle_contents(_call('undo_page_edit', instance_id=instance_id,
                                  page_token=page_token,
                                  max_side=max_side, crop=crop,
                                  include_mask=include_mask))


@mcp.tool()
def mark_page_reviewed(page_token: str, reviewed: bool = True,
                       advance: bool = False, max_side: int = 1600,
                       crop: list[int] | None = None,
                       include_mask: bool = False,
                       instance_id: str | None = None) -> list[TextContent | ImageContent]:
    """Save the human/AI review decision; optionally advance and get next bundle.

    Call after visual confirmation. Editing alone never marks a page reviewed.
    """
    return _bundle_contents(_call('mark_page_reviewed', instance_id=instance_id,
                                  page_token=page_token,
                                  reviewed=reviewed, advance=advance,
                                  max_side=max_side, crop=crop,
                                  include_mask=include_mask))


@mcp.tool()
def save_project(instance_id: str | None = None) -> dict:
    """Flush the currently edited page and report any save error."""
    return _call('save_project', instance_id=instance_id)


if __name__ == '__main__':
    # Keep Qt network's process application alive when the SDK executes tool
    # functions on AnyIO worker threads (needed by QLocalSocket on Windows).
    qt_app = QCoreApplication.instance() or QCoreApplication([])
    mcp.run(transport='stdio')
