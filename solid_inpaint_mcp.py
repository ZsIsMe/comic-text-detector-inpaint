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

from solid_inpaint_control import ControlError, request_control
from PySide6.QtCore import QCoreApplication


mcp = FastMCP('Solid Inpaint')


def _call(method: str, **params):
    try:
        return request_control(method, params)
    except ControlError as exc:
        raise ValueError(f'{exc.code}: {exc}') from exc


@mcp.tool()
def get_app_status() -> dict:
    """Read the running GUI's open folder, current page, busy state and save state."""
    return _call('get_app_status')


@mcp.tool()
def open_project(path: str) -> dict:
    """Open an absolute image folder in the running GUI without a folder picker."""
    return _call('open_project', path=path)


@mcp.tool()
def get_detection_settings() -> dict:
    """Read globally saved model, device, mask and bubble defaults."""
    return _call('get_detection_settings')


@mcp.tool()
def start_detection(
    start_page: int = 1,
    end_page: int | None = None,
    detector: str | None = None,
    detector_params: dict | None = None,
    bubble: dict | None = None,
    allow_redetect: bool = False,
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
    return _call('start_detection', **params)


@mcp.tool()
def get_job_status(job_id: str | None = None) -> dict:
    """Read detection progress, failed pages and final job result."""
    return _call('get_job_status', job_id=job_id)


@mcp.tool()
def list_pages() -> dict:
    """List every page in GUI order with detection, OTHER and review status."""
    return _call('list_pages')


@mcp.tool()
def go_to_page(index: int | None = None, name: str | None = None) -> dict:
    """Go to a one-based page index or exact filename; flush edits first."""
    return _call('go_to_page', index=index, name=name)


@mcp.tool()
def get_page_view(kind: str = 'composite', max_side: int = 1600,
                  crop: list[int] | None = None) -> list[TextContent | ImageContent]:
    """Return the visible page as PNG and its original dimensions/revision.

    kind is original, composite (solid fill plus pink OTHER) or mask (both
    selection classes). max_side controls a bounded preview image size.
    """
    page = _call('get_page_view', kind=kind, max_side=max_side, crop=crop)
    png = page.pop('image_png_base64')
    mime = page.pop('mime_type')
    return [TextContent(type='text', text=json.dumps(page, ensure_ascii=False)),
            ImageContent(type='image', data=png, mimeType=mime)]


@mcp.tool()
def save_project() -> dict:
    """Flush the currently edited page and report any save error."""
    return _call('save_project')


if __name__ == '__main__':
    # Keep Qt network's process application alive when the SDK executes tool
    # functions on AnyIO worker threads (needed by QLocalSocket on Windows).
    qt_app = QCoreApplication.instance() or QCoreApplication([])
    mcp.run(transport='stdio')
