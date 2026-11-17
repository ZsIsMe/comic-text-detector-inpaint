"""Bound MCP review client. One binding belongs to one GUI and one session.

This is cooperative isolation for agents using this script, not an OS lock on
MCP or the underlying project. Use separate GUI instances and folders for
concurrent review sessions.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import coordination


REPO = Path(__file__).resolve().parents[4]
INSTANCE_RE = re.compile(r"[0-9a-f]{32}\Z")
SESSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
DEFAULT_RUN_ROOT = Path(tempfile.gettempdir()) / "solid-inpaint-review-sessions"
REVIEW_TOOLS = frozenset({"get_app_status", "list_pages", "go_to_page",
                          "get_page_view", "get_review_bundle", "apply_page_edits",
                          "undo_page_edit", "mark_page_reviewed", "save_project"})
TOKEN_TOOLS = frozenset({"apply_page_edits", "undo_page_edit", "mark_page_reviewed"})


def canonical_folder(value: str) -> str:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError(f"Not a directory: {path}")
    return str(path)


def folder_key(folder: str) -> str:
    return hashlib.sha256(folder.encode("utf-8")).hexdigest()[:16]


def batches(indices: list[int], size: int = 20) -> list[list[int]]:
    if size < 2 or any(type(i) is not int or i < 1 for i in indices) or any(
        left >= right for left, right in zip(indices, indices[1:])
    ):
        raise ValueError("Batch indices must be strictly increasing positive integers; size at least 2")
    result = []
    offset = 0
    while len(indices) - offset >= 2 * size:
        result.append(indices[offset:offset + size])
        offset += size
    if offset < len(indices):
        result.append(indices[offset:])
    return result


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_binding(path: Path) -> dict:
    binding = json.loads(path.read_text(encoding="utf-8"))
    if not INSTANCE_RE.fullmatch(binding["instance_id"]):
        raise ValueError("Invalid binding instance_id")
    if type(binding["pid"]) is not int or binding["pid"] < 1:
        raise ValueError("Invalid binding PID")
    if not Path(binding["folder"]).is_absolute():
        raise ValueError("Invalid binding folder")
    return binding


def _metadata(result) -> tuple[dict, list]:
    metadata: dict = {}
    images = []
    for item in result.content:
        if item.type == "text":
            try:
                parsed = json.loads(item.text)
            except (TypeError, ValueError):
                metadata.setdefault("messages", []).append(item.text)
            else:
                if isinstance(parsed, dict):
                    metadata.update(parsed)
                else:
                    metadata.setdefault("messages", []).append(parsed)
        elif item.type == "image":
            images.append(item)
    return metadata, images


async def _tool(session, name: str, args: dict) -> tuple[dict, list]:
    result = await session.call_tool(name, args)
    metadata, images = _metadata(result)
    if result.isError:
        raise RuntimeError(f"{name}: {json.dumps(metadata, ensure_ascii=False)}")
    return metadata, images


def check_status(status: dict, binding: dict, *, folder: str | None = None) -> None:
    if status.get("instance_id") != binding["instance_id"] or status.get("pid") != binding["pid"]:
        raise RuntimeError("Bound GUI instance or PID changed; bind a new session")
    actual = status.get("folder")
    expected = folder if folder is not None else binding["folder"]
    if not actual or Path(actual).resolve() != Path(expected).resolve():
        raise RuntimeError(f"Bound GUI folder changed: expected {expected}, got {actual}")


async def _connect_and_run(operation):
    # Import after local validation so batches/help work without optional MCP.
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    python = REPO / ".venv" / "bin" / "python"
    server = REPO / "solid_inpaint_mcp.py"
    if not python.is_file() or not server.is_file():
        raise RuntimeError(f"MCP environment missing under {REPO}")
    async with stdio_client(StdioServerParameters(
        command=str(python), args=[str(server)], cwd=str(REPO)
    )) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            return await operation(session)


async def instances() -> dict:
    async def op(session):
        metadata, _ = await _tool(session, "list_app_instances", {})
        return metadata

    return await _connect_and_run(op)


async def bind(args) -> dict:
    if not SESSION_RE.fullmatch(args.session):
        raise ValueError("Session must use 1-64 letters, digits, underscore or hyphen")
    if not INSTANCE_RE.fullmatch(args.instance_id):
        raise ValueError("Instance ID must be 32 lowercase hex digits")
    if args.pid < 1:
        raise ValueError("PID must be positive")
    folder = canonical_folder(args.folder)
    binding = {"session": args.session, "instance_id": args.instance_id,
               "pid": args.pid, "folder": folder}
    root = Path(args.run_root).expanduser().resolve()
    binding_path = root / args.session / args.instance_id / "binding.json"
    binding_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    binding_path.parent.chmod(0o700)
    coord_root = getattr(args, 'coord_root', coordination.ROOT)

    async def op(session):
        status, _ = await _tool(session, "get_app_status", {"instance_id": args.instance_id})
        if status.get('instance_id') != args.instance_id or status.get('pid') != args.pid:
            raise RuntimeError('GUI instance or PID changed; inspect instances again')
        needs_open = not status.get('folder') or Path(status['folder']).resolve() != Path(folder)
        if needs_open and not getattr(args, 'open_project', False):
            raise RuntimeError('GUI has another folder or none; use bind --open-project explicitly')
        if binding_path.exists():
            existing = read_binding(binding_path)
            if any(existing.get(key) != binding[key] for key in ('session', 'instance_id', 'pid', 'folder')):
                raise RuntimeError("Binding already exists with another GUI or folder; use switch for folder changes")
        old_claim = coordination.read_claim(args.instance_id, coord_root)
        claim = coordination.create_claim(binding, binding_path, coord_root)
        binding['claim_token'] = claim['claim_token']
        try:
            if needs_open:
                try:
                    await _tool(session, 'open_project', {'instance_id': args.instance_id, 'path': folder})
                except Exception:
                    pass  # A timed-out response may still have changed the GUI.
                status, _ = await _tool(session, 'get_app_status', {'instance_id': args.instance_id})
                check_status(status, binding)
            atomic_json(binding_path, binding)
        except Exception:
            if old_claim != claim:
                coordination.claim_path(args.instance_id, coord_root).unlink(missing_ok=True)
            raise
        return {"binding": str(binding_path), "folder": folder,
                "output_dir": str(binding_path.parent / folder_key(folder))}

    with coordination.file_lock(coordination.instance_lock_path(args.instance_id, coord_root)):
        return await _connect_and_run(op)


async def switch(args) -> dict:
    path = Path(args.binding).expanduser().resolve(strict=True)
    new_folder = canonical_folder(args.folder)
    coord_root = getattr(args, 'coord_root', coordination.ROOT)

    async def op(session):
        binding = read_binding(path)
        coordination.verify_claim(binding, path, coord_root)
        status, _ = await _tool(session, "get_app_status", {"instance_id": binding["instance_id"]})
        check_status(status, binding)
        open_error = None
        try:
            await _tool(session, "open_project", {
                "instance_id": binding["instance_id"], "path": new_folder})
        except Exception as exc:
            open_error = exc
        # The GUI may complete an open after its response times out. Verify its
        # actual state even when the response itself looked successful.
        try:
            result, _ = await _tool(session, "get_app_status", {"instance_id": binding["instance_id"]})
        except Exception:
            async def query(reconnected):
                return await _tool(reconnected, 'get_app_status', {'instance_id': binding['instance_id']})
            try:
                result, _ = await _connect_and_run(query)
            except Exception as verify_error:
                raise RuntimeError(f'Open project state could not be verified: {open_error}') from verify_error
        check_status(result, binding, folder=new_folder)
        binding["folder"] = new_folder
        atomic_json(path, binding)
        return {"binding": str(path), "folder": new_folder,
                "output_dir": str(path.parent / folder_key(new_folder))}

    initial = read_binding(path)
    with coordination.file_lock(coordination.instance_lock_path(initial['instance_id'], coord_root)):
        return await _connect_and_run(op)


async def call(args) -> dict:
    path = Path(args.binding).expanduser().resolve(strict=True)
    coord_root = getattr(args, 'coord_root', coordination.ROOT)
    if args.tool not in REVIEW_TOOLS:
        raise ValueError("Tool is not part of the bound review workflow; use bind or switch for folder changes")
    params = json.loads(Path(args.params).read_text(encoding="utf-8")) if args.params else {}
    if not isinstance(params, dict):
        raise ValueError("Parameters file must contain a JSON object")
    if "instance_id" in params:
        raise ValueError("Parameters must not contain instance_id; binding supplies it")

    async def op(session):
        binding = read_binding(path)
        coordination.verify_claim(binding, path, coord_root)
        params['instance_id'] = binding['instance_id']
        status, _ = await _tool(session, "get_app_status", {"instance_id": binding["instance_id"]})
        check_status(status, binding)
        output = path.parent / folder_key(binding["folder"])
        if args.tool in TOKEN_TOOLS:
            latest = output / "latest-bundle.json"
            if not isinstance(params.get("page_token"), str) or not latest.is_file() or params["page_token"] != json.loads(
                latest.read_text(encoding="utf-8")).get("page_token"):
                raise ValueError("Page token is not the latest bundle for this bound GUI and folder")
        metadata, images = await _tool(session, args.tool, params)
        # A navigation or edit can return a new page. Recheck before any artifact write.
        status, _ = await _tool(session, "get_app_status", {"instance_id": binding["instance_id"]})
        check_status(status, binding)
        if "folder" in metadata and Path(metadata["folder"]).resolve() != Path(binding["folder"]).resolve():
            raise RuntimeError(f"MCP result belongs to another folder: {metadata['folder']} != {binding['folder']}")
        image_paths = []
        if images:
            order = metadata.get("image_order") or ["view"]
            if len(order) != len(images) or len(set(order)) != len(order) or any(
                not isinstance(kind, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", kind)
                for kind in order
            ):
                raise RuntimeError("Invalid image_order from MCP")
            decoded = [(kind, base64.b64decode(image.data, validate=True))
                       for kind, image in zip(order, images)]
            output.mkdir(parents=True, exist_ok=True)
            snapshot_name = uuid.uuid4().hex
            staging = output / f".{snapshot_name}.tmp"
            snapshot = output / snapshot_name
            try:
                staging.mkdir()
                for kind, data in decoded:
                    (staging / f"{kind}.png").write_bytes(data)
                os.replace(staging, snapshot)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            image_paths = [str(snapshot / f"{kind}.png") for kind, _ in decoded]
        if "page_token" in metadata:
            atomic_json(output / "latest-bundle.json", {**metadata, "image_paths": image_paths})
        elif args.tool in TOKEN_TOOLS or args.tool == "go_to_page":
            atomic_json(output / "latest-bundle.json", {"page_token": None})
        return {"isError": False, "metadata": metadata, "images": image_paths,
                "output_dir": str(output)}

    initial = read_binding(path)
    with coordination.file_lock(coordination.instance_lock_path(initial['instance_id'], coord_root)):
        return await _connect_and_run(op)


async def handoff(args) -> dict:
    path = Path(args.binding).expanduser().resolve(strict=True)
    coord_root = getattr(args, 'coord_root', coordination.ROOT)

    async def op(session):
        binding = read_binding(path)
        coordination.verify_claim(binding, path, coord_root)
        instance = {'instance_id': binding['instance_id']}
        status, _ = await _tool(session, 'get_app_status', instance)
        check_status(status, binding)
        saved, _ = await _tool(session, 'save_project', instance)
        pages_result, _ = await _tool(session, 'list_pages', instance)
        status, _ = await _tool(session, 'get_app_status', instance)
        check_status(status, binding)
        if not saved.get('saved') or saved.get('save_pending') or saved.get('save_error') or status.get('save_pending') or status.get('save_error'):
            raise RuntimeError('GUI did not confirm a completed save')
        if Path(pages_result.get('folder', '')).resolve() != Path(binding['folder']).resolve():
            raise RuntimeError('Page list belongs to another folder')
        pages = pages_result.get('pages', [])
        if len(pages) != pages_result.get('page_count'):
            raise RuntimeError('Page list count mismatch')
        start = args.start_page or 1
        end = args.end_page or len(pages)
        if not 1 <= start <= end <= len(pages):
            raise ValueError('Review range is outside the project page list')
        unreviewed = [i for i, page in enumerate(pages, 1) if page.get('reviewed') is not True]
        missing = [i for i in unreviewed if start <= i <= end]
        return {'folder': binding['folder'], 'instance_id': binding['instance_id'],
                'saved': True, 'page_count': len(pages), 'verified_range': [start, end],
                'verified': not missing, 'unreviewed_in_range': missing,
                'next_batch': batches(unreviewed)[0] if unreviewed else [],
                'next_unreviewed_index': unreviewed[0] if unreviewed else None}

    initial = read_binding(path)
    with coordination.file_lock(coordination.instance_lock_path(initial['instance_id'], coord_root)):
        return await _connect_and_run(op)


def release(args) -> dict:
    path = Path(args.binding).expanduser().resolve(strict=True)
    coord_root = getattr(args, 'coord_root', coordination.ROOT)
    initial = read_binding(path)
    with coordination.file_lock(coordination.instance_lock_path(initial['instance_id'], coord_root)):
        binding = read_binding(path)
        coordination.verify_claim(binding, path, coord_root)
        coordination.claim_path(binding['instance_id'], coord_root).unlink()
        path.unlink()
    return {'released': True, 'instance_id': initial['instance_id'], 'binding': str(path)}


async def reap(args) -> dict:
    coord_root = getattr(args, 'coord_root', coordination.ROOT)
    ident = args.instance_id
    with coordination.file_lock(coordination.instance_lock_path(ident, coord_root)):
        claim = coordination.read_claim(ident, coord_root)
        if not claim:
            return {'reaped': False, 'instance_id': ident}
        # A process killed between claim and binding writes leaves an orphan
        # even if its GUI is still alive. This check is under the instance lock.
        binding_file = Path(claim.get('binding', ''))
        try:
            existing = json.loads(binding_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            existing = None
        valid_binding = (isinstance(existing, dict) and binding_file.is_absolute()
                         and all(existing.get(key) == claim.get(key) for key in
                                 ('instance_id', 'pid', 'session', 'claim_token')))
        if valid_binding and coordination.process_alive(int(claim['pid'])):
            current = await instances()
            if any(item.get('instance_id') == ident for item in current.get('instances', [])):
                raise RuntimeError('GUI instance may still be active; release its claim explicitly')
        coordination.claim_path(ident, coord_root).unlink()
        return {'reaped': True, 'instance_id': ident, 'orphan': not valid_binding}


def _error_message(exc: Exception) -> str:
    """Keep nested TaskGroup/ExceptionGroup failures as one useful JSON error."""
    children = getattr(exc, 'exceptions', None)
    if isinstance(children, (tuple, list)) and children:
        parts = list(dict.fromkeys(_error_message(child) for child in children))
        return '; '.join(parts)
    return str(exc) or type(exc).__name__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("instances", help="List GUI IDs, PIDs and folders before binding")
    p = sub.add_parser("bind", help="Bind this session to an existing GUI instance and folder")
    p.add_argument("--session", required=True)
    p.add_argument("--instance-id", required=True)
    p.add_argument("--pid", required=True, type=int)
    p.add_argument("--folder", required=True)
    p.add_argument('--open-project', action='store_true', help='Open the target folder after claiming this GUI')
    p.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    p = sub.add_parser("switch", help="Explicitly open another folder in the bound GUI")
    p.add_argument("--binding", required=True)
    p.add_argument("--folder", required=True)
    p = sub.add_parser('release', help='Release a GUI claim and invalidate its binding')
    p.add_argument('--binding', required=True)
    p = sub.add_parser('reap', help='Remove an orphan claim or one whose GUI is confirmed gone')
    p.add_argument('--instance-id', required=True)
    p = sub.add_parser('handoff', help='Save and verify an actually reviewed page range')
    p.add_argument('--binding', required=True)
    p.add_argument('--start-page', type=int)
    p.add_argument('--end-page', type=int)
    p = sub.add_parser("call", help="Call one MCP tool with binding injected")
    p.add_argument("--binding", required=True)
    p.add_argument("tool")
    p.add_argument("params", nargs="?", help="JSON object file, as in the earlier MCP client")
    p = sub.add_parser("batches", help="Split page indices into review batches")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--pages", type=int, help="Pages 1 through N")
    group.add_argument("--indices", help="Comma-separated page indices in review order")
    p.add_argument("--size", type=int, default=20)
    args = parser.parse_args(argv)
    try:
        if args.command == "batches":
            if args.pages is not None and args.pages < 0:
                raise ValueError("Page count must be nonnegative")
            indices = list(range(1, args.pages + 1)) if args.pages is not None else [
                int(value) for value in args.indices.split(",")]
            output = {"batches": batches(indices, args.size)}
        elif args.command == "instances":
            output = asyncio.run(instances())
        elif args.command == 'release':
            output = release(args)
        elif args.command == 'reap':
            output = asyncio.run(reap(args))
        else:
            output = asyncio.run({"bind": bind, "switch": switch, "call": call,
                                  'handoff': handoff}[args.command](args))
        print(json.dumps(output, ensure_ascii=False))
        return 1 if args.command == 'handoff' and not output['verified'] else 0
    except Exception as exc:
        print(json.dumps({"error": _error_message(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
