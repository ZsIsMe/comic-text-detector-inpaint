"""Local control of the running Solid Inpaint window.

The Qt server lives in the GUI thread.  The stdio MCP process only forwards
bounded JSON requests to this socket; it never reads or writes project files.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import socket as py_socket
import stat
import sys
import time
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QObject
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication

if os.name == 'posix':
    import fcntl


MAX_MESSAGE = 64 * 1024 * 1024
MAX_VIEW_SIDE = 2048
MAX_BATCH_OPERATIONS = 64


def server_name() -> str:
    root = str(Path(__file__).resolve().parent)
    suffix = hashlib.sha256(root.encode('utf-8')).hexdigest()[:12]
    user = os.getuid() if hasattr(os, 'getuid') else os.environ.get('USERNAME', 'local')
    name = f'solid-inpaint-{user}-{suffix}'
    # The MCP stdio child can have a different TMPDIR from the GUI process.
    # An absolute Unix socket path keeps both ends on the same endpoint.
    return str(Path('/tmp') / name) if os.name == 'posix' else name


class ControlError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _remove_stale_unix_socket(name: str) -> bool:
    """Unlink only an abandoned socket owned by this user, never an active one."""
    path = Path(name)
    try:
        before = path.stat()
    except OSError:
        return False
    if not stat.S_ISSOCK(before.st_mode) or before.st_uid != os.getuid():
        return False
    try:
        with py_socket.socket(py_socket.AF_UNIX, py_socket.SOCK_STREAM) as probe:
            probe.settimeout(.25)
            probe.connect(name)
        return False
    except ConnectionRefusedError:
        # Recheck identity because another process may have rebound this path.
        try:
            after = path.stat()
            if (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino):
                return False
            path.unlink()
            return True
        except OSError:
            return False
    except OSError:
        # Permission errors or timeouts cannot prove that the listener is dead.
        return False


class ControlServer(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.name = server_name()
        self._lock_fd = None
        if os.name == 'posix':
            lock_name = self.name + '.lock'
            flags = os.O_CREAT | os.O_RDWR | getattr(os, 'O_NOFOLLOW', 0)
            fd = os.open(lock_name, flags, 0o600)
            try:
                owner = os.fstat(fd)
                if owner.st_uid != os.getuid() or not stat.S_ISREG(owner.st_mode):
                    raise RuntimeError('本機控制鎖檔權限不正確。')
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeError('已有 Solid Inpaint 視窗持有本機控制入口。') from exc
                self._lock_fd = fd
            except Exception:
                os.close(fd)
                raise
        try:
            # Qt with UserAccessOption can bind over another Unix socket on
            # macOS. The lock protects peers using this implementation; the
            # active-socket probe also protects an older running GUI.
            if os.name == 'posix' and Path(self.name).exists():
                if not _remove_stale_unix_socket(self.name):
                    raise RuntimeError('已有 Solid Inpaint 控制端正在執行。')
            self.server = QLocalServer(self)
            self.server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
            if not self.server.listen(self.name):
                raise RuntimeError(f'無法啟動本地控制端：{self.server.errorString()}')
        except Exception:
            self._release_lock()
            raise
        self.server.newConnection.connect(self._accept)
        self._buffers: dict[QLocalSocket, bytearray] = {}
        self.jobs: dict[str, dict] = {}
        self.latest_job_id = ''
        self._review_tokens: dict[str, tuple] = {}
        self._last_batch_stamp: tuple | None = None
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for socket in list(self._buffers):
            socket.abort()
        self._buffers.clear()
        self.server.close()
        QLocalServer.removeServer(self.name)
        self._release_lock()

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None

    def _accept(self) -> None:
        while self.server.hasPendingConnections():
            socket = self.server.nextPendingConnection()
            self._buffers[socket] = bytearray()
            socket.readyRead.connect(lambda s=socket: self._read(s))
            socket.disconnected.connect(lambda s=socket: self._buffers.pop(s, None))
            socket.disconnected.connect(socket.deleteLater)

    def _read(self, socket: QLocalSocket) -> None:
        data = self._buffers.get(socket)
        if data is None:
            return
        data.extend(bytes(socket.readAll()))
        if len(data) > MAX_MESSAGE:
            self._respond(socket, None, error=ControlError('too_large', '請求超過大小限制。'))
            return
        if b'\n' not in data:
            return
        line, _, remainder = data.partition(b'\n')
        if remainder:
            self._respond(socket, None, error=ControlError('bad_request', '每個連線只接受一個請求。'))
            return
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ControlError('bad_request', '請求必須是 JSON object。')
            request_id = request.get('id')
            method = request.get('method')
            params = request.get('params', {})
            if not isinstance(method, str) or not isinstance(params, dict):
                raise ControlError('bad_request', 'method / params 格式錯誤。')
            result = self.dispatch(method, params)
            self._respond(socket, request_id, result=result)
        except ControlError as exc:
            self._respond(socket, request_id if 'request_id' in locals() else None, error=exc)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._respond(socket, request_id if 'request_id' in locals() else None,
                          error=ControlError('failed', str(exc)))
        except Exception as exc:
            self._respond(socket, request_id if 'request_id' in locals() else None,
                          error=ControlError('internal_error', str(exc)))

    def _respond(self, socket, request_id, *, result=None, error=None):
        payload = {'id': request_id, 'ok': error is None}
        if error is None:
            payload['result'] = result
        else:
            payload['error'] = {'code': error.code, 'message': str(error)}
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8') + b'\n'
        if len(encoded) > MAX_MESSAGE:
            encoded = json.dumps({'id': request_id, 'ok': False,
                                  'error': {'code': 'too_large', 'message': '回應超過大小限制。'}}).encode() + b'\n'
        socket.write(encoded)
        socket.flush()
        socket.disconnectFromServer()

    def _ensure_idle(self):
        w = self.window
        if QApplication.activeModalWidget() is not None:
            raise ControlError('busy', '有尚未關閉的對話框，請先完成目前操作。')
        if w.is_local_edit_active or w.is_mask_stroke_active or w.ctd_selection_worker_thread is not None:
            raise ControlError('busy', '正在編輯選區，請完成目前操作後重試。')
        if w.worker_thread is not None:
            raise ControlError('busy', '批次任務正在執行。')
        if w.page_worker_thread is not None or w.render_timer.isActive():
            raise ControlError('busy', '當前頁預覽正在生成。')

    def _flush(self):
        if not self.window.save_all_edit_masks(silent=True):
            raise ControlError('save_failed', self.window._last_control_error or '儲存未完成。')

    def _status(self):
        w = self.window
        index = w.list_widget.currentRow() + 1 if w.current_img_path else 0
        return {'folder': w.folder or None, 'current_page': Path(w.current_img_path).name if w.current_img_path else None,
                'page_index': index, 'page_count': len(w.imglist),
                'page_loaded': bool(w.current_img_path and w.current_base is not None and w.current_page is not None),
                'job_id': self.latest_job_id or None,
                'job_running': w.worker_thread is not None,
                'save_pending': bool((w._pending_edit_future is not None and not w._pending_edit_future.done())
                                     or w._edit_revision != w._saved_edit_revision
                                     or w._last_edit_error),
                'save_error': w._last_edit_error,
                'edit_active': bool(w.is_local_edit_active or w.is_mask_stroke_active)}

    def _settings(self):
        # Use the same widget parsing/validation as the dialog without opening it.
        widget_class = getattr(sys.modules[type(self.window).__module__], 'DetectorSelectorWidget')
        widget = widget_class(self.window.settings)
        try:
            detector = widget.selected_detector()
            params = widget.selected_detector_params()
        finally:
            widget.deleteLater()
        settings = self.window.settings
        try:
            shrink = float(settings.value('bubble/shrink_percent', 2.0))
        except (TypeError, ValueError):
            shrink = 2.0
        if not math.isfinite(shrink) or not 0 <= shrink <= 10:
            shrink = 2.0
        bubble = {'enabled': settings.value('bubble/enabled', True, type=bool),
                  'direct_white': settings.value('bubble/direct_white', True, type=bool),
                  'shrink_percent': shrink}
        return {'detector': detector, 'detector_params': params, 'bubble': bubble,
                'page_range': [1, len(self.window.imglist)]}

    def dispatch(self, method: str, params: dict):
        w = self.window
        if method == 'get_app_status':
            return self._status()
        if method == 'get_detection_settings':
            return self._settings()
        if method == 'get_job_status':
            job_id = params.get('job_id') or self.latest_job_id
            if job_id not in self.jobs:
                raise ControlError('not_found', '找不到這次偵測任務。')
            return dict(self.jobs[job_id])
        if method == 'open_project':
            self._ensure_idle()
            folder = params.get('path')
            if not isinstance(folder, str) or not Path(folder).is_absolute() or not Path(folder).is_dir():
                raise ControlError('invalid_path', '請提供存在的絕對資料夾路徑。')
            if not w.load_folder(folder, silent=True):
                raise ControlError('open_failed', w._last_control_error or '無法開啟專案。')
            return self._status()
        if method == 'list_pages':
            if not w.folder:
                raise ControlError('no_project', '尚未開啟專案。')
            from project_store import is_page_reviewed
            pages = w.report.get('pages', {})
            result = []
            for i, path in enumerate(w.imglist, 1):
                name = Path(path).name
                info = pages.get(name, {})
                result.append({'index': i, 'name': name, 'processed': bool(info.get('processed')),
                               'other_pixels': int(info.get('other_pixels', 0)),
                               'reviewed': is_page_reviewed(info), 'failed': 'error' in info,
                               'error': info.get('error')})
            return {'folder': w.folder, 'page_count': len(result), 'pages': result,
                    'processed_count': sum(page['processed'] and not page['failed'] for page in result),
                    'failed_pages': [page['name'] for page in result if page['failed']],
                    'unreviewed_pages': [page['name'] for page in result if not page['reviewed']]}
        if method == 'go_to_page':
            self._ensure_idle()
            if not w.folder:
                raise ControlError('no_project', '尚未開啟專案。')
            index = params.get('index')
            name = params.get('name')
            if (index is None) == (name is None):
                raise ControlError('bad_request', '只可指定 index 或 name 其中一項。')
            if name is not None:
                matches = [i for i, path in enumerate(w.imglist) if Path(path).name == name]
                if not matches:
                    raise ControlError('not_found', '頁面檔名不存在。')
                row = matches[0]
            elif type(index) is int and 1 <= index <= len(w.imglist):
                row = index - 1
            else:
                raise ControlError('bad_request', '頁碼超出範圍。')
            self._flush()
            w.list_widget.setCurrentRow(row)
            if w.current_img_path != w.imglist[row]:
                raise ControlError('navigation_failed', '切頁失敗，請檢查保存狀態。')
            if w.current_base is None or w.current_page is None:
                raise ControlError('page_unavailable', '頁面未能載入，請檢查原圖及正式頁面資料。')
            return self._status()
        if method == 'save_project':
            self._ensure_idle()
            if not w.folder:
                raise ControlError('no_project', '尚未開啟專案。')
            if w.current_img_path and (w.current_base is None or w.current_page is None):
                raise ControlError('page_unavailable', '目前頁面未能載入，無法確認保存。')
            self._flush()
            return {'saved': True, **self._status()}
        if method == 'start_detection':
            self._ensure_idle()
            return self._start_detection(params)
        if method == 'get_page_view':
            return self._page_view(params)
        if method == 'get_review_bundle':
            return self._review_bundle(params)
        if method == 'apply_page_edits':
            return self._apply_page_edits(params)
        if method == 'undo_page_edit':
            return self._undo_page_edit(params)
        if method == 'mark_page_reviewed':
            return self._mark_page_reviewed(params)
        raise ControlError('unknown_method', f'未知方法：{method}')

    def _start_detection(self, params):
        from detect_solid_inpaint_folder import DETECTOR_LABELS
        w = self.window
        if not w.folder or not w.imglist:
            raise ControlError('no_project', '目前專案沒有圖片。')
        defaults = self._settings()
        detector = params.get('detector') or defaults['detector']
        if detector not in DETECTOR_LABELS:
            raise ControlError('bad_request', '不支援的文字偵測模型。')
        if detector == defaults['detector']:
            detector_params = dict(defaults['detector_params'])
        else:
            widget_class = getattr(sys.modules[type(w).__module__], 'DetectorSelectorWidget')
            widget = widget_class(w.settings)
            try:
                widget.buttons[detector].setChecked(True)
                detector_params = widget.selected_detector_params()
            finally:
                widget.deleteLater()
        overrides = params.get('detector_params')
        if overrides is not None and not isinstance(overrides, dict):
            raise ControlError('bad_request', 'detector_params 必須是 object。')
        detector_params.update(overrides or {})
        self._validate_detector_params(detector, detector_params)
        start = params.get('start_page', 1)
        end = params.get('end_page', len(w.imglist))
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(w.imglist):
            raise ControlError('bad_request', '偵測頁面範圍無效。')
        bubble = params.get('bubble') if params.get('bubble') is not None else defaults['bubble']
        if not isinstance(bubble, dict):
            raise ControlError('bad_request', 'bubble 必須是 object。')
        if set(bubble) - {'enabled', 'direct_white', 'shrink_percent'}:
            raise ControlError('bad_request', 'bubble 包含未知設定。')
        bubble = {**defaults['bubble'], **bubble}
        if (type(bubble['enabled']) is not bool or type(bubble['direct_white']) is not bool
                or type(bubble['shrink_percent']) not in (int, float)
                or not math.isfinite(bubble['shrink_percent'])
                or not 0 <= bubble['shrink_percent'] <= 10):
            raise ControlError('bad_request', '氣泡填充設定無效。')
        if type(params.get('allow_redetect', False)) is not bool:
            raise ControlError('bad_request', 'allow_redetect 必須是布林值。')
        images = w.imglist[start - 1:end]
        existing = [Path(path).name for path in images if w.has_existing_masks([path])]
        if existing and params.get('allow_redetect') is not True:
            raise ControlError('confirmation_required',
                               f'{len(existing)} 頁已有選區；若確定重偵測，請傳 allow_redetect=true。')
        self._flush()
        job_id = uuid.uuid4().hex
        job = {'job_id': job_id, 'state': 'running', 'current': 0, 'total': len(images),
               'name': None, 'error': None, 'folder': w.folder,
               'start_page': start, 'end_page': end, 'detector': detector,
               'image_names': [Path(path).name for path in images],
               'project_page_count': len(w.imglist), 'selected_page_count': len(images),
               'coverage_missing_pages': [Path(path).name for path in
                                          (w.imglist[:start - 1] + w.imglist[end:])]}
        self.jobs[job_id] = job
        self.latest_job_id = job_id
        def attach(worker):
            worker.progress.connect(lambda current, total, name, jid=job_id: self._job_progress(jid, current, total, name))
            worker.finished.connect(lambda report, jid=job_id: self._job_finished(jid, report))
            worker.failed.connect(lambda message, jid=job_id: self._job_failed(jid, message))
        if not w.start_worker('detect', images, detector_name=detector,
                              detector_params=detector_params, solid_settings=bubble,
                              silent=True, on_worker_ready=attach):
            self.jobs.pop(job_id, None)
            self.latest_job_id = ''
            raise ControlError('start_failed', w._last_control_error or '偵測未啟動。')
        if len(self.jobs) > 20:
            for old_id in list(self.jobs)[:-20]:
                self.jobs.pop(old_id, None)
        # Mirror the successful dialog choice as global defaults.
        w.settings.setValue('detector/last_name', detector)
        for key, value in detector_params.items():
            w.settings.setValue(f'detector/{detector}/{key}', value)
        for key, value in bubble.items():
            w.settings.setValue(f'bubble/{key}', value)
        w.settings.sync()
        return dict(job)

    @staticmethod
    def _validate_detector_params(detector, values):
        from ysg_detector import YSG_LABEL_DESCRIPTIONS
        schemas = {
            'legacy_ctd': {},
            'ctbd': {'inpaint_mask_dilate': ('int', 0, 64),
                     'mask_unification_method': {'none', 'rectangle', 'hull'},
                     'text_region_filter': {'all', 'text_bubble', 'text_free'}},
            'rfdetr': {'device': {'auto', 'cuda', 'mps', 'cpu'},
                       'inpaint_mask_dilate': ('int', 0, 64),
                       'mask_mode': {'text_onomatopoeia', 'text', 'onomatopoeia', 'all'}},
            'ysgyolo': {'device': {'auto', 'cuda', 'mps', 'cpu'},
                        'labels': ('labels', set(YSG_LABEL_DESCRIPTIONS)),
                        'merge_text_lines': ('bool',),
                        'source_text_vertical': ('bool',),
                        'mask_dilate_size': ('int', 0, 64),
                        'mask_corner_radius': ('int', 0, 1000)},
        }
        schema = schemas[detector]
        for name, value in values.items():
            rule = schema.get(name)
            if rule is None:
                raise ControlError('bad_request', f'{detector} 不支援參數 {name}。')
            if isinstance(rule, set):
                valid = value in rule if isinstance(value, str) else False
            elif rule[0] == 'int':
                valid = type(value) is int and rule[1] <= value <= rule[2]
            elif rule[0] == 'bool':
                valid = type(value) is bool
            else:
                valid = (isinstance(value, list)
                         and all(isinstance(label, str) and label in rule[1] for label in value)
                         and len(value) == len(set(value)))
            if not valid:
                raise ControlError('bad_request', f'{detector} 參數 {name} 無效。')

    def _job_progress(self, job_id, current, total, name):
        if job_id in self.jobs:
            self.jobs[job_id].update(current=current, total=total, name=name)

    def _job_finished(self, job_id, report):
        if job_id in self.jobs:
            pages = report.get('pages', {})
            failed = [name for name in self.jobs[job_id]['image_names']
                      if 'error' in pages.get(name, {})]
            self.jobs[job_id].update(state='completed', current=self.jobs[job_id]['total'],
                                     failed_pages=failed, error=None)

    def _job_failed(self, job_id, message):
        if job_id in self.jobs:
            self.jobs[job_id].update(state='failed', error=message)

    def _page_stamp(self) -> tuple:
        """Bind a review decision to the loaded page, saved data and source file."""
        from project_store import page_path, page_state_digest, source_signature
        w = self.window
        if not w.folder or not w.current_img_path or w.current_base is None or w.current_page is None:
            raise ControlError('no_page', '目前沒有可校對的頁面。')
        try:
            source = source_signature(w.current_img_path)
            archive = page_path(w.paths, w.current_img_path)
            archive_stat = archive.stat() if archive.exists() else None
        except OSError as exc:
            raise ControlError('page_unavailable', str(exc)) from exc
        loaded_source = getattr(w, '_control_loaded_source_signature', source)
        if loaded_source != source:
            raise ControlError('stale_page', '原圖載入後已變更；請重新載入頁面並核對。')
        info = w.report.get('pages', {}).get(Path(w.current_img_path).name, {})
        archive_signature = (archive_stat.st_size, archive_stat.st_mtime_ns) if archive_stat else None
        if getattr(w, '_control_loaded_archive_signature', archive_signature) != archive_signature:
            raise ControlError('stale_page', '正式頁面資料載入後已變更；請重新載入頁面並核對。')
        if info.get('has_data') and archive_signature is None:
            raise ControlError('page_unavailable', '正式頁面資料遺失；請重新載入頁面並核對。')
        return (str(Path(w.folder).resolve()), str(Path(w.current_img_path).resolve()),
                w.list_widget.currentRow(), getattr(w, '_control_page_generation', 0),
                id(w.current_base), id(w.current_page), w._edit_revision, w._saved_edit_revision,
                source, archive_signature,
                page_state_digest(w.current_page, source), info.get('state_digest'),
                info.get('reviewed_digest'))

    def _issue_page_token(self) -> str:
        token = uuid.uuid4().hex
        self._review_tokens[token] = self._page_stamp()
        while len(self._review_tokens) > 32:
            self._review_tokens.pop(next(iter(self._review_tokens)))
        return token

    def _validate_page_token(self, token):
        if not isinstance(token, str) or token not in self._review_tokens:
            raise ControlError('stale_page', '頁面憑證無效；請重新讀取校對影像。')
        if self._review_tokens[token] != self._page_stamp():
            raise ControlError('stale_page', '頁面或原圖已變更；請重新讀取校對影像。')

    def _review_bundle(self, params):
        self._ensure_idle()
        self._page_stamp()
        self._flush()
        before = self._page_stamp()
        include_mask = params.get('include_mask', False)
        if type(include_mask) is not bool:
            raise ControlError('bad_request', 'include_mask 必須是布林值。')
        view_params = {key: params[key] for key in ('max_side', 'crop') if key in params}
        images = {kind: self._page_view({**view_params, 'kind': kind})
                  for kind in (('original', 'composite', 'mask') if include_mask
                               else ('original', 'composite'))}
        if before != self._page_stamp():
            raise ControlError('stale_page', '讀取影像時頁面或原圖已變更。')
        first = images['original']
        status = self._status()
        from project_store import is_page_reviewed
        info = self.window.report.get('pages', {}).get(first['name'], {})
        return {'folder': status['folder'], 'name': first['name'],
                'page_index': first['page_index'], 'page_count': status['page_count'],
                'width': first['width'], 'height': first['height'],
                'revision': first['revision'], 'origin': first['origin'],
                'crop_width': first['crop_width'], 'crop_height': first['crop_height'],
                'image_width': first['image_width'], 'image_height': first['image_height'],
                'scale': first['scale'], 'saved': not status['save_pending'],
                'save_error': status['save_error'], 'processed': bool(info.get('processed')),
                'failed': 'error' in info,
                'reviewed': is_page_reviewed(info), 'page_token': self._issue_page_token(),
                'images': images}

    @staticmethod
    def _validated_operations(operations, width, height):
        if not isinstance(operations, list) or not 1 <= len(operations) <= MAX_BATCH_OPERATIONS:
            raise ControlError('bad_request', 'operations 須為 1 至 64 個操作。')
        result = []
        for number, operation in enumerate(operations, 1):
            if not isinstance(operation, dict) or set(operation) - {'kind', 'target', 'shape', 'color_rgb'}:
                raise ControlError('bad_request', f'第 {number} 個操作格式無效。')
            kind, target, shape = operation.get('kind'), operation.get('target'), operation.get('shape')
            if kind not in ('restore', 'add', 'transfer') or not isinstance(shape, dict) or len(shape) != 1:
                raise ControlError('bad_request', f'第 {number} 個操作類型或形狀無效。')
            if kind == 'restore':
                if target is not None or 'color_rgb' in operation:
                    raise ControlError('bad_request', f'第 {number} 個恢復操作不能指定類別或顏色。')
            elif target not in ('solid', 'other'):
                raise ControlError('bad_request', f'第 {number} 個操作須指定 solid 或 other。')
            color = operation.get('color_rgb')
            if target == 'solid':
                if (not isinstance(color, list) or len(color) != 3
                        or any(type(value) is not int or not 0 <= value <= 255 for value in color)):
                    raise ControlError('bad_request', f'第 {number} 個純色操作須指定 RGB 顏色。')
            elif color is not None:
                raise ControlError('bad_request', f'第 {number} 個操作不需顏色。')
            if 'rect' in shape:
                rect = shape['rect']
                if (not isinstance(rect, list) or len(rect) != 4
                        or any(type(value) is not int for value in rect)):
                    raise ControlError('bad_request', f'第 {number} 個矩形格式無效。')
                x, y, rw, rh = rect
                if not (0 <= x < width and 0 <= y < height and rw > 0 and rh > 0
                        and x + rw <= width and y + rh <= height):
                    raise ControlError('bad_request', f'第 {number} 個矩形超出原圖。')
                bbox = (x, y, x + rw, y + rh)
                points = None
            elif 'polygon' in shape:
                points = shape['polygon']
                if (not isinstance(points, list) or not 3 <= len(points) <= 256
                        or any(not isinstance(p, list) or len(p) != 2
                               or any(type(v) is not int for v in p) for p in points)
                        or any(not (0 <= p[0] < width and 0 <= p[1] < height) for p in points)):
                    raise ControlError('bad_request', f'第 {number} 個多邊形格式或範圍無效。')
                if abs(cv2.contourArea(np.asarray(points, dtype=np.float32))) < .5:
                    raise ControlError('bad_request', f'第 {number} 個多邊形面積為零。')
                bbox = (min(p[0] for p in points), min(p[1] for p in points),
                        max(p[0] for p in points) + 1, max(p[1] for p in points) + 1)
            else:
                raise ControlError('bad_request', f'第 {number} 個操作缺少矩形或多邊形。')
            result.append((kind, target, bbox, points, color))
        return result

    def _apply_page_edits(self, params):
        self._ensure_idle()
        self._validate_page_token(params.get('page_token'))
        self._flush()
        self._validate_page_token(params.get('page_token'))
        if type(params.get('include_mask', False)) is not bool:
            raise ControlError('bad_request', 'include_mask 必須是布林值。')
        view_params = {key: params[key] for key in ('max_side', 'crop') if key in params}
        self._page_view({**view_params, 'kind': 'original'})
        w = self.window
        height, width = w.current_base.shape[:2]
        operations = self._validated_operations(params.get('operations'), width, height)
        # Prepare the full batch in detached arrays. A late invalid operation or
        # colour calculation cannot leave the GUI with a partly edited page.
        from detect_solid_inpaint_folder import prepare_page_edits
        solid = w.current_manual_solid.copy()
        other = w.current_manual_other.copy()
        page = {key: value.copy() for key, value in w.current_page.items()}
        for kind, target, bbox, points, color in operations:
            selection = np.zeros((height, width), dtype=np.uint8)
            x1, y1, x2, y2 = bbox
            if points is None:
                selection[y1:y2, x1:x2] = 1
            else:
                cv2.fillPoly(selection, [np.asarray(points, dtype=np.int32)], 1, lineType=cv2.LINE_8)
            selection = selection > 0
            if kind == 'restore':
                solid[selection] = 0
                other[selection] = 0
            elif kind == 'add':
                (solid if target == 'solid' else other)[selection] = 255
                (other if target == 'solid' else solid)[selection] = 0
            else:
                source = other if target == 'solid' else solid
                moved = selection & (source > 0)
                source[moved] = 0
                (solid if target == 'solid' else other)[moved] = 255
            try:
                page = prepare_page_edits(w.current_base, solid, other, state=page,
                                          dirty_bbox=bbox,
                                          fill_color=tuple(reversed(color)) if color is not None else None)
            except (OSError, ValueError) as exc:
                raise ControlError('edit_failed', str(exc)) from exc
        previous = w.current_page
        changed = np.any(previous['overlay'] != page['overlay'], axis=2)
        changed |= previous['other'] != page['other']
        changed |= previous['edited'] != page['edited']
        rows, cols = np.flatnonzero(np.any(changed, axis=1)), np.flatnonzero(np.any(changed, axis=0))
        if rows.size:
            dirty = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
            w.push_undo_snapshot()
            w.current_manual_solid, w.current_manual_other, w.current_page = solid, other, page
            w._edit_revision += 1
            w._update_committed_left(dirty)
            w._enqueue_current_page()
            try:
                self._flush()
            except ControlError as exc:
                raise ControlError('save_failed',
                                   f'{exc} 修改保留在 GUI 記憶體；可用 GUI 撤銷或重試保存。') from exc
            self._last_batch_stamp = self._page_stamp()
        else:
            dirty = None
        bundle = self._review_bundle(params)
        bundle['changed_bbox'] = list(dirty) if dirty else None
        bundle['saved'] = True
        return bundle

    def _undo_page_edit(self, params):
        self._ensure_idle()
        self._validate_page_token(params.get('page_token'))
        self._flush()
        self._validate_page_token(params.get('page_token'))
        if type(params.get('include_mask', False)) is not bool:
            raise ControlError('bad_request', 'include_mask 必須是布林值。')
        view_params = {key: params[key] for key in ('max_side', 'crop') if key in params}
        self._page_view({**view_params, 'kind': 'original'})
        w = self.window
        if self._last_batch_stamp != self._page_stamp() or not w.undo_stack:
            raise ControlError('stale_page', '最近一次操作已變更，不能代替 GUI 撤銷。')
        w.undo_mask()
        self._flush()
        self._last_batch_stamp = None
        return self._review_bundle(params)

    def _mark_page_reviewed(self, params):
        self._ensure_idle()
        self._validate_page_token(params.get('page_token'))
        self._flush()
        self._validate_page_token(params.get('page_token'))
        if type(params.get('include_mask', False)) is not bool:
            raise ControlError('bad_request', 'include_mask 必須是布林值。')
        view_params = {key: params[key] for key in ('max_side', 'crop') if key in params}
        self._page_view({**view_params, 'kind': 'original'})
        reviewed, advance = params.get('reviewed', True), params.get('advance', False)
        if type(reviewed) is not bool or type(advance) is not bool:
            raise ControlError('bad_request', 'reviewed 與 advance 必須是布林值。')
        w = self.window
        if not w.set_current_reviewed(reviewed):
            raise ControlError('save_failed', w._last_control_error or '頁面尚未成功偵測或校對標記保存失敗。')
        old_path = w.current_img_path
        if advance:
            w.next_image()
            if w.current_img_path == old_path and w.list_widget.currentRow() < len(w.imglist) - 1:
                raise ControlError('navigation_failed', '校對標記已保存，但切頁失敗。')
        if w.current_base is None or w.current_page is None:
            return {'saved': True, 'marked_reviewed': reviewed,
                    'advanced': w.current_img_path != old_path,
                    'next_page_unavailable': True, **self._status(), 'images': {}}
        try:
            result = self._review_bundle({**params, 'crop': None} if w.current_img_path != old_path else params)
        except ControlError as exc:
            if w.current_img_path == old_path:
                raise
            return {'saved': True, 'marked_reviewed': reviewed, 'advanced': True,
                    'next_bundle_error': {'code': exc.code, 'message': str(exc)},
                    **self._status(), 'images': {}}
        result['marked_reviewed'] = reviewed
        result['advanced'] = advance and w.current_img_path != old_path
        return result

    def _page_view(self, params):
        w = self.window
        if not w.current_img_path or w.current_base is None or w.current_page is None:
            raise ControlError('no_page', '目前沒有可查看的頁面。')
        if w.is_mask_stroke_active or w.is_local_edit_active or QApplication.activeModalWidget() is not None:
            raise ControlError('busy', '目前正在編輯選區，請完成後再讀取預覽。')
        kind = params.get('kind', 'composite')
        if kind not in ('original', 'composite', 'mask'):
            raise ControlError('bad_request', 'kind 只可為 original、composite 或 mask。')
        max_side = params.get('max_side', 1600)
        if type(max_side) is not int or not 256 <= max_side <= MAX_VIEW_SIDE:
            raise ControlError('bad_request', 'max_side 必須介於 256 和 2048。')
        base = w.current_base.copy()
        page = {name: value.copy() for name, value in w.current_page.items()}
        if base.ndim == 2:
            image = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        elif base.shape[2] == 4:
            image = cv2.cvtColor(base, cv2.COLOR_BGRA2BGR)
        else:
            image = base[:, :, :3].copy()
        if image.dtype != np.uint8:
            image = np.clip(image / 257 if image.dtype == np.uint16 else image, 0, 255).astype(np.uint8)
        if kind == 'composite':
            overlay = page['overlay']
            alpha = overlay[:, :, 3:4].astype(np.float32) / 255.0
            image = np.clip(image.astype(np.float32) * (1 - alpha) + overlay[:, :, :3].astype(np.float32) * alpha,
                            0, 255).astype(np.uint8)
            mask = page['other'] > 0
            if np.any(mask) and getattr(w, 'show_other_mask', True):
                tint = np.array(getattr(w, 'other_mask_display_color', (165, 110, 255)), dtype=np.float32)
                opacity = float(getattr(w, 'other_mask_display_alpha', .25))
                image[mask] = (image[mask].astype(np.float32) * (1 - opacity) + tint * opacity).astype(np.uint8)
        elif kind == 'mask':
            solid = page['overlay'][:, :, 3] > 0
            other = page['other'] > 0
            image[solid] = (image[solid] * .5 + np.array([255, 255, 255]) * .5).astype(np.uint8)
            image[other] = (image[other] * .5 + np.array([165, 110, 255]) * .5).astype(np.uint8)
        height, width = image.shape[:2]
        crop = params.get('crop')
        if crop is None:
            x, y, crop_width, crop_height = 0, 0, width, height
        elif (isinstance(crop, list) and len(crop) == 4
              and all(type(value) is int for value in crop)):
            x, y, crop_width, crop_height = crop
            if not (0 <= x < width and 0 <= y < height and crop_width > 0 and crop_height > 0
                    and x + crop_width <= width and y + crop_height <= height):
                raise ControlError('bad_request', 'crop 超出原圖範圍。')
        else:
            raise ControlError('bad_request', 'crop 必須是 [x, y, width, height] 原圖座標。')
        image = image[y:y + crop_height, x:x + crop_width]
        scale = min(4.0 if crop is not None else 1.0, max_side / max(crop_height, crop_width))
        if scale != 1:
            image = cv2.resize(image, (max(1, round(crop_width * scale)), max(1, round(crop_height * scale))),
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        ok, png = cv2.imencode('.png', image)
        if not ok:
            raise ControlError('image_failed', '無法編碼頁面預覽。')
        return {'name': Path(w.current_img_path).name, 'page_index': w.list_widget.currentRow() + 1,
                'width': width, 'height': height, 'revision': w._edit_revision,
                'origin': [x, y], 'crop_width': crop_width, 'crop_height': crop_height,
                'image_width': image.shape[1], 'image_height': image.shape[0], 'scale': scale,
                'save_pending': self._status()['save_pending'],
                'kind': kind, 'mime_type': 'image/png',
                'image_png_base64': base64.b64encode(png).decode('ascii')}


def start_control(window):
    """Start the GUI endpoint after MainWindow has built its widgets."""
    return ControlServer(window)


def request_control(method: str, params: dict | None = None, *, timeout_ms: int = 30000) -> dict:
    """Call the running GUI from another process or a non-GUI test thread."""
    request_id = uuid.uuid4().hex
    payload = json.dumps({'id': request_id, 'method': method, 'params': params or {}},
                         ensure_ascii=False).encode('utf-8') + b'\n'
    if len(payload) > MAX_MESSAGE:
        raise ControlError('too_large', '請求超過大小限制。')
    if os.name == 'posix':
        # Qt uses a Unix socket in the user temp directory. The stdlib client
        # reliably receives the final response even when Qt closes immediately.
        path = server_name()
        buffer = bytearray()
        try:
            with py_socket.socket(py_socket.AF_UNIX, py_socket.SOCK_STREAM) as client:
                client.settimeout(timeout_ms / 1000)
                client.connect(path)
                client.sendall(payload)
                while b'\n' not in buffer:
                    part = client.recv(min(65536, MAX_MESSAGE - len(buffer) + 1))
                    if not part:
                        break
                    buffer.extend(part)
                    if len(buffer) > MAX_MESSAGE:
                        raise ControlError('too_large', '回應超過大小限制。')
        except FileNotFoundError as exc:
            raise ControlError('unavailable', 'Solid Inpaint GUI 未執行或控制端尚未就緒。') from exc
        except py_socket.timeout as exc:
            raise ControlError('timeout', '等待 GUI 回應逾時。') from exc
        except ConnectionError as exc:
            raise ControlError('unavailable', f'無法連接 Solid Inpaint GUI：{exc}') from exc
    else:
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        socket = QLocalSocket()
        socket.connectToServer(server_name())
        if not socket.waitForConnected(min(timeout_ms, 3000)):
            raise ControlError('unavailable', 'Solid Inpaint GUI 未執行或控制端尚未就緒。')
        socket.write(payload)
        if not socket.waitForBytesWritten(min(timeout_ms, 3000)):
            raise ControlError('timeout', '無法送出控制請求。')
        buffer = bytearray()
        deadline = time.monotonic() + timeout_ms / 1000
        while b'\n' not in buffer:
            if socket.bytesAvailable():
                buffer.extend(bytes(socket.readAll()))
                if len(buffer) > MAX_MESSAGE:
                    raise ControlError('too_large', '回應超過大小限制。')
                continue
            left = int((deadline - time.monotonic()) * 1000)
            if left <= 0:
                raise ControlError('timeout', '等待 GUI 回應逾時。')
            ready = socket.waitForReadyRead(left)
            if not ready and not socket.bytesAvailable():
                raise ControlError('timeout', '等待 GUI 回應逾時。')
    if b'\n' not in buffer:
        raise ControlError('protocol_error', 'GUI 未回傳完整控制結果。')
    response = json.loads(buffer.partition(b'\n')[0])
    if response.get('id') != request_id:
        raise ControlError('protocol_error', '控制回應 ID 不符。')
    if not response.get('ok'):
        error = response.get('error', {})
        raise ControlError(error.get('code', 'failed'), error.get('message', '控制請求失敗。'))
    return response['result']
