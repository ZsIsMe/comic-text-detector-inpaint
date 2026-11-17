"""Independent, resumable ComfyUI repair methods using bundled API workflows.

This client uploads and queues its own jobs. It never interrupts other jobs,
clears queues, frees models or launches a ComfyUI server. Filename readiness
checks do not establish the identity/hash of installed model weights.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np


METHOD_IDS = ('klein9b_fp8_lanpaint', 'qwen21_int8_convrot')
WORKFLOW_ROOT = Path(__file__).resolve().parent / 'repair_workflows'
_UNRESOLVED = {'submitting', 'submission_unknown', 'submitted', 'completed_remote'}


class ComfyRepairError(RuntimeError):
    """An actionable service, workflow or saved-job error."""


class SubmissionUnknownError(ComfyRepairError):
    """Submission may have reached the service; never resubmit automatically."""


def validate_methods(methods):
    if isinstance(methods, (str, bytes)):
        raise ValueError('Repair methods must be a sequence, not a string.')
    try:
        values = tuple(methods)
    except TypeError as exc:
        raise ValueError('Repair methods must be a sequence.') from exc
    if any(not isinstance(value, str) or value not in METHOD_IDS for value in values):
        raise ValueError(f'Unknown repair method; choose from {METHOD_IDS}.')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate repair methods are not allowed.')
    return values


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _array_digest(array):
    array = np.ascontiguousarray(array)
    return _digest(_canonical({'shape': array.shape, 'dtype': str(array.dtype)}) + array.tobytes())


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _save_job(path, job):
    job['updated_at'] = time.time()
    job.pop('manifest_digest', None)
    job['manifest_digest'] = _digest(_canonical(job))
    _atomic_bytes(path, _canonical(job) + b'\n')


def _load_job(path):
    try:
        job = json.loads(Path(path).read_text(encoding='utf-8'))
        saved_digest = job.get('manifest_digest')
        unsigned = {key: value for key, value in job.items() if key != 'manifest_digest'}
        if not saved_digest or saved_digest != _digest(_canonical(unsigned)):
            raise ValueError('job manifest checksum does not match')
        return job
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ComfyRepairError(f'Cannot trust saved job {path}: {exc}. No new job was submitted.') from exc


def _png_bytes(array, *, rgb=False):
    pixels = cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if rgb else array
    ok, encoded = cv2.imencode('.png', pixels)
    if not ok:
        raise ComfyRepairError('Cannot encode repair input/output PNG.')
    return encoded.tobytes()


def _read_png(path):
    try:
        pixels = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    except (OSError, ValueError, cv2.error) as exc:
        raise ComfyRepairError(f'Cannot decode PNG {path}: {exc}') from exc
    if pixels is None:
        raise ComfyRepairError(f'Cannot decode PNG: {path}')
    return pixels


def _remote_path(name, folder):
    if (not isinstance(name, str) or not name or name in ('.', '..')
            or any(c in name for c in ('/', '\\', ':', '\x00'))
            or not isinstance(folder, str) or folder.startswith(('/', '\\'))
            or ':' in folder or '\x00' in folder
            or any(part in ('.', '..') for part in folder.replace('\\', '/').split('/'))):
        raise ValueError('invalid remote image path')
    return '/'.join(filter(None, (folder.replace('\\', '/'), name)))


def _pid_alive(pid):
    if not isinstance(pid, int) or pid < 1:
        return None
    if os.name == 'nt':
        # os.kill(pid, 0) is not a safe existence probe on Windows.
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            return False if error == 87 else None
        try:
            code = wintypes.DWORD()
            return code.value == 259 if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else None
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None


@contextmanager
def _job_lock(path):
    path = Path(path).with_suffix('.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    token = str(uuid.uuid4())
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            owner = json.loads(path.read_text(encoding='utf-8'))
            alive = _pid_alive(owner.get('pid')) if owner.get('host') == socket.gethostname() else None
            state = 'active' if alive is True else 'stale (owner exited)' if alive is False else 'unverified'
            detail = f"{state}, PID {owner.get('pid')}, host {owner.get('host')}"
        except (OSError, ValueError, AttributeError):
            detail = 'unreadable lock metadata'
        raise ComfyRepairError(f'Page job is locked: {path} ({detail}). '
                               'Inspect the owning process before removing the lock.') from None
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(_canonical({'pid': os.getpid(), 'host': socket.gethostname(),
                                     'token': token, 'created_at': time.time()}))
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        try:
            if json.loads(path.read_text(encoding='utf-8')).get('token') == token:
                path.unlink()
        except (OSError, ValueError, AttributeError):
            pass


def difference_mask(original, generated, protected_mask=None):
    """RGB delta >=12, close3, area>=16, ellipse-grow8, then manual protection.

    The result intentionally is not intersected with OTHER: the workflow's real
    changes determine which generated pixels become a separate repair layer.
    """
    delta = np.max(np.abs(generated.astype(np.int16) - original.astype(np.int16)), axis=2)
    binary = np.where(delta >= 12, 255, 0).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, np.uint8)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] >= 16).astype(np.uint8) * 255
    binary = keep[labels]
    binary = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
    if protected_mask is not None:
        binary[np.asarray(protected_mask) > 0] = 0
    return binary


def _definition(method_id):
    validate_methods((method_id,))
    folder = WORKFLOW_ROOT / method_id
    manifest_bytes = (folder / 'manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get('method_id') != method_id or manifest.get('workflow_file') != 'workflow_api.json':
        raise ComfyRepairError(f'Invalid bundled method manifest: {method_id}')
    if manifest.get('dependency_setup_manifest') != '../setup_manifest.json':
        raise ComfyRepairError(f'Invalid bundled dependency manifest reference: {method_id}')
    # Expected model identities and dependency pins are authoritative metadata,
    # not proof about remote weights. Hash only this small JSON, never the
    # installed multi-GB model files during an actual repair run.
    manifest['_setup_manifest_sha256'] = _digest((folder.parent / 'setup_manifest.json').read_bytes())
    workflow_bytes = (folder / 'workflow_api.json').read_bytes()
    workflow = json.loads(workflow_bytes)
    return manifest, workflow, _digest(manifest_bytes), _digest(workflow_bytes)


def _dynamic_field(field, definitions):
    """Accept only declared COMFY_AUTOGROW_V3 named children, not arbitrary dots."""
    if '.' not in field:
        return False
    prefix, name = field.split('.', 1)
    spec = definitions.get(prefix)
    if not isinstance(spec, (list, tuple)) or not spec or spec[0] != 'COMFY_AUTOGROW_V3':
        return False
    options = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    template = options.get('template', {})
    names = (template.get('names', options.get('names', []))
             if isinstance(template, dict) else options.get('names', []))
    return name in names


class ComfyRepairClient:
    def __init__(self, server_url='http://127.0.0.1:8188', *, model_map=None,
                 timeout=30, poll_interval=1, task_timeout=3600):
        parsed = urllib.parse.urlsplit(server_url)
        if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError('ComfyUI URL must be an HTTP(S) server URL without query/fragment.')
        self.server_url = server_url.rstrip('/')
        for name, value in (('timeout', timeout), ('poll_interval', poll_interval), ('task_timeout', task_timeout)):
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or value < 0 or (name != 'poll_interval' and value == 0)):
                raise ValueError(f'Invalid ComfyUI {name}.')
        self.timeout, self.poll_interval, self.task_timeout = timeout, poll_interval, task_timeout
        if model_map is not None and not isinstance(model_map, dict):
            raise ValueError('model_map must map method IDs to model roles and installed filenames.')
        self.model_map = copy.deepcopy(model_map or {})
        validate_methods(tuple(self.model_map))
        for method, roles in self.model_map.items():
            known = _definition(method)[0]['model_nodes']
            if not isinstance(roles, dict) or any(role not in known for role in roles):
                raise ValueError(f'Unknown model role for {method}; expected {tuple(known)}.')
            if any(not isinstance(name, str) or not name.strip() for name in roles.values()):
                raise ValueError('Installed model filenames must be nonempty strings.')

    def _request(self, method, path, *, data=None, headers=None):
        request = urllib.request.Request(self.server_url + path, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode('utf-8', errors='replace')[:1600]
            raise ComfyRepairError(f'ComfyUI {method} {path} HTTP {exc.code}: {detail}') from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise ComfyRepairError(f'ComfyUI {method} {path} failed: {exc}') from exc

    def _json(self, method, path, payload=None):
        headers = {'Content-Type': 'application/json'} if payload is not None else {}
        data = _canonical(payload) if payload is not None else None
        raw = self._request(method, path, data=data, headers=headers)
        try:
            return json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ComfyRepairError(f'Invalid JSON from ComfyUI {path}.') from exc

    def _models(self, method_id, manifest):
        overrides = self.model_map.get(method_id, {})
        return {role: overrides.get(role, node['filename'])
                for role, node in manifest['model_nodes'].items()}

    def check(self, methods):
        methods = validate_methods(methods)
        report = {'ready': False, 'server_url': self.server_url, 'methods': {},
                  'missing_nodes': [], 'missing_models': [], 'missing_inputs': [], 'errors': [],
                  'model_identity_verified': False,
                  'model_check': 'Installed dropdown filenames only; weights are not hash-verified.'}
        if not methods:
            report['ready'] = True
            return report
        try:
            report['system_stats'] = self._json('GET', '/system_stats')
            object_info = self._json('GET', '/object_info')
            if not isinstance(object_info, dict):
                raise ComfyRepairError('ComfyUI /object_info is not a node registry.')
            for method in methods:
                manifest, workflow, _, _ = _definition(method)
                models = self._models(method, manifest)
                missing_nodes, missing_inputs, missing_models = [], [], []
                for node_id, node in workflow.items():
                    class_type = node['class_type']
                    schema = object_info.get(class_type)
                    if not isinstance(schema, dict):
                        missing_nodes.append(class_type)
                        continue
                    definitions = {}
                    for category in ('required', 'optional', 'hidden'):
                        definitions.update(schema.get('input', {}).get(category, {}))
                    for field in node['inputs']:
                        if field not in definitions and not _dynamic_field(field, definitions):
                            missing_inputs.append({'node': node_id, 'class_type': class_type, 'input': field})
                    for field, definition in schema.get('input', {}).get('required', {}).items():
                        if field not in node['inputs'] and not any(
                                _dynamic_field(name, {field: definition}) for name in node['inputs']):
                            missing_inputs.append({'node': node_id, 'class_type': class_type,
                                                   'input': field, 'reason': 'required input absent from workflow'})
                for role, model in manifest['model_nodes'].items():
                    schema = object_info.get(model['class_type'], {})
                    fields = {**schema.get('input', {}).get('required', {}),
                              **schema.get('input', {}).get('optional', {})}
                    definition = fields.get(model['input'], [])
                    choices = definition[0] if isinstance(definition, (list, tuple)) and definition else None
                    if not isinstance(choices, (list, tuple)) or models[role] not in choices:
                        missing_models.append({'role': role, 'filename': models[role],
                                               'reason': 'filename not in loader dropdown'})
                details = {'ready': not (missing_nodes or missing_inputs or missing_models),
                           'missing_nodes': sorted(set(missing_nodes)), 'missing_inputs': missing_inputs,
                           'missing_models': missing_models, 'models': models,
                           'model_identity_verified': False}
                report['methods'][method] = details
                report['missing_nodes'].extend(details['missing_nodes'])
                report['missing_inputs'].extend({'method_id': method, **v} for v in missing_inputs)
                report['missing_models'].extend({'method_id': method, **v} for v in missing_models)
        except (ComfyRepairError, OSError, ValueError, TypeError, AttributeError) as exc:
            report['errors'].append(str(exc))
        report['missing_nodes'] = sorted(set(report['missing_nodes']))
        report['ready'] = (len(report['methods']) == len(methods) and not report['errors']
                           and all(value['ready'] for value in report['methods'].values()))
        return report

    def _upload(self, data, filename, subfolder):
        boundary = 'SolidInpaint-' + uuid.uuid4().hex
        chunks = []
        for name, value in (('type', 'input'), ('overwrite', 'false'), ('subfolder', subfolder)):
            chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
        chunks.extend((f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                       f'filename="{filename}"\r\nContent-Type: image/png\r\n\r\n'.encode('utf-8'),
                       data, f'\r\n--{boundary}--\r\n'.encode()))
        raw = self._request('POST', '/upload/image', data=b''.join(chunks),
                            headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
        try:
            descriptor = json.loads(raw)
            name = descriptor['name']
            folder = descriptor.get('subfolder', '')
            remote = _remote_path(name, folder)
            if descriptor.get('type', 'input') != 'input':
                raise ValueError('invalid upload type')
        except (ValueError, KeyError, TypeError) as exc:
            raise ComfyRepairError('ComfyUI upload did not return a valid input image path.') from exc
        return remote

    @staticmethod
    def _progress(callback, stage, **values):
        if callback is not None:
            callback({'stage': stage, **values})

    @staticmethod
    def _entry_id(entry, client_id, requested_id):
        if isinstance(entry, (list, tuple)) and len(entry) > 3:
            prompt_id, metadata = entry[1], entry[3]
        elif isinstance(entry, dict):
            prompt_id = entry.get('prompt_id')
            metadata = entry.get('extra_data', entry)
        else:
            return None
        if prompt_id == requested_id or (isinstance(metadata, dict) and metadata.get('client_id') == client_id):
            return str(prompt_id) if prompt_id else None
        return None

    def _recover(self, job, manifest_path):
        requested = job['requested_prompt_id']
        exact = self._json('GET', '/history/' + urllib.parse.quote(requested, safe=''))
        candidates = {requested} if isinstance(exact, dict) and requested in exact else set()
        queue = self._json('GET', '/queue')
        if not isinstance(queue, dict):
            raise ComfyRepairError('Invalid ComfyUI queue response; submission was not repeated.')
        for section in ('queue_running', 'queue_pending'):
            for entry in queue.get(section, []):
                found = self._entry_id(entry, job['client_id'], requested)
                if found:
                    candidates.add(found)
        history = self._json('GET', '/history')
        if not isinstance(history, dict):
            raise ComfyRepairError('Invalid ComfyUI history response; submission was not repeated.')
        for prompt_id, entry in history.items():
            if prompt_id == requested or (isinstance(entry, dict)
                    and self._entry_id(entry.get('prompt'), job['client_id'], requested)):
                candidates.add(prompt_id)
        if len(candidates) != 1:
            job['status'] = 'submission_unknown'
            job['last_error'] = ('Submission cannot be uniquely located in ComfyUI queue/history. '
                                 'It will not be submitted again automatically.')
            _save_job(manifest_path, job)
            raise SubmissionUnknownError(job['last_error'])
        job['prompt_id'] = candidates.pop()
        job['status'] = 'submitted'
        _save_job(manifest_path, job)

    def _wait(self, job, manifest, manifest_path, progress):
        deadline = time.monotonic() + self.task_timeout
        prompt_id = job['prompt_id']
        while True:
            history = self._json('GET', '/history/' + urllib.parse.quote(prompt_id, safe=''))
            if not isinstance(history, dict):
                raise ComfyRepairError('Invalid ComfyUI history response; run again to resume the same prompt.')
            entry = history.get(prompt_id)
            if entry:
                status = entry.get('status', {})
                failures = [v for v in status.get('messages', [])
                            if isinstance(v, (list, tuple)) and v and v[0] in ('execution_error', 'execution_interrupted')]
                if status.get('status_str') == 'error' or failures:
                    job['status'], job['last_error'] = 'failed', f'ComfyUI execution failed: {failures or status}'
                    _save_job(manifest_path, job)
                    raise ComfyRepairError(job['last_error'])
                images = entry.get('outputs', {}).get(manifest['io']['output_node'], {}).get('images')
                if images is not None or status.get('completed') or status.get('status_str') == 'success':
                    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
                        raise ComfyRepairError('Expected exactly one image from the method output node.')
                    descriptor = images[0]
                    try:
                        _remote_path(descriptor.get('filename'), descriptor.get('subfolder', ''))
                        if descriptor.get('type', 'output') not in ('output', 'temp'):
                            raise ValueError('invalid output type')
                    except ValueError as exc:
                        raise ComfyRepairError('Invalid ComfyUI output image descriptor.') from exc
                    job['output_descriptor'] = descriptor
                    job['status'] = 'completed_remote'
                    _save_job(manifest_path, job)
                    return
            if time.monotonic() >= deadline:
                raise ComfyRepairError(f'ComfyUI job {prompt_id} is still pending/running; '
                                       'run again to resume this prompt without resubmitting.')
            self._progress(progress, 'waiting', method_id=job['method_id'], prompt_id=prompt_id)
            time.sleep(self.poll_interval)

    @staticmethod
    def _cache_valid(root, job, shape):
        if job.get('status') != 'success':
            return False
        outputs = job.get('outputs')
        if not isinstance(outputs, dict):
            return False
        if not outputs:
            return bool(job.get('empty') and job.get('empty_reason') == 'no_other')
        try:
            for role, detail in outputs.items():
                path = (root / detail['path']).resolve()
                if not path.is_relative_to(root) or _digest(path.read_bytes()) != detail['sha256']:
                    return False
                pixels = _read_png(path)
                expected_shape = (*shape, 4) if role == 'rgba' else shape if role == 'mask' else (*shape, 3)
                if pixels.shape != expected_shape or pixels.dtype != np.uint8:
                    return False
            return 'result' in outputs and (job.get('empty') or {'rgba', 'mask'}.issubset(outputs))
        except (OSError, ValueError, KeyError, TypeError, ComfyRepairError):
            return False

    @staticmethod
    def _result(root, manifest_path, job, *, reused):
        outputs = job.get('outputs', {})
        def output(role):
            return str(root / outputs[role]['path']) if role in outputs else None
        return {'method_id': job['method_id'], 'rgba_path': output('rgba'), 'mask_path': output('mask'),
                'result_path': output('result'), 'manifest_path': str(manifest_path),
                'reused': reused, 'empty': job['empty'], 'empty_reason': job.get('empty_reason'),
                'prompt_id': job.get('prompt_id'), 'seed': job['seed'],
                'model_identity_verified': False}

    def run(self, method_id, image_rgb, other_mask, output_dir, *, key,
            protected_mask=None, progress=None):
        """Run/resume one page; progress receives a dictionary with a stage field.

        Seeds initially come from the tested workflow and are persisted in the
        job manifest. Empty OTHER causes no HTTP requests and writes no PNGs.
        """
        validate_methods((method_id,))
        image, mask = np.asarray(image_rgb), np.asarray(other_mask)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
            raise ValueError('Comfy repair requires a nonempty uint8 RGB source image.')
        if mask.dtype != np.uint8 or mask.shape != image.shape[:2]:
            raise ValueError('Comfy repair requires an original uint8 OTHER mask matching the image.')
        protected = np.zeros(mask.shape, bool) if protected_mask is None else np.asarray(protected_mask)
        if (protected.shape != mask.shape or protected.dtype.kind not in 'buif'
                or not np.isfinite(protected).all() or np.any(protected < 0)):
            raise ValueError('Protected mask must be finite, nonnegative and match the source image.')
        protected = protected > 0
        if (not isinstance(key, str) or not key or key in ('.', '..')
                or any(ord(c) < 32 or c in '<>:"/\\|?*' for c in key) or key.endswith((' ', '.'))):
            raise ValueError('Repair key must be a safe original filename stem.')
        root = Path(output_dir).resolve()
        manifest_path = root / 'workflow_jobs' / method_id / f'{key}.json'
        manifest, workflow, manifest_hash, workflow_hash = _definition(method_id)
        models = self._models(method_id, manifest)
        seed = workflow[manifest['seed']['node']]['inputs'][manifest['seed']['input']]
        identity = {'method_id': method_id, 'server_url': self.server_url,
                    'input_rgb_sha256': _array_digest(image), 'input_other_sha256': _array_digest(mask),
                    'protected_sha256': _array_digest(protected), 'workflow_sha256': workflow_hash,
                    'method_manifest_sha256': manifest_hash,
                    'dependency_setup_manifest_sha256': manifest['_setup_manifest_sha256'],
                    'models': models, 'seed': seed}
        fingerprint = _digest(_canonical(identity))
        with _job_lock(manifest_path):
            job = _load_job(manifest_path) if manifest_path.exists() else None
            if job and job.get('fingerprint') != fingerprint:
                if job.get('status') in _UNRESOLVED:
                    raise ComfyRepairError('Inputs/workflow/server changed while a previous job is unresolved. '
                                           f'Recover that job first: {manifest_path}')
                job = None
            if job and self._cache_valid(root, job, mask.shape):
                return self._result(root, manifest_path, job, reused=True)
            if job and job.get('status') == 'success':
                # Corrupt/missing local outputs are fetched again from the same
                # completed prompt. They must not trigger another inference.
                job['status'] = 'completed_remote' if job.get('output_descriptor') else 'submitted'
            if job is None:
                job = {'schema_version': 1, **identity, 'fingerprint': fingerprint, 'status': 'prepared',
                       'client_id': str(uuid.uuid4()), 'attempt_id': str(uuid.uuid4()),
                       'requested_prompt_id': str(uuid.uuid4()), 'created_at': time.time(),
                       'outputs': {}, 'model_identity_verified': False}
                _save_job(manifest_path, job)
            if not np.any(mask):
                job.update(status='success', empty=True, empty_reason='no_other', outputs={})
                _save_job(manifest_path, job)
                return self._result(root, manifest_path, job, reused=False)
            try:
                if job['status'] == 'prepared':
                    readiness = self.check((method_id,))
                    if not readiness['ready']:
                        raise ComfyRepairError('ComfyUI method is not ready: ' + json.dumps(readiness, ensure_ascii=False))
                    self._progress(progress, 'upload', method_id=method_id)
                    subfolder = f"solid_inpaint/{job['client_id']}/{job['attempt_id']}"
                    input_name = self._upload(_png_bytes(image, rgb=True), 'source.png', subfolder)
                    mask_name = self._upload(_png_bytes(np.where(mask > 0, 255, 0).astype(np.uint8)),
                                             'other_mask.png', subfolder)
                    graph = copy.deepcopy(workflow)
                    graph[manifest['io']['image_node']]['inputs']['image'] = input_name
                    graph[manifest['io']['mask_node']]['inputs']['image'] = mask_name
                    graph[manifest['seed']['node']]['inputs'][manifest['seed']['input']] = job['seed']
                    for role, node in manifest['model_nodes'].items():
                        graph[node['node']]['inputs'][node['input']] = models[role]
                    graph[manifest['io']['output_node']]['inputs']['filename_prefix'] = (
                        f"SolidInpaint/{method_id}/{job['attempt_id']}")
                    job['submitted_workflow'] = graph
                    job['status'] = 'submitting'
                    _save_job(manifest_path, job)
                    self._progress(progress, 'submit', method_id=method_id)
                    try:
                        response = self._json('POST', '/prompt', {
                            'prompt': graph, 'client_id': job['client_id'],
                            'prompt_id': job['requested_prompt_id'],
                            'extra_data': {'client_id': job['client_id'], 'solid_inpaint_attempt_id': job['attempt_id']},
                        })
                        prompt_id = response.get('prompt_id')
                        if not isinstance(prompt_id, str) or not prompt_id:
                            raise ComfyRepairError('ComfyUI did not return a prompt_id: ' + str(response))
                    except Exception as exc:
                        job['status'], job['last_error'] = 'submission_unknown', str(exc)
                        _save_job(manifest_path, job)
                        raise SubmissionUnknownError(f'{exc}. Submission is unknown; rerun to recover, '
                                                     'not to submit again.') from exc
                    job['prompt_id'], job['status'] = prompt_id, 'submitted'
                    _save_job(manifest_path, job)
                if job['status'] in ('submitting', 'submission_unknown'):
                    self._recover(job, manifest_path)
                if job['status'] == 'failed':
                    raise ComfyRepairError('Saved ComfyUI job failed: ' + job.get('last_error', 'unknown error'))
                if job['status'] == 'submitted':
                    self._wait(job, manifest, manifest_path, progress)
                if job['status'] != 'completed_remote':
                    raise ComfyRepairError(f"Cannot safely resume job status: {job['status']}")
                self._progress(progress, 'download', method_id=method_id, prompt_id=job['prompt_id'])
                descriptor = job['output_descriptor']
                query = urllib.parse.urlencode({name: descriptor.get(name, default) for name, default in
                                                (('filename', ''), ('subfolder', ''), ('type', 'output'))})
                raw = self._request('GET', '/view?' + query)
                try:
                    decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
                except (ValueError, cv2.error) as exc:
                    raise ComfyRepairError('ComfyUI output image cannot be decoded.') from exc
                if (decoded is None or decoded.dtype != np.uint8 or decoded.ndim != 3
                        or decoded.shape[2] not in (3, 4) or decoded.shape[:2] != mask.shape):
                    raise ComfyRepairError('ComfyUI output must be a readable uint8 color image at original dimensions.')
                generated = cv2.cvtColor(decoded, cv2.COLOR_BGRA2RGB if decoded.shape[2] == 4 else cv2.COLOR_BGR2RGB)
                diff = difference_mask(image, generated, protected)
                outputs = {}

                def save_output(role, relative, pixels, *, rgb=False):
                    path = root / relative
                    data = _png_bytes(pixels, rgb=rgb)
                    _atomic_bytes(path, data)
                    outputs[role] = {'path': str(relative), 'sha256': _digest(data),
                                     'shape': list(pixels.shape), 'bytes': len(data)}

                save_output('result', Path('workflow_results') / method_id / f'{key}.png', generated, rgb=True)
                if np.any(diff):
                    rgba = np.zeros((*mask.shape, 4), np.uint8)
                    rgba[diff > 0, :3] = generated[diff > 0]
                    rgba[diff > 0, 3] = 255
                    save_output('mask', Path('repair_masks') / method_id / f'{key}.png', diff)
                    save_output('rgba', Path(method_id) / f'{key}.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
                job.update(status='success', outputs=outputs, empty=not bool(np.any(diff)),
                           empty_reason=None if np.any(diff) else 'no_unprotected_difference')
                job.pop('last_error', None)
                _save_job(manifest_path, job)
                self._progress(progress, 'complete', method_id=method_id, prompt_id=job['prompt_id'], empty=job['empty'])
                return self._result(root, manifest_path, job, reused=False)
            except Exception as exc:
                job['last_error'] = str(exc)
                _save_job(manifest_path, job)
                raise
