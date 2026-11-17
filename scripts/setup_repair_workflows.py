"""Inspect or explicitly prepare optional ComfyUI repair workflow dependencies.

All resources are relative to this repository. Planning/checking never installs
anything. Model downloads and pinned node installation are separate commands;
this tool never installs ComfyUI, changes its Git revision, or selects PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / 'repair_workflows' / 'setup_manifest.json'
ALIASES = {'klein': 'klein9b_fp8_lanpaint', 'qwen': 'qwen21_int8_convrot'}


def load_manifest(path=MANIFEST_PATH):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if value.get('schema_version') != 1:
        raise ValueError('Unsupported repair setup manifest version')
    return value


def selected_methods(manifest, requested=None):
    names = [ALIASES.get(name, name) for name in (requested or manifest['methods'])]
    if not names or any(name not in manifest['methods'] for name in names):
        raise ValueError('Unknown method; use the list command for available methods')
    return list(dict.fromkeys(names))


def selected_resources(manifest, methods, section):
    ids = {item for method in methods for item in manifest['methods'][method][section]}
    return {key: value for key, value in manifest[section].items() if key in ids}


def read_model_map(path):
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or any(not isinstance(v, str) for v in value.values()):
        raise ValueError('Model map must be a JSON object of model IDs/filenames to paths')
    return value


def model_path(model_id, model, comfy_root, model_map=None):
    """A mapping can reuse a filename or an absolute, existing external model."""
    model_map = model_map or {}
    mapped = model_map.get(model_id, model_map.get(model['filename']))
    if mapped is None:
        if comfy_root is None:
            return None
        return Path(comfy_root) / 'models' / model['category'] / model['filename']
    path = Path(mapped).expanduser()
    if path.is_absolute():
        return path
    if comfy_root is None:
        raise ValueError('Relative model-map entries require --comfy-root')
    if '..' in path.parts:
        raise ValueError('Relative model-map entries may not escape the model category')
    return Path(comfy_root) / 'models' / model['category'] / path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_model(path, model, *, verify=False):
    if path is None:
        return {'state': 'path_required'}
    path = Path(path)
    if not path.is_file():
        return {'state': 'missing', 'path': str(path)}
    size = path.stat().st_size
    if size != model['size']:
        return {'state': 'size_mismatch', 'path': str(path), 'bytes': size}
    if verify:
        digest = file_sha256(path)
        return {'state': 'verified' if digest == model['sha256'] else 'checksum_mismatch',
                'path': str(path), 'bytes': size, 'sha256': digest}
    return {'state': 'present_unverified', 'path': str(path), 'bytes': size}


def normalized_remote(value):
    value = value.strip().rstrip('/').removesuffix('.git').casefold()
    return re.sub(r'^(?:ssh://)?git@github\.com[:/]', 'https://github.com/', value)


def git_read(directory, *arguments):
    # Scoped trust permits read-only inspection of an explicitly named Comfy
    # checkout without modifying the user's global Git configuration.
    directory = Path(directory).resolve()
    process = subprocess.run(['git', '--no-optional-locks', '-c', f'safe.directory={directory}', '-C', str(directory),
                              *arguments], capture_output=True, text=True, check=True)
    return process.stdout.strip()


def inspect_node(directory, node):
    directory = Path(directory)
    if not directory.exists():
        return {'state': 'missing', 'path': str(directory)}
    if not (directory / '.git').exists():
        return {'state': 'unmanaged_conflict', 'path': str(directory),
                'message': 'Existing non-Git installation is preserved; its pin is not certified'}
    try:
        remote = git_read(directory, 'config', '--get', 'remote.origin.url')
        commit = git_read(directory, 'rev-parse', 'HEAD')
        dirty = git_read(directory, 'status', '--porcelain', '--untracked-files=no')
        matching = (normalized_remote(remote) == normalized_remote(node['repo'])
                    and commit == node['commit'] and not dirty)
        return {'state': 'verified' if matching else 'version_conflict', 'path': str(directory),
                'remote': remote, 'commit': commit, 'tracked_changes': bool(dirty)}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'state': 'inspection_failed', 'path': str(directory), 'error': str(exc)}


def server_check(server, classes, models):
    """Only read documented ComfyUI discovery endpoints; never enqueue work."""
    result = {}
    base = server.rstrip('/')
    try:
        for endpoint in ('system_stats', 'queue', 'object_info'):
            with urllib.request.urlopen(base + '/' + endpoint, timeout=10) as response:
                result[endpoint] = json.load(response)
        objects = result.pop('object_info')
        result['missing_nodes'] = sorted(set(classes) - set(objects))
        missing_models = []
        for model_id, model in models.items():
            entry = objects.get(model['loader'], {}).get('input', {}).get('required', {})
            declaration = entry.get(model['loader_input'], [[]])[0]
            choices = declaration if isinstance(declaration, list) else []
            normalized_choices = {name.replace('\\', '/') for name in choices}
            if model['runtime_name'].replace('\\', '/') not in normalized_choices:
                missing_models.append({'id': model_id, 'filename': model['runtime_name'],
                                       'loader': model['loader']})
        result['missing_models'] = missing_models
        queue = result['queue']
        result['idle'] = not queue.get('queue_running') and not queue.get('queue_pending')
        result['state'] = 'ready' if not result['missing_nodes'] and not missing_models else 'incomplete'
    except (OSError, ValueError, KeyError) as exc:
        result.update(state='unreachable_or_invalid', error=str(exc))
    return result


def plan(manifest, methods, *, comfy_root=None, python=None, model_map=None,
         verify=False, inspect=False, server=None):
    root = Path(comfy_root).expanduser().resolve() if comfy_root else None
    selected = selected_resources(manifest, methods, 'models')
    models = {}
    runtime_models = {}
    for model_id, item in selected.items():
        path = model_path(model_id, item, root, model_map)
        runtime_name = item['filename']
        if path is not None:
            category = root / 'models' / item['category'] if root else None
            if category is not None and path.is_relative_to(category):
                runtime_name = path.relative_to(category).as_posix()
            else:
                runtime_name = path.name
        models[model_id] = {**item, 'target': str(path) if path is not None else None,
                            'runtime_name': runtime_name}
        if inspect:
            models[model_id]['local'] = inspect_model(path, item, verify=verify)
        runtime_models[model_id] = {**item, 'runtime_name': runtime_name}
    nodes = {}
    for node_id, item in selected_resources(manifest, methods, 'nodes').items():
        target = root / 'custom_nodes' / item['directory'] if root else None
        nodes[node_id] = {**item, 'target': str(target) if target else None}
        if inspect:
            nodes[node_id]['local'] = (inspect_node(target, item) if target else {'state': 'path_required'})
    result = {'methods': methods, 'machine': {'os': platform.system(), 'architecture': platform.machine()},
              'comfy_root': str(root) if root else None, 'comfy_python': str(python) if python else None,
              'runtime': manifest['runtime'], 'comfy_installation': manifest['comfy_installation'],
              'models': models, 'nodes': nodes,
              'model_map_note': 'Absolute files must also be visible in ComfyUI extra_model_paths; check queries its loader names.',
              'actions': ['Install ComfyUI/PyTorch separately using the official instructions if absent',
                          'download-models downloads only selected exact model versions',
                          'install-nodes adds missing pinned repositories; it never resets existing repositories',
                          'Requirements run only with --install-requirements and explicit --python; installed PyTorch is constrained, never selected or upgraded'],
              'required_disk_bytes': sum(item['size'] for item in selected.values())}
    if server:
        classes = {name for method in methods for name in manifest['methods'][method]['required_classes']}
        result['server'] = server_check(server, classes, runtime_models)
    return result


def download_model(model, target, *, opener=None, token=None):
    """Resume a verified-size Range download; preserve invalid existing files."""
    target = Path(target)
    opener = opener or urllib.request.urlopen
    if target.exists():
        if inspect_model(target, model, verify=True)['state'] != 'verified':
            raise FileExistsError(f'Existing model is invalid and was preserved: {target}')
        return {'state': 'reused', 'path': str(target)}
    source_info = model['sources'].get('huggingface')
    if source_info is None:
        raise ValueError(f'No verified Hugging Face source for {model["filename"]}')
    parsed = urllib.parse.urlsplit(source_info['url'])
    if parsed.scheme != 'https' or parsed.hostname != 'huggingface.co':
        raise ValueError('Pinned model downloads must use the Hugging Face HTTPS source')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > model['size']:
        raise ValueError(f'Partial model is oversized and was preserved: {partial}')
    if offset < model['size']:
        headers = {'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        if token:
            headers['Authorization'] = 'Bearer ' + token
        request = urllib.request.Request(source_info['url'], headers=headers)
        try:
            with opener(request, timeout=60) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',
                                         response.headers.get('Content-Range', ''))
                    if (not match or int(match[1]) != offset or int(match[3]) != model['size']
                            or int(match[2]) != model['size'] - 1):
                        raise ValueError('Download Content-Range does not match the requested bytes')
                elif status != 200 or offset:
                    raise ValueError('Resumed model download requires a matching HTTP 206 response')
                expected_response = model['size'] - offset
                length = response.headers.get('Content-Length')
                if length is not None and int(length) != expected_response:
                    raise ValueError('Download Content-Length differs from the pinned model size')
                with partial.open('ab' if offset else 'wb') as stream:
                    downloaded = offset
                    while chunk := response.read(1024 * 1024):
                        if downloaded + len(chunk) > model['size']:
                            raise ValueError('Download exceeds the pinned model size')
                        stream.write(chunk)
                        downloaded += len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError('Hugging Face access denied: accept the model license and set HF_TOKEN; no alternative model was selected') from None
            raise
    if inspect_model(partial, model, verify=True)['state'] != 'verified':
        raise ValueError(f'Incomplete or checksum-mismatched download was preserved: {partial}')
    if target.exists():
        raise FileExistsError(f'Model appeared during download; partial was preserved: {target}')
    # Publish without overwriting a file that another process may have created
    # since the last check. The verified partial is on the same filesystem.
    os.link(partial, target)
    partial.unlink()
    return {'state': 'downloaded', 'path': str(target)}


def run_command(arguments):
    return subprocess.run([str(argument) for argument in arguments], check=True,
                          capture_output=True, text=True).stdout.strip()


def install_requirements(directory, python):
    """Constrain already installed framework versions while installing node deps."""
    if python is None:
        raise ValueError('--install-requirements requires explicit --python for the ComfyUI environment')
    requirements = Path(directory) / 'requirements.txt'
    if not requirements.is_file():
        return {'state': 'no_requirements_file'}
    code = ('import importlib.metadata as m,json; '
            'names={d.metadata["Name"].lower():d.version for d in m.distributions()}; '
            'print(json.dumps({k:names[k] for k in ("torch","torchvision","torchaudio") if k in names}))')
    versions = json.loads(run_command([python, '-c', code]))
    if 'torch' not in versions:
        raise ValueError('Install ComfyUI and its supported PyTorch first; this tool does not choose PyTorch')
    with tempfile.TemporaryDirectory(prefix='repair-node-requirements-') as temp:
        constraints = Path(temp) / 'framework-constraints.txt'
        constraints.write_text(''.join(f'{name}=={version}\n' for name, version in versions.items()), encoding='utf-8')
        run_command([python, '-m', 'pip', 'install', '--no-input', '--constraint', constraints,
                     '--requirement', requirements])
    return {'state': 'installed', 'framework_constraints': versions}


def install_node(comfy_root, node, *, python=None, requirements=False):
    root = Path(comfy_root).resolve()
    name = node['directory']
    if (not isinstance(name, str) or name in ('', '.', '..')
            or '/' in name or '\\' in name or ':' in name):
        raise ValueError('Node directory must be a single relative directory name')
    if not (root / 'main.py').is_file():
        raise ValueError('--comfy-root must identify the ComfyUI directory containing main.py')
    if requirements and python is None:
        raise ValueError('--install-requirements requires explicit --python')
    destination = root / 'custom_nodes' / node['directory']
    current = inspect_node(destination, node)
    if current['state'] == 'missing':
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_name('.' + destination.name + '.install-' + uuid.uuid4().hex[:10])
        try:
            run_command(['git', 'clone', '--no-checkout', '--filter=blob:none', node['repo'], staging])
            run_command(['git', '-C', staging, 'checkout', '--detach', node['commit']])
            current = inspect_node(staging, node)
            if current['state'] != 'verified':
                raise ValueError(f'Cloned node does not match its public repository pin: {current}')
            if destination.exists():
                raise FileExistsError(f'Node appeared during installation: {destination}')
            os.replace(staging, destination)
            current = {'state': 'installed', 'path': str(destination), 'commit': node['commit']}
        except Exception as exc:
            raise RuntimeError(f'Node installation failed; existing files were preserved, staging: {staging}; {exc}') from exc
    elif current['state'] != 'verified':
        return current
    if requirements:
        current['requirements'] = install_requirements(destination, python)
    return current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('list', 'plan', 'check', 'verify', 'download-models', 'install-nodes'))
    parser.add_argument('--methods', nargs='+', help='klein, qwen, or the canonical method IDs; default both')
    parser.add_argument('--comfy-root', type=Path)
    parser.add_argument('--python', type=Path, help='Explicit Python executable belonging to ComfyUI')
    parser.add_argument('--model-map', type=Path, help='JSON mapping model IDs or filenames to reusable actual paths')
    parser.add_argument('--server', help='Read-only ComfyUI API check, e.g. http://127.0.0.1:8188')
    parser.add_argument('--install-requirements', action='store_true', help='Explicitly install node requirements in --python; constrain existing framework versions')
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest()
        methods = selected_methods(manifest, args.methods)
        model_map = read_model_map(args.model_map)
        known_model_keys = set(manifest['models']) | {
            item['filename'] for item in manifest['models'].values()}
        if set(model_map) - known_model_keys:
            raise ValueError('Unknown model-map keys: ' + ', '.join(sorted(set(model_map) - known_model_keys)))
        if args.command == 'list':
            result = {'methods': {name: manifest['methods'][name] for name in methods}}
        elif args.command in ('plan', 'check', 'verify'):
            result = plan(manifest, methods, comfy_root=args.comfy_root, python=args.python,
                          model_map=model_map, verify=args.command == 'verify',
                          inspect=args.command != 'plan', server=args.server)
        else:
            if args.comfy_root is None:
                raise ValueError('This command requires explicit --comfy-root')
            result = {'methods': methods, 'results': {}}
            section = 'models' if args.command == 'download-models' else 'nodes'
            for resource_id, resource in selected_resources(manifest, methods, section).items():
                try:
                    if section == 'models':
                        target = model_path(resource_id, resource, args.comfy_root, model_map)
                        item = download_model(resource, target, token=os.environ.get('HF_TOKEN'))
                    else:
                        item = install_node(args.comfy_root, resource, python=args.python,
                                            requirements=args.install_requirements)
                    result['results'][resource_id] = item
                except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                    result['results'][resource_id] = {'state': 'failed', 'error': str(exc)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.command in ('check', 'verify'):
            states = [entry['local']['state'] for category in ('models', 'nodes') for entry in result[category].values()]
            return 0 if all(state in ('verified', 'present_unverified') for state in states) and result.get('server', {}).get('state', 'ready') == 'ready' else 1
        if 'results' in result:
            return 0 if all(item['state'] in ('downloaded', 'reused', 'installed', 'verified') for item in result['results'].values()) else 1
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(json.dumps({'state': 'failed', 'error': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
