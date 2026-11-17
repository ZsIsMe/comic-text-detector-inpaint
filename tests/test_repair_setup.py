"""Dependency preparation contracts with fake HTTP bodies and subprocesses."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from scripts import setup_repair_workflows as setup


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class RepairSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.payload = b'correct pinned model bytes'
        self.model = {'filename': 'model.safetensors', 'category': 'diffusion_models',
                      'loader': 'UNETLoader', 'loader_input': 'unet_name',
                      'size': len(self.payload), 'sha256': hashlib.sha256(self.payload).hexdigest(),
                      'sources': {'huggingface': {'url': 'https://huggingface.co/test/model/resolve/pin/model.safetensors'}}}
        self.target = self.root / 'models' / 'diffusion_models' / self.model['filename']
        self.node = {'directory': 'test-node', 'repo': 'https://github.com/example/test-node.git',
                     'commit': 'a' * 40}

    def test_manifest_has_six_exact_models_and_four_public_pinned_nodes(self):
        manifest = setup.load_manifest()
        self.assertEqual(len(manifest['models']), 6)
        self.assertEqual(len(manifest['nodes']), 4)
        for model in manifest['models'].values():
            self.assertEqual(set(model['sources']), {'huggingface'})
            self.assertRegex(model['sha256'], r'^[0-9a-f]{64}$')
            self.assertGreater(model['size'], 0)
            for source in model['sources'].values():
                self.assertRegex(source['revision'], r'^[0-9a-f]{40}$')
                self.assertIn('/resolve/' + source['revision'] + '/', source['url'])
                self.assertNotIn('D:\\cursor', source['url'])
        self.assertTrue(manifest['models']['klein_unet']['sources']['huggingface']['gated'])
        self.assertNotIn('base', manifest['models']['klein_unet']['filename'])
        self.assertEqual(manifest['models']['qwen_unet']['sha256'],
                         'cb74113cb03faecd79611b01fd7fd642f0aa60d6f0b95086abee214d75eaa57d')
        for node in manifest['nodes'].values():
            self.assertTrue(node['repo'].startswith('https://github.com/'))
            self.assertRegex(node['commit'], r'^[0-9a-f]{40}$')

    def test_plan_is_read_only_and_can_guide_a_machine_without_comfy(self):
        manifest = setup.load_manifest()
        before = list(self.root.rglob('*'))
        with patch.object(setup.urllib.request, 'urlopen', side_effect=AssertionError('no network')):
            result = setup.plan(manifest, setup.selected_methods(manifest, ['qwen']))
        self.assertIsNone(result['comfy_root'])
        self.assertIn('https://docs.comfy.org/installation/manual_install',
                      result['comfy_installation'].values())
        self.assertEqual(len(result['models']), 3)
        self.assertEqual(list(self.root.rglob('*')), before)

    def test_existing_correct_model_is_reused_without_network(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.payload)
        with patch.object(setup.urllib.request, 'urlopen', side_effect=AssertionError('no download')):
            self.assertEqual(setup.download_model(self.model, self.target)['state'], 'reused')

    def test_existing_invalid_model_is_preserved_and_no_partial_is_created(self):
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(b'hand placed invalid weights')
        with patch.object(setup.urllib.request, 'urlopen', side_effect=AssertionError('no download')):
            with self.assertRaises(FileExistsError):
                setup.download_model(self.model, self.target)
        self.assertEqual(self.target.read_bytes(), b'hand placed invalid weights')
        self.assertFalse(self.target.with_name(self.target.name + '.partial').exists())

    def test_new_download_is_checked_and_atomically_published(self):
        calls = []
        def open_response(request, **kwargs):
            calls.append(request)
            self.assertFalse(self.target.exists())
            return Response(self.payload, headers={'Content-Length': str(len(self.payload))})
        record = setup.download_model(self.model, self.target, opener=open_response, token='test-secret')
        self.assertEqual(record['state'], 'downloaded')
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.target.with_name(self.target.name + '.partial').exists())
        self.assertEqual(calls[0].get_header('Authorization'), 'Bearer test-secret')

    def test_resume_requires_exact_206_range_and_keeps_partial_on_protocol_failure(self):
        self.target.parent.mkdir(parents=True)
        partial = self.target.with_name(self.target.name + '.partial')
        partial.write_bytes(self.payload[:7])
        def wrong(request, **kwargs):
            self.assertEqual(request.get_header('Range'), 'bytes=7-')
            return Response(self.payload, status=200)
        with self.assertRaisesRegex(ValueError, '206'):
            setup.download_model(self.model, self.target, opener=wrong)
        self.assertEqual(partial.read_bytes(), self.payload[:7])
        def correct(request, **kwargs):
            return Response(self.payload[7:], status=206, headers={
                'Content-Range': f'bytes 7-{len(self.payload)-1}/{len(self.payload)}',
                'Content-Length': str(len(self.payload) - 7)})
        setup.download_model(self.model, self.target, opener=correct)
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_bad_range_size_and_checksum_never_publish_target(self):
        with self.assertRaisesRegex(ValueError, 'Content-Range'):
            setup.download_model(self.model, self.target, opener=lambda *a, **k: Response(
                self.payload, 206, {'Content-Range': f'bytes 0-2/{len(self.payload)}'}))
        with self.assertRaisesRegex(ValueError, 'preserved'):
            setup.download_model(self.model, self.target, opener=lambda *a, **k: Response(
                b'x' * len(self.payload)))
        self.assertFalse(self.target.exists())
        self.assertEqual(self.target.with_name(self.target.name + '.partial').read_bytes(),
                         b'x' * len(self.payload))

    def test_interruption_preserves_received_partial_bytes(self):
        class Interrupted(Response):
            def read(self, size=-1):
                if self.tell():
                    raise OSError('connection interrupted')
                return super().read(5)
        with self.assertRaisesRegex(OSError, 'interrupted'):
            setup.download_model(self.model, self.target,
                                 opener=lambda *a, **k: Interrupted(self.payload))
        self.assertEqual(self.target.with_name(self.target.name + '.partial').read_bytes(), self.payload[:5])
        self.assertFalse(self.target.exists())

    def test_gated_access_reports_action_without_leaking_token(self):
        def denied(*args, **kwargs):
            raise urllib.error.HTTPError('https://huggingface.co/test', 403, 'denied', {}, None)
        with self.assertRaisesRegex(PermissionError, 'accept the model license') as caught:
            setup.download_model(self.model, self.target, opener=denied, token='secret-token')
        self.assertNotIn('secret-token', str(caught.exception))

    def test_model_map_reuses_actual_filename_and_check_does_not_hash(self):
        mapped = self.root / 'models' / 'diffusion_models' / 'actual-name.safetensors'
        mapped.parent.mkdir(parents=True)
        mapped.write_bytes(self.payload)
        self.assertEqual(setup.model_path('test', self.model, self.root,
                                         {'model.safetensors': 'actual-name.safetensors'}), mapped)
        with patch.object(setup, 'file_sha256', side_effect=AssertionError('not a verification')):
            self.assertEqual(setup.inspect_model(mapped, self.model)['state'], 'present_unverified')
        self.assertEqual(setup.inspect_model(mapped, self.model, verify=True)['state'], 'verified')
        with self.assertRaises(ValueError):
            setup.model_path('test', self.model, self.root, {'test': '../escape'})

    def test_existing_different_node_is_reported_without_reset_or_install(self):
        (self.root / 'main.py').touch()
        with patch.object(setup, 'inspect_node', return_value={'state': 'version_conflict'}), \
                patch.object(setup, 'run_command', side_effect=AssertionError('must not reset')):
            result = setup.install_node(self.root, self.node)
        self.assertEqual(result['state'], 'version_conflict')

    def test_node_paths_cannot_escape_explicit_comfy_root(self):
        (self.root / 'main.py').touch()
        for name in ('../outside', 'C:\\outside', '/outside', '..', 'nested/node'):
            with self.subTest(name=name), patch.object(setup, 'run_command', side_effect=AssertionError('no install')):
                with self.assertRaisesRegex(ValueError, 'single relative'):
                    setup.install_node(self.root, {**self.node, 'directory': name})

    def test_model_appearing_during_publish_is_not_overwritten(self):
        def racing_link(source, target):
            Path(target).write_bytes(b'appeared during download')
            raise FileExistsError('exists')
        with patch.object(setup.os, 'link', side_effect=racing_link):
            with self.assertRaises(FileExistsError):
                setup.download_model(self.model, self.target,
                                     opener=lambda *a, **k: Response(self.payload))
        self.assertEqual(self.target.read_bytes(), b'appeared during download')
        self.assertEqual(self.target.with_name(self.target.name + '.partial').read_bytes(), self.payload)

    def test_new_node_is_checked_at_pin_before_atomic_move(self):
        (self.root / 'main.py').touch()
        commands = []
        def command(args):
            commands.append(args)
            if args[1] == 'clone':
                Path(args[-1]).mkdir()
            return ''
        with patch.object(setup, 'inspect_node', side_effect=[{'state': 'missing'}, {'state': 'verified'}]), \
                patch.object(setup, 'run_command', side_effect=command):
            result = setup.install_node(self.root, self.node)
        self.assertEqual(result['state'], 'installed')
        self.assertTrue((self.root / 'custom_nodes' / 'test-node').is_dir())
        self.assertIn('checkout', commands[1])
        self.assertIn(self.node['commit'], commands[1])
        self.assertFalse(any('reset' in item for command in commands for item in map(str, command)))

    def test_requirements_need_explicit_comfy_python_and_lock_existing_framework(self):
        directory = self.root / 'node'
        directory.mkdir()
        (directory / 'requirements.txt').write_text('numpy\ntorch\n')
        with self.assertRaisesRegex(ValueError, '--python'):
            setup.install_requirements(directory, None)
        commands = []
        def run(arguments):
            commands.append(arguments)
            if arguments[1] == '-c':
                return '{"torch": "2.10.0+cu130", "torchvision": "0.25.0"}'
            constraint = Path(arguments[arguments.index('--constraint') + 1]).read_text()
            self.assertIn('torch==2.10.0+cu130', constraint)
            return ''
        with patch.object(setup, 'run_command', side_effect=run):
            result = setup.install_requirements(directory, self.root / 'comfy-python')
        self.assertEqual(result['state'], 'installed')
        self.assertEqual(str(commands[1][0]), str(self.root / 'comfy-python'))
        self.assertNotIn('--upgrade', commands[1])

    def test_check_reads_only_three_api_endpoints_and_reports_filename_only(self):
        bodies = {'system_stats': {'system': {'comfyui_version': '0.45.3'}},
                  'queue': {'queue_running': [], 'queue_pending': []},
                  'object_info': {'UNETLoader': {'input': {'required': {'unet_name': [['model.safetensors']]}}}}}
        calls = []
        def read(url, **kwargs):
            calls.append(url)
            return Response(json.dumps(bodies[url.rsplit('/', 1)[-1]]).encode())
        with patch.object(setup.urllib.request, 'urlopen', side_effect=read):
            result = setup.server_check('http://127.0.0.1:8188', ['UNETLoader'],
                                        {'test': {**self.model, 'runtime_name': 'model.safetensors'}})
        self.assertEqual(result['state'], 'ready')
        self.assertTrue(result['idle'])
        self.assertEqual(len(calls), 3)
        self.assertFalse(any('/prompt' in url for url in calls))


if __name__ == '__main__':
    unittest.main()
