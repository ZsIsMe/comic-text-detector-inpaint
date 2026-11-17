"""Workflow and recovery contracts exercised through mocked HTTP, without models."""
import copy
from email import policy
from email.parser import BytesParser
import io
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import urllib.parse

import cv2
import numpy as np

import comfy_repair as repair


def registry():
    nodes = {}
    for method in repair.METHOD_IDS:
        manifest, graph, _, _ = repair._definition(method)
        for node in graph.values():
            schema = nodes.setdefault(node['class_type'], {'input': {'required': {}, 'optional': {}}})
            for field in node['inputs']:
                if field.startswith('images.'):
                    schema['input']['required']['images'] = ['COMFY_AUTOGROW_V3', {
                        'template': {'input': {'required': {'image': ['IMAGE', {}]}},
                                     'names': ['image_' + str(n) for n in range(1, 17)], 'min': 0}}]
                else:
                    schema['input']['required'][field] = ['*']
        for role, node in manifest['model_nodes'].items():
            field = nodes[node['class_type']]['input']['required'].get(node['input'])
            choices = field[0] if field and isinstance(field[0], list) else []
            nodes[node['class_type']]['input']['required'][node['input']] = [choices + [node['filename']]]
    # VAEs and CLIPs in the two workflows have the same schemas but different
    # dropdown files. Accumulate both choices explicitly after the schema pass.
    for method in repair.METHOD_IDS:
        manifest = repair._definition(method)[0]
        for node in manifest['model_nodes'].values():
            choices = nodes[node['class_type']]['input']['required'][node['input']][0]
            if node['filename'] not in choices:
                choices.append(node['filename'])
    return nodes


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FakeComfy:
    def __init__(self, generated):
        self.generated = generated.copy()
        self.object_info = registry()
        self.requests = []
        self.uploads = []
        self.prompts = {}
        self.last_payload = None
        self.submit_count = 0
        self.prompt_timeout = False
        self.accept_unknown = True
        self.complete = True
        self.poll_error_once = False
        self.execution_error = False
        self.image_count = 1
        self.view_bytes = None

    def history_entry(self, prompt_id):
        payload = self.prompts[prompt_id]
        graph = payload['prompt']
        output_node = next(node_id for node_id, node in graph.items() if node['class_type'] == 'SaveImage')
        return {'prompt': [0, prompt_id, graph, payload['extra_data'], [output_node]],
                'status': {'completed': True, 'status_str': 'error' if self.execution_error else 'success',
                           'messages': [['execution_error', {'exception_message': 'fake inference failed'}]]
                           if self.execution_error else []},
                'outputs': {output_node: {'images': [
                    {'filename': 'generated.png', 'subfolder': 'actual-results', 'type': 'output'}
                ] * self.image_count}}}

    def urlopen(self, request, timeout):
        parsed = urllib.parse.urlsplit(request.full_url)
        path, method = parsed.path, request.get_method()
        self.requests.append((method, path))
        if method == 'GET' and path == '/system_stats':
            value = {'system': {'comfyui_version': '0.45.3'}}
        elif method == 'GET' and path == '/object_info':
            value = self.object_info
        elif method == 'POST' and path == '/upload/image':
            message = BytesParser(policy=policy.default).parsebytes(
                ('MIME-Version: 1.0\r\nContent-Type: ' + request.get_header('Content-type')
                 + '\r\n\r\n').encode() + request.data)
            fields = {part.get_param('name', header='content-disposition'): part for part in message.iter_parts()}
            self.uploads.append({name: part.get_payload(decode=True) for name, part in fields.items()})
            value = {'name': f'remote-{len(self.uploads)}.png', 'subfolder': 'renamed/uploads', 'type': 'input'}
        elif method == 'POST' and path == '/prompt':
            self.submit_count += 1
            self.last_payload = json.loads(request.data)
            prompt_id = 'server-returned-' + str(self.submit_count)
            if not self.prompt_timeout or self.accept_unknown:
                self.prompts[prompt_id] = self.last_payload
            if self.prompt_timeout:
                raise TimeoutError('POST response timed out after possible acceptance')
            value = {'prompt_id': prompt_id, 'node_errors': {}}
        elif method == 'GET' and path == '/queue':
            value = {'queue_running': [[0, prompt_id, payload['prompt'], payload['extra_data'], []]
                                       for prompt_id, payload in self.prompts.items()] if not self.complete else [],
                     'queue_pending': []}
        elif method == 'GET' and path == '/history':
            value = {prompt_id: self.history_entry(prompt_id) for prompt_id in self.prompts} if self.complete else {}
        elif method == 'GET' and path.startswith('/history/'):
            if self.poll_error_once:
                self.poll_error_once = False
                raise TimeoutError('transient history timeout')
            prompt_id = urllib.parse.unquote(path[len('/history/'):])
            value = ({prompt_id: self.history_entry(prompt_id)}
                     if self.complete and prompt_id in self.prompts else {})
        elif method == 'GET' and path == '/view':
            self.view_query = urllib.parse.parse_qs(parsed.query)
            return Response(self.view_bytes if self.view_bytes is not None
                            else repair._png_bytes(self.generated, rgb=True))
        else:
            raise AssertionError(f'Unexpected HTTP operation: {method} {path}')
        return Response(json.dumps(value).encode())


class ComfyRepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.original = np.full((64, 80, 3), 100, np.uint8)
        self.other = np.zeros((64, 80), np.uint8)
        self.other[5:7, 5:7] = 255
        self.generated = self.original.copy()
        self.generated[30:36, 40:46] = (180, 140, 120)
        self.fake = FakeComfy(self.generated)
        self.http = patch.object(repair.urllib.request, 'urlopen', side_effect=self.fake.urlopen)
        self.http.start()
        self.addCleanup(self.http.stop)
        self.client = repair.ComfyRepairClient(poll_interval=0, task_timeout=1)
        self.method = repair.METHOD_IDS[0]

    def run_page(self, **kwargs):
        return self.client.run(self.method, self.original, self.other, self.root, key='page.01', **kwargs)

    def job_path(self):
        return self.root / 'workflow_jobs' / self.method / 'page.01.json'

    def test_bundled_workflows_preserve_tested_models_prompts_and_sampler_parameters(self):
        _, klein, _, _ = repair._definition(repair.METHOD_IDS[0])
        _, qwen, _, _ = repair._definition(repair.METHOD_IDS[1])
        self.assertEqual(klein['269']['inputs']['unet_name'], 'flux-2-klein-9b-fp8.safetensors')
        self.assertNotIn('base', klein['269']['inputs']['unet_name'])
        self.assertEqual(klein['268']['inputs']['clip_name'], 'qwen_3_8b_fp8mixed.safetensors')
        self.assertEqual(klein['272']['inputs']['vae_name'], 'flux2-vae.safetensors')
        self.assertEqual((klein['318']['inputs']['steps'], klein['318']['inputs']['cfg'],
                          klein['318']['inputs']['LanPaint_NumSteps']), (4, 1, 2))
        self.assertEqual((qwen['13']['inputs']['steps'], qwen['13']['inputs']['cfg']), (25, 1))
        for graph, sampler, grow in ((klein, '318', '401'), (qwen, '13', '6')):
            self.assertEqual(graph[sampler]['inputs']['sampler_name'], 'euler')
            self.assertEqual(graph[sampler]['inputs']['scheduler'], 'simple')
            self.assertEqual(graph[grow]['inputs']['expand'], 8)
            self.assertTrue(graph[grow]['inputs']['tapered_corners'])
            self.assertNotIn('14_original', json.dumps(graph))
        self.assertEqual(klein['281']['inputs']['text'], qwen['12']['inputs']['prompt'])

    def test_method_and_model_map_validation_and_readiness_do_not_claim_hash_identity(self):
        for methods in ('klein9b_fp8_lanpaint', ['unknown'], [self.method, self.method]):
            with self.assertRaises(ValueError):
                repair.validate_methods(methods)
        self.assertEqual(repair.validate_methods([]), ())
        with self.assertRaises(ValueError):
            repair.ComfyRepairClient(model_map={self.method: {'checkpoint': 'x'}})
        with self.assertRaises(ValueError):
            repair.ComfyRepairClient(poll_interval=-.1)
        report = self.client.check(repair.METHOD_IDS)
        self.assertTrue(report['ready'], report)
        self.assertFalse(report['model_identity_verified'])
        self.fake.object_info.pop('LanPaint_KSampler')
        self.fake.object_info['UNETLoader']['input']['required']['unet_name'] = [['not-installed.safetensors']]
        self.fake.object_info['GrowMask']['input']['required'].pop('expand')
        report = self.client.check((self.method,))
        self.assertFalse(report['ready'])
        self.assertIn('LanPaint_KSampler', report['missing_nodes'])
        self.assertTrue(report['missing_models'])
        self.assertTrue(report['missing_inputs'])
        with self.assertRaisesRegex(repair.ComfyRepairError, 'not ready'):
            self.run_page()
        self.assertEqual(self.fake.submit_count, 0)

    def test_method_expected_model_identities_match_authoritative_setup_metadata(self):
        setup_path = repair.WORKFLOW_ROOT / 'setup_manifest.json'
        setup = json.loads(setup_path.read_text(encoding='utf-8'))
        seen = set()
        for method in repair.METHOD_IDS:
            manifest = repair._definition(method)[0]
            self.assertEqual(manifest['_setup_manifest_sha256'], repair._digest(setup_path.read_bytes()))
            for role, node in manifest['model_nodes'].items():
                identity = setup['models'][node['setup_model_id']]
                self.assertIn(node['setup_model_id'], setup['methods'][method]['models'])
                seen.add(node['setup_model_id'])
                self.assertEqual(node['filename'], identity['filename'])
                self.assertEqual(node['class_type'], identity['loader'])
                self.assertEqual(node['input'], identity['loader_input'])
                for field in ('sha256', 'size', 'category', 'sources'):
                    self.assertEqual(node[field], identity[field], (method, role, field))
                self.assertRegex(node['sha256'], r'^[0-9a-f]{64}$')
                self.assertGreater(node['size'], 0)
                hf = node['sources']['huggingface']
                self.assertIn('/resolve/' + hf['revision'] + '/', hf['url'])
                self.assertFalse(node['identity_verified'])
        self.assertEqual(len(seen), 6)

    def test_setup_manifest_bytes_change_invalidates_cache_without_reading_model_weights(self):
        self.run_page()
        before = repair._load_job(self.job_path())
        setup_path = (repair.WORKFLOW_ROOT / 'setup_manifest.json').resolve()
        original_read = Path.read_bytes
        reads = []

        def changed_read(path):
            reads.append(path)
            data = original_read(path)
            return data + b'\n' if path.resolve() == setup_path else data

        with patch.object(Path, 'read_bytes', changed_read):
            self.assertFalse(self.run_page()['reused'])
        after = repair._load_job(self.job_path())
        self.assertNotEqual(before['dependency_setup_manifest_sha256'], after['dependency_setup_manifest_sha256'])
        self.assertNotEqual(before['fingerprint'], after['fingerprint'])
        self.assertEqual(self.fake.submit_count, 2)
        self.assertFalse(any(path.suffix in ('.safetensors', '.ckpt', '.pt') for path in reads))

    def test_qwen_dynamic_input_fields_are_explicitly_validated(self):
        self.assertTrue(self.client.check((repair.METHOD_IDS[1],))['ready'])
        spec = self.fake.object_info['TextEncodeQwenImage21']['input']['required']['images']
        spec[1]['template']['names'] = ['image_1']
        report = self.client.check((repair.METHOD_IDS[1],))
        self.assertFalse(report['ready'])
        self.assertTrue(any(v['input'] == 'images.image_2' for v in report['missing_inputs']))

    def test_multipart_upload_uses_returned_names_raw_mask_and_model_role_overrides(self):
        override = 'installed/renamed-klein.safetensors'
        self.fake.object_info['UNETLoader']['input']['required']['unet_name'][0].append(override)
        self.client = repair.ComfyRepairClient(model_map={self.method: {'unet': override}}, poll_interval=0)
        before, mask_before = self.original.copy(), self.other.copy()
        protected = np.zeros_like(self.other, bool)
        protected[30:34, 40:44] = True
        result = self.run_page(protected_mask=protected)
        self.assertEqual(len(self.fake.uploads), 2)
        source_png = cv2.imdecode(np.frombuffer(self.fake.uploads[0]['image'], np.uint8), cv2.IMREAD_COLOR)
        np.testing.assert_array_equal(source_png[:, :, ::-1], self.original)
        mask_png = cv2.imdecode(np.frombuffer(self.fake.uploads[1]['image'], np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(mask_png, self.other)  # No second GrowMask in the runner.
        graph = self.fake.last_payload['prompt']
        self.assertEqual(graph['297']['inputs']['image'], 'renamed/uploads/remote-1.png')
        self.assertEqual(graph['304']['inputs']['image'], 'renamed/uploads/remote-2.png')
        self.assertEqual(graph['269']['inputs']['unet_name'], override)
        self.assertNotEqual(self.fake.last_payload['prompt_id'], result['prompt_id'])
        self.assertEqual(result['prompt_id'], 'server-returned-1')
        diff = repair._read_png(result['mask_path'])
        rgba_bgra = repair._read_png(result['rgba_path'])
        np.testing.assert_array_equal(rgba_bgra[:, :, 3], diff)
        self.assertFalse(np.any(rgba_bgra[diff == 0]))
        np.testing.assert_array_equal(rgba_bgra[diff > 0, :3][:, ::-1], self.generated[diff > 0])
        self.assertFalse(np.any(diff[protected]))
        self.assertGreater(int(diff[30:36, 40:46].sum()), 0)
        self.assertFalse(np.any(self.other[diff > 0]))  # Difference deliberately is not intersected with OTHER.
        np.testing.assert_array_equal(self.original, before)
        np.testing.assert_array_equal(self.other, mask_before)
        self.assertEqual(self.fake.view_query['subfolder'], ['actual-results'])
        self.assertTrue(Path(result['manifest_path']).is_relative_to(self.root))

    def test_difference_threshold_area_and_dilation_follow_verified_rule(self):
        generated = self.original.copy()
        generated[10:13, 10:15, 0] += 12  # area 15 removed
        generated[35:39, 50:54, 1] += 12  # area 16 retained
        generated[50:55, 10:15, 2] += 11  # below RGB threshold
        diff = repair.difference_mask(self.original, generated)
        self.assertFalse(np.any(diff[10:13, 10:15]))
        self.assertFalse(np.any(diff[50:55, 10:15]))
        self.assertTrue(np.all(diff[35:39, 50:54] == 255))
        self.assertEqual(int(diff[27, 51]), 255)  # ellipse radius 8
        self.assertEqual(int(diff[26, 51]), 0)

    def test_empty_other_performs_no_http_and_writes_no_empty_png(self):
        self.other[:] = 0
        result = self.run_page()
        self.assertTrue(result['empty'])
        self.assertEqual(result['empty_reason'], 'no_other')
        self.assertIsNone(result['rgba_path'])
        self.assertIsNone(result['mask_path'])
        self.assertIsNone(result['result_path'])
        self.assertFalse(self.fake.requests)
        self.assertFalse(list(self.root.rglob('*.png')))
        self.assertTrue(self.run_page()['reused'])

    def test_empty_difference_or_fully_protected_changes_omit_transparent_png(self):
        self.fake.generated = self.original.copy()
        result = self.run_page()
        self.assertTrue(result['empty'])
        self.assertIsNone(result['rgba_path'])
        self.assertIsNone(result['mask_path'])
        self.assertTrue(Path(result['result_path']).exists())
        requests = len(self.fake.requests)
        self.assertTrue(self.run_page()['reused'])
        self.assertEqual(len(self.fake.requests), requests)
        self.fake.generated = self.generated
        result = self.run_page(protected_mask=np.ones_like(self.other, bool))
        self.assertTrue(result['empty'])
        self.assertIsNone(result['rgba_path'])

    def test_cache_hashes_are_checked_and_corrupt_png_is_retrieved_without_new_prompt(self):
        first = self.run_page()
        requests = len(self.fake.requests)
        self.assertTrue(self.run_page()['reused'])
        self.assertEqual(len(self.fake.requests), requests)
        Path(first['rgba_path']).write_bytes(b'corrupt cache')
        recovered = self.run_page()
        self.assertFalse(recovered['reused'])
        self.assertEqual(self.fake.submit_count, 1)
        self.assertEqual(len(self.fake.uploads), 2)
        self.assertEqual(repair._read_png(recovered['rgba_path']).shape, (64, 80, 4))
        Path(recovered['mask_path']).unlink()
        self.run_page()
        self.assertEqual(self.fake.submit_count, 1)
        job = json.loads(self.job_path().read_text())
        job['seed'] += 1
        self.job_path().write_text(json.dumps(job))
        with self.assertRaisesRegex(repair.ComfyRepairError, 'Cannot trust saved job'):
            self.run_page()
        self.assertEqual(self.fake.submit_count, 1)

    def test_changed_rgb_protection_workflow_and_server_do_not_reuse_old_result(self):
        self.run_page()
        self.original[0, 0] = 101
        self.run_page()
        self.assertEqual(self.fake.submit_count, 2)
        protected = np.zeros_like(self.other, bool)
        protected[0, 0] = True
        self.run_page(protected_mask=protected)
        self.assertEqual(self.fake.submit_count, 3)
        original_definition = repair._definition

        def changed(method):
            manifest, graph, manifest_hash, workflow_hash = original_definition(method)
            return manifest, graph, manifest_hash, 'changed-workflow-hash'

        with patch.object(repair, '_definition', side_effect=changed):
            self.run_page(protected_mask=protected)
        self.assertEqual(self.fake.submit_count, 4)
        self.client = repair.ComfyRepairClient('http://other-local-server:8188', poll_interval=0)
        self.run_page(protected_mask=protected)
        self.assertEqual(self.fake.submit_count, 5)

    def test_post_timeout_is_durable_and_recovers_changed_server_id_without_resubmission(self):
        self.fake.prompt_timeout = True
        with self.assertRaises(repair.SubmissionUnknownError):
            self.run_page()
        job = repair._load_job(self.job_path())
        self.assertEqual(job['status'], 'submission_unknown')
        self.assertTrue(job['attempt_id'])
        self.assertTrue(job['client_id'])
        self.assertTrue(job['requested_prompt_id'])
        result = self.run_page()
        self.assertEqual(result['prompt_id'], 'server-returned-1')
        self.assertEqual(self.fake.submit_count, 1)
        self.assertEqual(len(self.fake.uploads), 2)

    def test_unlocatable_timeout_never_blindly_posts_again(self):
        self.fake.prompt_timeout, self.fake.accept_unknown = True, False
        with self.assertRaises(repair.SubmissionUnknownError):
            self.run_page()
        with self.assertRaisesRegex(repair.SubmissionUnknownError, 'not be submitted again'):
            self.run_page()
        self.assertEqual(self.fake.submit_count, 1)
        self.assertEqual(repair._load_job(self.job_path())['status'], 'submission_unknown')
        # Changing inputs cannot silently discard an unresolved job either.
        self.original[0, 0] = 101
        with self.assertRaisesRegex(repair.ComfyRepairError, 'unresolved'):
            self.run_page()
        self.assertEqual(self.fake.submit_count, 1)

    def test_accepted_post_timeout_can_be_recovered_from_running_queue(self):
        self.fake.prompt_timeout, self.fake.complete = True, False
        with self.assertRaises(repair.SubmissionUnknownError):
            self.run_page()
        with patch.object(repair.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaisesRegex(repair.ComfyRepairError, 'pending/running'):
                self.run_page()
        job = repair._load_job(self.job_path())
        self.assertEqual(job['status'], 'submitted')
        self.assertEqual(job['prompt_id'], 'server-returned-1')
        self.fake.complete = True
        self.run_page()
        self.assertEqual(self.fake.submit_count, 1)

    def test_poll_network_failure_and_pending_timeout_resume_same_prompt(self):
        self.fake.poll_error_once = True
        with self.assertRaisesRegex(repair.ComfyRepairError, 'history'):
            self.run_page()
        self.assertEqual(repair._load_job(self.job_path())['status'], 'submitted')
        self.fake.complete = False
        with patch.object(repair.time, 'monotonic', side_effect=[0, 2]):
            with self.assertRaisesRegex(repair.ComfyRepairError, 'pending/running'):
                self.run_page()
        self.fake.complete = True
        result = self.run_page()
        self.assertEqual(self.fake.submit_count, 1)
        self.assertEqual(result['prompt_id'], 'server-returned-1')

    def test_execution_failure_is_explicit_and_not_repeated(self):
        self.fake.execution_error = True
        with self.assertRaisesRegex(repair.ComfyRepairError, 'fake inference failed'):
            self.run_page()
        self.assertEqual(repair._load_job(self.job_path())['status'], 'failed')
        with self.assertRaisesRegex(repair.ComfyRepairError, 'Saved ComfyUI job failed'):
            self.run_page()
        self.assertEqual(self.fake.submit_count, 1)

    def test_output_dimensions_invalid_images_and_multiple_outputs_are_rejected(self):
        self.fake.generated = np.zeros((63, 80, 3), np.uint8)
        with self.assertRaisesRegex(repair.ComfyRepairError, 'original dimensions'):
            self.run_page()
        self.fake.view_bytes = b''
        with self.assertRaisesRegex(repair.ComfyRepairError, 'cannot be decoded'):
            self.run_page()
        self.fake.view_bytes = None
        self.fake.generated = self.generated
        self.run_page()
        self.assertEqual(self.fake.submit_count, 1)
        self.original[0, 0] = 101
        self.fake.image_count = 2
        with self.assertRaisesRegex(repair.ComfyRepairError, 'exactly one image'):
            self.run_page()

    def test_cross_process_lock_is_exclusive_and_stale_lock_is_reported_not_deleted(self):
        with repair._job_lock(self.job_path()):
            with self.assertRaisesRegex(repair.ComfyRepairError, 'locked'):
                self.run_page()
        self.assertFalse(self.fake.requests)
        lock = self.job_path().with_suffix('.lock')
        lock.write_text(json.dumps({'pid': 12345, 'host': socket.gethostname()}))
        with patch.object(repair, '_pid_alive', return_value=False):
            with self.assertRaisesRegex(repair.ComfyRepairError, 'stale'):
                self.run_page()
        self.assertTrue(lock.exists())

    def test_bad_input_types_keys_and_mask_sizes_never_reach_server(self):
        for key in ('../page', 'a\\b', 'bad:stream', '', 'trailing.'):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.client.run(self.method, self.original, self.other, self.root, key=key)
        with self.assertRaises(ValueError):
            self.client.run(self.method, self.original.astype(np.float32), self.other, self.root, key='a')
        with self.assertRaises(ValueError):
            self.client.run(self.method, self.original, self.other[:-1], self.root, key='a')
        with self.assertRaises(ValueError):
            self.run_page(protected_mask=np.full(self.other.shape, np.nan))
        self.assertFalse(self.fake.requests)


if __name__ == '__main__':
    unittest.main()
