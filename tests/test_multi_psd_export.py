"""Independent repair backends and versioned PSD exports with a fake server."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from psd_tools import PSDImage

from image_io import imread, imwrite
import project_store as store
import psd_export as export
from scripts import batch_export_psd as batch
from scripts.batch_export_common import paths_for, preflight_project, raw_hashes


METHODS = ('klein9b_fp8_lanpaint', 'qwen21_int8_convrot')


def flat_lama(image, mask):
    rgba = np.zeros((*mask.shape, 4), np.uint8)
    rgba[mask > 0] = [90, 100, 110, 255]
    return rgba


class FakeClient:
    calls = []
    fail = None
    empty_method = None

    def __init__(self, server_url, model_map=None):
        self.server_url = server_url

    def check(self, methods):
        return {'ready': True}

    def run(self, method_id, image_rgb, other_mask, output_dir, *, key,
            protected_mask=None, progress=None):
        self.calls.append((method_id, key, image_rgb.copy(), other_mask.copy(), protected_mask.copy()))
        if self.fail == (key, method_id):
            raise RuntimeError('fake selected method failed')
        root = Path(output_dir)
        mask = np.zeros(other_mask.shape, np.uint8)
        mask[2:4, 2:4] = 255  # Deliberately outside OTHER; independent difference layer.
        mask[protected_mask] = 0
        if method_id == self.empty_method:
            mask[:] = 0
        rgba = np.zeros((*mask.shape, 4), np.uint8)
        color = [20, 40, 180, 255] if method_id == METHODS[0] else [180, 40, 20, 255]
        rgba[mask > 0] = color
        paths = {'rgba_path': root / method_id / f'{key}.png',
                 'mask_path': root / 'repair_masks' / method_id / f'{key}.png',
                 'result_path': root / 'workflow_results' / method_id / f'{key}.png',
                 'manifest_path': root / 'workflow_jobs' / method_id / f'{key}.json'}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        if np.any(mask):
            imwrite(paths['rgba_path'], cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            imwrite(paths['mask_path'], mask)
        else:
            paths['rgba_path'] = paths['mask_path'] = None
        imwrite(paths['result_path'], image_rgb[:, :, ::-1])
        paths['manifest_path'].write_text(json.dumps({'method': method_id, 'key': key}), 'utf-8')
        return {'method_id': method_id, 'empty': not np.any(mask),
                **{name: str(path) if path is not None else None for name, path in paths.items()}}


def validate_methods(methods):
    methods = tuple(dict.fromkeys(methods))
    if any(method not in METHODS for method in methods):
        raise ValueError('unknown repair method')
    return methods


class MultiPsdExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / 'project'
        self.folder.mkdir()
        self.source = self.folder / '01.png'
        self.base = np.full((40, 48, 3), (130, 150, 170), np.uint8)
        imwrite(self.source, self.base)
        self.paths = paths_for(self.folder)
        store.update_project(self.paths['raw'], images=[self.source.name])
        self.page = store.empty_page(self.base.shape[:2])
        self.page['overlay'][5:8, 5:8] = [15, 25, 35, 255]
        self.page['other'][20:22, 30:32] = 255
        self.page['edited'][10, 10] = 255
        store.save_page(self.paths, self.source, self.page)
        FakeClient.calls, FakeClient.fail, FakeClient.empty_method = [], None, None
        fake = types.ModuleType('comfy_repair')
        fake.METHOD_IDS = METHODS
        fake.validate_methods = validate_methods
        fake.ComfyRepairClient = FakeClient
        self.addCleanup(patch.stopall)
        patch.dict(sys.modules, {'comfy_repair': fake}).start()
        patch.object(export, 'inpaint_other', side_effect=flat_lama).start()

    def generate(self, **kwargs):
        return export.export_project_assets([self.source], self.paths, extra_methods=METHODS,
                                             export_name='trial', **kwargs)

    def test_independent_inputs_hidden_extra_layers_and_pixels_match_saved_pngs(self):
        before = raw_hashes(self.folder)
        stages = []
        result = self.generate(progress=lambda *args: stages.append(args))
        self.assertTrue(result['complete'])
        self.assertEqual(raw_hashes(self.folder), before)
        self.assertEqual(len(FakeClient.calls), 2)
        first, second = FakeClient.calls
        np.testing.assert_array_equal(first[2], second[2])
        np.testing.assert_array_equal(first[3], self.page['other'])
        self.assertEqual(first[2].dtype, np.uint8)
        np.testing.assert_array_equal(first[2][5, 5], [35, 25, 15])
        self.assertTrue(first[4][5, 5])
        self.assertTrue(first[4][10, 10])
        psd = PSDImage.open(result['created'][0])
        self.assertEqual([layer.name for layer in psd], ['bg', 'overlay-manual',
            'lama_large_512px', *METHODS])
        self.assertEqual([layer.visible for layer in psd], [True, True, True, False, False])
        for layer, method in zip(list(psd)[-2:], METHODS):
            png = imread(Path(result['assets_dir']) / method / '01.png', cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(np.rint(layer.numpy() * 255).astype(np.uint8),
                                          png[:, :, [2, 1, 0, 3]])
        self.assertEqual(stages[-1][4:], (4, 4))

    def test_16bit_background_preserved_and_extra_repairs_promoted_to_depth(self):
        imwrite(self.source, self.base.astype(np.uint16) * 257)
        store.save_page(self.paths, self.source, self.page)
        result = self.generate()
        psd = PSDImage.open(result['created'][0])
        self.assertEqual(psd.depth, 16)
        actual = np.rint(psd[0].numpy('color') * 65535).astype(np.uint16)
        np.testing.assert_array_equal(actual, self.base[:, :, ::-1].astype(np.uint16) * 257)
        self.assertTrue(all(call[2].dtype == np.uint8 for call in FakeClient.calls))
        self.assertEqual(int(np.rint(psd[-1].numpy('shape')[2, 2, 0] * 65535)), 65535)

    def test_empty_other_does_not_construct_server_or_create_empty_layers(self):
        store.save_page(self.paths, self.source, store.empty_page(self.base.shape[:2]))
        with patch.object(FakeClient, '__init__', side_effect=AssertionError('empty must not connect')):
            result = self.generate()
        self.assertTrue(result['complete'])
        self.assertEqual([p.name for p in PSDImage.open(result['created'][0])], ['bg'])
        self.assertFalse(any((Path(result['assets_dir']) / m).exists() for m in METHODS))

    def test_empty_difference_omits_only_that_method_layer(self):
        FakeClient.empty_method = METHODS[1]
        result = self.generate()
        self.assertTrue(result['complete'])
        self.assertNotIn(METHODS[1], [p.name for p in PSDImage.open(result['created'][0])])
        self.assertFalse((Path(result['assets_dir']) / METHODS[1] / '01.png').exists())

    def test_failure_page_does_not_get_incomplete_psd_and_other_pages_continue(self):
        second = self.folder / '02.png'
        imwrite(second, self.base)
        store.update_project(self.paths['raw'], images=['01.png', '02.png'])
        store.save_page(self.paths, second, self.page)
        FakeClient.fail = ('01', METHODS[1])
        result = export.export_project_assets([self.source, second], self.paths,
            extra_methods=METHODS, export_name='failure')
        self.assertFalse(result['complete'])
        self.assertEqual(result['failed_pages'], ['01.png'])
        self.assertEqual(len(result['created']), 1)
        self.assertTrue(result['created'][0].endswith('02.psd'))
        root = Path(result['assets_dir'])
        self.assertFalse((root / 'psds' / '01.psd').exists())
        self.assertTrue((root / METHODS[0] / '01.png').exists())

    def test_all_pages_all_method_materials_finish_before_first_psd(self):
        second = self.folder / '02.png'
        imwrite(second, self.base)
        store.update_project(self.paths['raw'], images=['01.png', '02.png'])
        store.save_page(self.paths, second, self.page)
        build = export.export_page_psd

        def checked_build(*args, **kwargs):
            self.assertEqual(len(FakeClient.calls), 4)
            return build(*args, **kwargs)

        with patch.object(export, 'export_page_psd', side_effect=checked_build):
            result = export.export_project_assets([self.source, second], self.paths,
                extra_methods=METHODS, export_name='phases')
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['created']), 2)

    def test_invalid_extra_alpha_fails_page_instead_of_assembling_psd(self):
        run = FakeClient.run

        def invalid(client, *args, **kwargs):
            response = run(client, *args, **kwargs)
            path = response['rgba_path']
            pixels = imread(path, cv2.IMREAD_UNCHANGED)
            pixels[0, 0] = [10, 20, 30, 255]  # Outside its recorded difference mask.
            imwrite(path, pixels)
            return response

        with patch.object(FakeClient, 'run', invalid):
            result = self.generate()
        self.assertFalse(result['complete'])
        self.assertEqual(result['failed_pages'], ['01.png'])
        self.assertEqual(result['created'], [])

    def test_resuming_same_run_certifies_unchanged_psd_and_preserves_hand_edits_as_stale(self):
        first = self.generate()
        path = Path(first['created'][0])
        before = path.read_bytes()
        repeated = self.generate()
        self.assertTrue(repeated['complete'])
        self.assertEqual(repeated['created'], [])
        self.assertEqual(path.read_bytes(), before)
        psd = PSDImage.open(path)
        psd[0].name = 'hand-edited'
        psd.save(path)
        edited = path.read_bytes()
        stale = self.generate()
        self.assertFalse(stale['complete'])
        self.assertEqual(stale['preserved_stale'], [str(path)])
        self.assertEqual(path.read_bytes(), edited)

    def test_assets_entry_can_resume_returned_run_directory_with_raw_protection(self):
        first = self.generate()
        result = export.export_psds_from_assets([self.source], first['assets_dir'],
            extra_methods=METHODS, export_name='trial')
        self.assertTrue(result['complete'])
        self.assertEqual(result['assets_dir'], first['assets_dir'])
        self.assertTrue(FakeClient.calls[-1][4][10, 10])

    def test_default_versions_are_isolated_and_invalid_or_changed_names_are_rejected(self):
        baseline = Path(self.paths['output']) / 'psd_assets' / 'psds' / '01.psd'
        baseline.parent.mkdir(parents=True)
        baseline.write_bytes(b'baseline hand edits')
        a = export.export_project_assets([self.source], self.paths, extra_methods=METHODS)
        b = export.export_project_assets([self.source], self.paths, extra_methods=METHODS)
        self.assertNotEqual(a['assets_dir'], b['assets_dir'])
        self.assertEqual(baseline.read_bytes(), b'baseline hand edits')
        self.generate()
        with self.assertRaisesRegex(ValueError, '另一組'):
            export.export_project_assets([self.source], self.paths, extra_methods=[METHODS[0]],
                                        export_name='trial')
        for invalid in ('../escape', '.', 'a/b', 'a\\b', ''):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                export.export_project_assets([self.source], self.paths, extra_methods=METHODS,
                                            export_name=invalid)

    def test_cli_enhanced_selection_resume_signature_and_corruption(self):
        options = {'extra_methods': METHODS, 'export_name': 'cli',
                   'comfy_url': 'http://localhost:8188', 'model_map': {'unet': 'model.safetensors'}}
        images = preflight_project(self.folder)
        record = batch.export_project(self.folder, images, **options)
        self.assertEqual(record['status'], 'success')
        self.assertTrue(batch.verified_resume(self.folder, record, **options))
        self.assertFalse(batch.verified_resume(self.folder, record, **dict(options, comfy_url='http://other:8188')))
        self.assertFalse(batch.verified_resume(self.folder, record, **dict(options, extra_methods=[METHODS[0]])))
        self.assertFalse(batch.verified_resume(self.folder, record, **dict(options, model_map={'unet': 'changed'})))
        artifact = Path(record['output']) / METHODS[0] / '01.png'
        artifact.write_bytes(b'corrupt')
        self.assertFalse(batch.verified_resume(self.folder, record, **options))

    def test_cli_parser_passes_selection_and_model_map(self):
        model_map = self.root / 'models.json'
        model_map.write_text(json.dumps({'unet': 'local.safetensors'}), 'utf-8')
        with patch.object(batch, 'run', return_value=0) as run:
            self.assertEqual(batch.main(['--root', str(self.folder), '--report-dir', str(self.root / 'reports'),
                '--extra-methods', *METHODS, '--comfy-url', 'http://remote:8188',
                '--model-map', str(model_map), '--export-name', 'cli']), 0)
        self.assertEqual(run.call_args.kwargs['extra_methods'], list(METHODS))
        self.assertEqual(run.call_args.kwargs['model_map'], {'unet': 'local.safetensors'})

    def test_only_selected_workflow_assets_and_authoritative_setup_affect_signature(self):
        workflows = self.root / 'workflows'
        shutil.copytree(export.REPAIR_WORKFLOW_ROOT, workflows)
        with patch.object(export, 'REPAIR_WORKFLOW_ROOT', workflows):
            first = export.export_signature([METHODS[0]])
            unselected = workflows / METHODS[1] / 'manifest.json'
            unselected.write_bytes(unselected.read_bytes() + b'\n')
            self.assertEqual(export.export_signature([METHODS[0]]), first)
            for path in (workflows / METHODS[0] / 'workflow_api.json',
                         workflows / METHODS[0] / 'manifest.json',
                         workflows / 'setup_manifest.json'):
                original = path.read_bytes()
                value = json.loads(original)
                value['signature_test_revision'] = 'changed authoritative content'
                path.write_text(json.dumps(value, ensure_ascii=False), 'utf-8')
                with self.subTest(path=path):
                    changed = export.export_signature([METHODS[0]])
                    self.assertNotEqual(changed, first)
                path.write_bytes(original)
            self.assertEqual(export.export_signature([METHODS[0]]), first)

    def test_metadata_change_invalidates_resume_and_named_export_before_overwriting_materials(self):
        workflows = self.root / 'workflows'
        shutil.copytree(export.REPAIR_WORKFLOW_ROOT, workflows)
        options = {'extra_methods': METHODS, 'export_name': 'metadata'}
        with patch.object(export, 'REPAIR_WORKFLOW_ROOT', workflows):
            record = batch.export_project(self.folder, preflight_project(self.folder), **options)
            self.assertTrue(batch.verified_resume(self.folder, record, **options))
            material = Path(record['output']) / 'solid' / '01.png'
            before = material.read_bytes()
            metadata = workflows / 'setup_manifest.json'
            value = json.loads(metadata.read_text('utf-8'))
            value['signature_test_revision'] = 'new dependency pins'
            metadata.write_text(json.dumps(value), 'utf-8')
            self.assertFalse(batch.verified_resume(self.folder, record, **options))
            with self.assertRaisesRegex(ValueError, '另一組'):
                export.export_project_assets([self.source], self.paths, **options)
            self.assertEqual(material.read_bytes(), before)

    def test_no_extra_signature_does_not_import_or_require_comfy(self):
        with patch.dict(sys.modules, {'comfy_repair': None}), \
                patch.object(export, '_hash_file', side_effect=AssertionError('no workflow files needed')):
            signature = export.export_signature()
        self.assertEqual(signature['workflow_assets'], {})


if __name__ == '__main__':
    unittest.main()
