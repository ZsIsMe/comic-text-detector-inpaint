"""Synthetic checks for the skill's cooperative review-session helpers."""

import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / ".agents/skills/comic-detect-review/scripts"
sys.path.insert(0, str(SCRIPTS))
import coordination
spec = importlib.util.spec_from_file_location("review_session", SCRIPTS / "review_session.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def claimed_binding(path, state, coord_root):
    with coordination.file_lock(coordination.instance_lock_path(state['instance_id'], coord_root)):
        claim = coordination.create_claim(state, path, coord_root)
        review.atomic_json(path, {**state, 'claim_token': claim['claim_token']})


class ReviewSessionTests(unittest.TestCase):
    def test_default_twenty_page_batches_keep_tail_under_forty_together(self):
        cases = {0: [], 1: [1], 19: [19], 20: [20], 21: [21], 39: [39],
                 40: [20, 20], 41: [20, 21], 59: [20, 39], 60: [20, 20, 20],
                 61: [20, 20, 21], 200: [20] * 10, 201: [20] * 9 + [21]}
        for count, lengths in cases.items():
            groups = review.batches(list(range(1, count + 1)))
            self.assertEqual([len(g) for g in groups], lengths)
            self.assertEqual([page for group in groups for page in group], list(range(1, count + 1)))
        self.assertEqual(review.batches([7]), [[7]])
        self.assertEqual([len(g) for g in review.batches(list(range(1, 40)), size=10)], [10, 10, 19])
        for invalid in ([1, 1], [2, 1], [0], [-1]):
            with self.assertRaises(ValueError):
                review.batches(invalid)
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(review.main(['batches', '--indices', '1,,2']), 1)
            self.assertEqual(review.main(['batches', '--indices', '']), 1)

    def test_claim_contention_idempotence_and_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'images'
            folder.mkdir()
            ident = 'a' * 32
            coord = root / 'coord'
            async def connect(op):
                return await op(object())
            async def tool(_session, name, params):
                self.assertEqual(name, 'get_app_status')
                return {'instance_id': ident, 'pid': os.getpid(), 'folder': str(folder)}, []
            def args(session):
                return SimpleNamespace(session=session, instance_id=ident, pid=os.getpid(),
                                       folder=str(folder), run_root=str(root / 'runs'), coord_root=coord)
            with mock.patch.object(review, '_connect_and_run', connect), \
                 mock.patch.object(review, '_tool', tool):
                first = asyncio.run(review.bind(args('first')))
                original = review.read_binding(Path(first['binding']))['claim_token']
                again = asyncio.run(review.bind(args('first')))
                self.assertEqual(original, review.read_binding(Path(again['binding']))['claim_token'])
                with self.assertRaisesRegex(RuntimeError, 'already claimed'):
                    asyncio.run(review.bind(args('second')))
            result = review.release(SimpleNamespace(binding=first['binding'], coord_root=coord))
            self.assertTrue(result['released'])
            self.assertFalse(Path(first['binding']).exists())
            with mock.patch.object(review, '_connect_and_run', connect), \
                 mock.patch.object(review, '_tool', tool):
                second = asyncio.run(review.bind(args('second')))
            self.assertNotEqual(original, review.read_binding(Path(second['binding']))['claim_token'])

    def test_bind_can_claim_then_open_empty_gui(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'images'
            folder.mkdir()
            ident = 'a' * 32
            current = [None]
            seen = []
            async def connect(op):
                return await op(object())
            async def tool(_session, name, params):
                seen.append(name)
                if name == 'open_project':
                    current[0] = params['path']
                return {'instance_id': ident, 'pid': 42, 'folder': current[0]}, []
            args = SimpleNamespace(session='s', instance_id=ident, pid=42,
                                   folder=str(folder), run_root=str(root / 'runs'),
                                   coord_root=root / 'coord', open_project=True)
            with mock.patch.object(review, '_connect_and_run', connect), \
                 mock.patch.object(review, '_tool', tool):
                result = asyncio.run(review.bind(args))
            self.assertEqual(seen, ['get_app_status', 'open_project', 'get_app_status'])
            self.assertEqual(review.read_binding(Path(result['binding']))['folder'], str(folder.resolve()))

    def test_reap_invalid_instance_claim_and_invalidate_legacy_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ident = 'a' * 32
            folder = root / 'images'
            folder.mkdir()
            path = root / 'session' / ident / 'binding.json'
            claimed_binding(path, {'session': 'session', 'instance_id': ident,
                                   'pid': os.getpid(), 'folder': str(folder)}, root / 'coord')
            async def missing_instances():
                return {'instances': []}
            with mock.patch.object(review, 'instances', missing_instances):
                result = asyncio.run(review.reap(SimpleNamespace(instance_id=ident, coord_root=root / 'coord')))
            self.assertTrue(result['reaped'])
            with self.assertRaisesRegex(RuntimeError, 'released or belongs'):
                coordination.verify_claim(review.read_binding(path), path, root / 'coord')
            legacy = review.read_binding(path)
            legacy.pop('claim_token')
            review.atomic_json(path, legacy)
            with self.assertRaisesRegex(RuntimeError, 'Legacy binding'):
                coordination.verify_claim(review.read_binding(path), path, root / 'coord')

    def test_reap_orphan_claims_but_not_valid_live_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            coord = root / 'coord'
            ident = 'a' * 32
            state = {'session': 'session', 'instance_id': ident,
                     'pid': os.getpid(), 'folder': str(root)}
            path = root / 'runs' / 'binding.json'
            with coordination.file_lock(coordination.instance_lock_path(ident, coord)):
                coordination.create_claim(state, path, coord)
            async def must_not_query():
                self.fail('Orphan claim should be reaped without GUI discovery')
            with mock.patch.object(review, 'instances', must_not_query):
                missing = asyncio.run(review.reap(SimpleNamespace(instance_id=ident, coord_root=coord)))
            self.assertTrue(missing['orphan'])

            claimed_binding(path, state, coord)
            stale = review.read_binding(path)
            stale['claim_token'] = 'old-token'
            review.atomic_json(path, stale)
            with mock.patch.object(review, 'instances', must_not_query):
                old_token = asyncio.run(review.reap(SimpleNamespace(instance_id=ident, coord_root=coord)))
            self.assertTrue(old_token['orphan'])

            claimed_binding(path, state, coord)
            async def live_instances():
                return {'instances': [{'instance_id': ident, 'pid': os.getpid()}]}
            with mock.patch.object(review, 'instances', live_instances):
                with self.assertRaisesRegex(RuntimeError, 'still be active'):
                    asyncio.run(review.reap(SimpleNamespace(instance_id=ident, coord_root=coord)))
            self.assertIsNotNone(coordination.read_claim(ident, coord))

    def test_nested_taskgroup_error_is_single_json_without_traceback(self):
        class SyntheticGroup(Exception):
            def __init__(self, *exceptions):
                super().__init__('unhelpful TaskGroup wrapper')
                self.exceptions = exceptions
        async def failed_instances():
            raise SyntheticGroup(SyntheticGroup(RuntimeError('GUI already claimed')),
                                 RuntimeError('GUI already claimed'))
        captured = io.StringIO()
        with mock.patch.object(review, 'instances', failed_instances), \
             contextlib.redirect_stderr(captured):
            exit_code = review.main(['instances'])
        self.assertEqual(exit_code, 1)
        self.assertEqual(captured.getvalue().count('\n'), 1)
        self.assertEqual(json.loads(captured.getvalue()), {'error': 'GUI already claimed'})

    def test_dead_gui_claim_can_be_reaped_without_mcp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ident = 'a' * 32
            path = root / 'binding.json'
            state = {'session': 's', 'instance_id': ident, 'pid': 99999999,
                     'folder': str(root)}
            claimed_binding(path, state, root / 'coord')
            async def no_instances():
                self.fail('Dead PID should not need MCP discovery')
            with mock.patch.object(review, 'instances', no_instances):
                result = asyncio.run(review.reap(SimpleNamespace(instance_id=ident, coord_root=root / 'coord')))
            self.assertTrue(result['reaped'])

    def test_handoff_reads_marks_without_creating_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / 'images'
            folder.mkdir()
            path = root / 'session' / ('a' * 32) / 'binding.json'
            claimed_binding(path, {'session': 'session', 'instance_id': 'a' * 32,
                                   'pid': 42, 'folder': str(folder)}, root / 'coord')
            seen = []
            async def connect(op):
                return await op(object())
            async def tool(_session, name, params):
                seen.append(name)
                self.assertEqual(params['instance_id'], 'a' * 32)
                if name == 'get_app_status':
                    return {'instance_id': 'a' * 32, 'pid': 42, 'folder': str(folder),
                            'save_pending': False, 'save_error': None}, []
                if name == 'save_project':
                    return {'saved': True, 'save_pending': False, 'save_error': None}, []
                if name == 'list_pages':
                    return {'folder': str(folder), 'page_count': 20,
                            'pages': [{'reviewed': i < 10} for i in range(20)]}, []
                self.fail('Unexpected mutating MCP call')
            with mock.patch.object(review, '_connect_and_run', connect), \
                 mock.patch.object(review, '_tool', tool):
                outcome = asyncio.run(review.handoff(SimpleNamespace(
                    binding=str(path), start_page=1, end_page=10, coord_root=root / 'coord')))
            self.assertTrue(outcome['verified'])
            self.assertEqual(outcome['next_batch'], list(range(11, 21)))
            self.assertEqual(seen, ['get_app_status', 'save_project', 'list_pages', 'get_app_status'])

    def test_switch_timeout_after_success_repairs_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / 'old', root / 'new'
            old.mkdir()
            new.mkdir()
            path = root / 'session' / ('a' * 32) / 'binding.json'
            claimed_binding(path, {'session': 'session', 'instance_id': 'a' * 32,
                                   'pid': 42, 'folder': str(old)}, root / 'coord')
            current = [str(old)]
            async def connect(op):
                return await op(object())
            async def tool(_session, name, _params):
                if name == 'open_project':
                    current[0] = str(new)
                    raise RuntimeError('simulated timeout')
                return {'instance_id': 'a' * 32, 'pid': 42, 'folder': current[0]}, []
            with mock.patch.object(review, '_connect_and_run', connect), \
                 mock.patch.object(review, '_tool', tool):
                result = asyncio.run(review.switch(SimpleNamespace(
                    binding=str(path), folder=str(new), coord_root=root / 'coord')))
            self.assertEqual(result['folder'], str(new.resolve()))
            self.assertEqual(review.read_binding(path)['folder'], str(new.resolve()))

    def test_sessions_have_separate_outputs_and_reject_wrong_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder_a, folder_b = root / "chapter-a", root / "chapter-b"
            folder_a.mkdir()
            folder_b.mkdir()
            states = []
            for name, folder, ident in (("a", folder_a, "a" * 32),
                                        ("b", folder_b, "b" * 32)):
                path = root / name / ident / "binding.json"
                state = {"session": name, "instance_id": ident, "pid": 123,
                         "folder": str(folder)}
                claimed_binding(path, state, root / 'coord')
                states.append((path, state))

            image_data = base64.b64encode(b"synthetic png bytes").decode()

            async def fake_connect(operation):
                return await operation(object())

            async def fake_tool(_session, name, params):
                state = next(s for _, s in states if s["instance_id"] == params["instance_id"])
                if name == "get_app_status":
                    return {"instance_id": state["instance_id"], "pid": 123,
                            "folder": state["folder"]}, []
                self.assertEqual(name, "get_review_bundle")
                return {"folder": state["folder"], "page_token": state["session"] + "-token",
                        "image_order": ["original", "composite"]}, [
                            SimpleNamespace(data=image_data), SimpleNamespace(data=image_data)]

            with mock.patch.object(review, "_connect_and_run", fake_connect), \
                 mock.patch.object(review, "_tool", fake_tool):
                outputs = [asyncio.run(review.call(SimpleNamespace(
                    binding=str(path), tool="get_review_bundle", params=None, coord_root=root / 'coord')))
                    for path, _ in states]
                self.assertNotEqual(outputs[0]["output_dir"], outputs[1]["output_dir"])
                for output, (_, state) in zip(outputs, states):
                    self.assertEqual(output["metadata"]["page_token"], state["session"] + "-token")
                    self.assertEqual(Path(output["images"][0]).read_bytes(), b"synthetic png bytes")

                override = root / "override.json"
                override.write_text(json.dumps({"instance_id": "b" * 32}))
                with self.assertRaisesRegex(ValueError, "must not contain instance_id"):
                    asyncio.run(review.call(SimpleNamespace(
                        binding=str(states[0][0]), tool="get_review_bundle", params=str(override), coord_root=root / 'coord')))

                wrong_token = root / "wrong-token.json"
                wrong_token.write_text(json.dumps({"page_token": "b-token", "operations": []}))
                with self.assertRaisesRegex(ValueError, "not the latest bundle"):
                    asyncio.run(review.call(SimpleNamespace(
                        binding=str(states[0][0]), tool="apply_page_edits", params=str(wrong_token), coord_root=root / 'coord')))

                states[0][1]["folder"] = str(folder_b)
                with self.assertRaisesRegex(RuntimeError, "folder changed"):
                    asyncio.run(review.call(SimpleNamespace(
                        binding=str(states[0][0]), tool="get_review_bundle", params=None, coord_root=root / 'coord')))
                saved = Path(outputs[0]["output_dir"]) / "latest-bundle.json"
                self.assertEqual(json.loads(saved.read_text())["page_token"], "a-token")

    def test_result_from_other_folder_is_not_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "chapter"
            other = root / "other"
            folder.mkdir()
            other.mkdir()
            path = root / "s" / ("a" * 32) / "binding.json"
            claimed_binding(path, {"session": "s", "instance_id": "a" * 32,
                                   "pid": 42, "folder": str(folder)}, root / 'coord')

            async def connect(op):
                return await op(object())

            async def tool(_session, name, _params):
                if name == "get_app_status":
                    return {"instance_id": "a" * 32, "pid": 42, "folder": str(folder)}, []
                return {"folder": str(other), "page_token": "wrong",
                        "image_order": ["original"]}, [SimpleNamespace(data=base64.b64encode(b"x").decode())]

            with mock.patch.object(review, "_connect_and_run", connect), \
                 mock.patch.object(review, "_tool", tool):
                with self.assertRaisesRegex(RuntimeError, "another folder"):
                    asyncio.run(review.call(SimpleNamespace(
                        binding=str(path), tool="get_review_bundle", params=None, coord_root=root / 'coord')))
            self.assertFalse((path.parent / review.folder_key(str(folder))).exists())

    def test_explicit_switch_updates_folder_and_output_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            path = root / "session" / ("a" * 32) / "binding.json"
            claimed_binding(path, {"session": "session", "instance_id": "a" * 32,
                                   "pid": 42, "folder": str(first)}, root / 'coord')
            current = [str(first)]

            async def connect(op):
                return await op(object())

            async def tool(_session, name, params):
                self.assertEqual(params["instance_id"], "a" * 32)
                if name == 'open_project':
                    current[0] = str(second)
                return {"instance_id": "a" * 32, "pid": 42,
                        "folder": current[0]}, []

            with mock.patch.object(review, "_connect_and_run", connect), \
                 mock.patch.object(review, "_tool", tool):
                result = asyncio.run(review.switch(SimpleNamespace(binding=str(path), folder=str(second), coord_root=root / 'coord')))
            canonical_second = str(second.resolve())
            self.assertEqual(review.read_binding(path)["folder"], canonical_second)
            self.assertEqual(result["output_dir"], str(path.resolve().parent / review.folder_key(canonical_second)))

    def test_desktop_lock_competes_and_process_exit_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = str(Path(tmp) / "desktop.lock")
            command = [sys.executable, str(SCRIPTS / "desktop_lock.py"), "--lock-file", lock]
            holder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True)
            try:
                self.assertTrue(json.loads(holder.stdout.readline())["locked"])
                loser = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=5)
                self.assertEqual(loser.returncode, 2)
                holder.terminate()
                holder.wait(timeout=5)
                winner = subprocess.run(command, input="\n", capture_output=True, text=True, timeout=5)
                self.assertEqual(winner.returncode, 0, winner.stderr)
            finally:
                if holder.poll() is None:
                    holder.kill()
                    holder.wait(timeout=5)
                holder.stdin.close()
                holder.stdout.close()
                holder.stderr.close()


if __name__ == "__main__":
    unittest.main()
