"""Synthetic checks for the skill's cooperative review-session helpers."""

import asyncio
import base64
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / ".agents/skills/comic-detect-review/scripts"
spec = importlib.util.spec_from_file_location("review_session", SCRIPTS / "review_session.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class ReviewSessionTests(unittest.TestCase):
    def test_batch_remainder_under_twenty_stays_together(self):
        cases = {39: [10, 10, 19], 31: [10, 10, 11], 30: [10, 10, 10],
                 21: [10, 11], 20: [10, 10], 19: [19], 201: [10] * 19 + [11]}
        for count, lengths in cases.items():
            groups = review.batches(list(range(1, count + 1)))
            self.assertEqual([len(g) for g in groups], lengths)
            self.assertEqual([page for group in groups for page in group], list(range(1, count + 1)))
        self.assertEqual(review.batches([7]), [[7]])

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
                review.atomic_json(path, state)
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
                    binding=str(path), tool="get_review_bundle", params=None)))
                    for path, _ in states]
                self.assertNotEqual(outputs[0]["output_dir"], outputs[1]["output_dir"])
                for output, (_, state) in zip(outputs, states):
                    self.assertEqual(output["metadata"]["page_token"], state["session"] + "-token")
                    self.assertEqual(Path(output["images"][0]).read_bytes(), b"synthetic png bytes")

                override = root / "override.json"
                override.write_text(json.dumps({"instance_id": "b" * 32}))
                with self.assertRaisesRegex(ValueError, "must not contain instance_id"):
                    asyncio.run(review.call(SimpleNamespace(
                        binding=str(states[0][0]), tool="get_review_bundle", params=str(override))))

                wrong_token = root / "wrong-token.json"
                wrong_token.write_text(json.dumps({"page_token": "b-token", "operations": []}))
                with self.assertRaisesRegex(ValueError, "not the latest bundle"):
                    asyncio.run(review.call(SimpleNamespace(
                        binding=str(states[0][0]), tool="apply_page_edits", params=str(wrong_token))))

                states[0][1]["folder"] = str(folder_b)
                with self.assertRaisesRegex(RuntimeError, "folder changed"):
                    asyncio.run(review.call(SimpleNamespace(
                        binding=str(states[0][0]), tool="get_review_bundle", params=None)))
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
            review.atomic_json(path, {"session": "s", "instance_id": "a" * 32,
                                      "pid": 42, "folder": str(folder)})

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
                        binding=str(path), tool="get_review_bundle", params=None)))
            self.assertFalse((path.parent / review.folder_key(str(folder))).exists())

    def test_explicit_switch_updates_folder_and_output_namespace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            path = root / "session" / ("a" * 32) / "binding.json"
            review.atomic_json(path, {"session": "session", "instance_id": "a" * 32,
                                      "pid": 42, "folder": str(first)})

            async def connect(op):
                return await op(object())

            async def tool(_session, name, params):
                self.assertEqual(params["instance_id"], "a" * 32)
                return {"instance_id": "a" * 32, "pid": 42,
                        "folder": str(second if name == "open_project" else first)}, []

            with mock.patch.object(review, "_connect_and_run", connect), \
                 mock.patch.object(review, "_tool", tool):
                result = asyncio.run(review.switch(SimpleNamespace(binding=str(path), folder=str(second))))
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
