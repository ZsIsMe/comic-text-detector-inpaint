"""Launcher environment compatibility and migration without installing packages."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import bootstrap


class BootstrapEnvironmentTests(unittest.TestCase):
    def test_python_310_reports_the_required_version(self):
        with patch.object(bootstrap.sys, 'version_info', (3, 10)), \
                patch('sys.stderr') as stderr:
            with self.assertRaises(SystemExit) as error:
                bootstrap.ensure_python_version()
        self.assertEqual(error.exception.code, 1)
        message = ''.join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn('Python 3.11 or newer', message)

    def test_compatible_environment_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = Path(tmp) / '.venv'
            environment.mkdir()
            with patch.object(bootstrap, 'VENV_DIR', environment), \
                    patch.object(bootstrap, 'venv_python', return_value=Path(sys.executable)), \
                    patch.object(bootstrap.venv, 'EnvBuilder') as builder:
                self.assertEqual(bootstrap.ensure_venv(), Path(sys.executable))
            builder.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [environment])

    def test_incompatible_environment_is_preserved_before_recreation(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = Path(tmp) / '.venv'
            environment.mkdir()
            (environment / 'previous-packages').write_text('keep me', encoding='utf-8')
            python = environment / 'bin' / 'python'

            def create_environment(path):
                self.assertFalse(path.exists())
                python.parent.mkdir(parents=True)
                python.touch()

            with patch.object(bootstrap, 'VENV_DIR', environment), \
                    patch.object(bootstrap, 'venv_python', return_value=python), \
                    patch.object(bootstrap.subprocess, 'check_call',
                                 side_effect=subprocess.CalledProcessError(1, ['old-python'])), \
                    patch.object(bootstrap.venv, 'EnvBuilder') as builder:
                builder.return_value.create.side_effect = create_environment
                self.assertEqual(bootstrap.ensure_venv(), python)

            backups = list(Path(tmp).glob('.venv.backup-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / 'previous-packages').read_text(encoding='utf-8'), 'keep me')
            self.assertFalse((environment / 'previous-packages').exists())

    def test_failed_recreation_keeps_previous_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = Path(tmp) / '.venv'
            environment.mkdir()
            (environment / 'previous-packages').write_text('keep me', encoding='utf-8')
            with patch.object(bootstrap, 'VENV_DIR', environment), \
                    patch.object(bootstrap, 'is_venv_healthy', return_value=False), \
                    patch.object(bootstrap.venv, 'EnvBuilder') as builder, \
                    patch('sys.stderr'):
                builder.return_value.create.side_effect = OSError('cannot create environment')
                with self.assertRaises(SystemExit):
                    bootstrap.ensure_venv()
            backups = list(Path(tmp).glob('.venv.backup-*'))
            self.assertEqual(len(backups), 1)
            self.assertEqual((backups[0] / 'previous-packages').read_text(encoding='utf-8'), 'keep me')

    @unittest.skipIf(sys.platform == 'win32', 'symlink creation may require administrator privileges')
    def test_broken_environment_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            environment = Path(tmp) / '.venv'
            missing_target = Path(tmp) / 'missing-environment'
            environment.symlink_to(missing_target, target_is_directory=True)
            python = environment / 'bin' / 'python'

            def create_environment(path):
                python.parent.mkdir(parents=True)
                python.touch()

            with patch.object(bootstrap, 'VENV_DIR', environment), \
                    patch.object(bootstrap.venv, 'EnvBuilder') as builder:
                builder.return_value.create.side_effect = create_environment
                self.assertEqual(bootstrap.ensure_venv(), python)

            backups = list(Path(tmp).glob('.venv.backup-*'))
            self.assertEqual(len(backups), 1)
            self.assertTrue(backups[0].is_symlink())
            self.assertEqual(backups[0].readlink(), missing_target)
            self.assertFalse(environment.is_symlink())


if __name__ == '__main__':
    unittest.main()
