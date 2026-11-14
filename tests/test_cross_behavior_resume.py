"""Meaningful crash/recovery and no-replay invariants, no physical steps."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from cross_behavior.common import recover_step, write, atomic, PROJECT


class ResumeTests(unittest.TestCase):
    def test_committed_step_recovers_after_pointer_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root/'cursor.json', {'step':7})
            write(root/'inflight.json', {'next_step':8})
            atomic(root/'steps/000008.pkl', b'durable-step')
            self.assertEqual(recover_step(root), 8)
            self.assertFalse((root/'inflight.json').exists())
            self.assertEqual(recover_step(root), 8)

    def test_uncertain_physics_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root/'cursor.json', {'step':7})
            write(root/'inflight.json', {'next_step':8})
            before = (root/'cursor.json').read_bytes()
            with self.assertRaisesRegex(RuntimeError, 'AMBIGUOUS_INFLIGHT_PHYSICS'):
                recover_step(root)
            self.assertEqual((root/'cursor.json').read_bytes(), before)
            self.assertTrue((root/'inflight.json').exists())

    def test_status_needs_no_site_packages_and_is_read_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root/'status.json', {'status':'PAUSED', 'current':{'condition':4}})
            before = {p.name:p.read_bytes() for p in root.iterdir()}
            env = dict(os.environ, CROSS_BEHAVIOR_ROOT=folder)
            proc = subprocess.run([sys.executable, '-S', str(PROJECT/'scripts/run_cross_behavior_expansion.py'),
                                   '--stage','auto','--resume','--status'], env=env,
                                  capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(proc.stdout)['current']['condition'], 4)
            self.assertEqual({p.name:p.read_bytes() for p in root.iterdir()}, before)


if __name__ == '__main__':
    unittest.main()
