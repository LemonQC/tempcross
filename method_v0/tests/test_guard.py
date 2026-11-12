import importlib
import sys
import unittest

from method_v0.common import NoPhysics,guard


class GuardTest(unittest.TestCase):
    def test_engine_imports_are_rejected_without_loading_an_engine(self):
        previous=list(sys.meta_path)
        try:
            guard()
            for name in sorted(NoPhysics.forbidden):
                with self.assertRaises(RuntimeError):importlib.import_module(name)
                self.assertNotIn(name,sys.modules)
        finally:
            sys.meta_path[:]=previous


if __name__=='__main__':unittest.main()
