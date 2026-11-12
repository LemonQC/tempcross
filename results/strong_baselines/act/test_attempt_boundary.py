"""Prevent accidentally evaluating old weights through the repaired interface."""
import os
import subprocess
import unittest
from baselines.strong.common import PROJECT,PYTHON


class AttemptBoundaryTest(unittest.TestCase):
    def test_original_root_is_rejected_before_any_cuda_model_allocation(self):
        env=os.environ.copy();env['STRONG_BASELINE_RUN_ROOT']='/home/users/lemon/strong-baselines-20261112'
        code=("import sys; from results.strong_baselines.act.model import make; "
              "\ntry: make()\n"
              "except AssertionError as e:\n"
              " assert 'corrected ACT root' in str(e)\n"
              " assert 'torch' not in sys.modules\n"
              "else: raise AssertionError('Original and repaired attempts were mixed')\n")
        result=subprocess.run([str(PYTHON),'-c',code],env=env,cwd=PROJECT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)


if __name__=='__main__':unittest.main()
