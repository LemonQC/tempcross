"""Import author implementations, never a local DP-network substitute."""
import sys
from pathlib import Path
SOURCE=Path('/home/users/lemon/baseline-benchmark/vendor/hri-eu-flow-matching')
REVISION='516e8e18875b27741bdbdb8a252e904032c90723'
sys.path.insert(0,str(SOURCE/'external/models'))
from unet import ConditionalUnet1D
from resnet import get_resnet,replace_bn_with_gn
from pusht import normalize_data,unnormalize_data,sample_sequence
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
