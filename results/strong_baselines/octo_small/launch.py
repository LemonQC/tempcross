"""Set the isolated CUDA11 library search path before TensorFlow/JAX imports."""
import os
import sys
from baselines.strong.common import ROOT

def main():
    directory=ROOT/'octo_small/env'
    site=directory/'lib/python3.10/site-packages'
    cuda=site/'nvidia/cuda_nvcc'
    environment=dict(os.environ)
    libraries=[str(p) for p in sorted((site/'nvidia').glob('*/lib'))]
    assert libraries and (cuda/'bin/ptxas').exists()
    environment['LD_LIBRARY_PATH']=':'.join(libraries+[environment.get('LD_LIBRARY_PATH','')])
    environment['PATH']=str(cuda/'bin')+':'+environment['PATH']
    environment['XLA_FLAGS']=environment.get('XLA_FLAGS','')+' --xla_gpu_cuda_data_dir='+str(cuda)
    environment.update(XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',
        TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='2',TOKENIZERS_PARALLELISM='false')
    arguments=sys.argv[1:]
    if arguments[0]=='--cpu':environment['JAX_PLATFORMS']='cpu';arguments=arguments[1:]
    module='results.strong_baselines.octo_small.'+arguments[0]
    executable=str(directory/'bin/python')
    os.execve(executable,[executable,'-u','-m',module,*arguments[1:]],environment)

if __name__=='__main__':main()
