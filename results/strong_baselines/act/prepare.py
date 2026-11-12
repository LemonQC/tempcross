"""Pin/copy the official implementation and apply only dimensional I/O edits."""
import shutil
import subprocess
from pathlib import Path
from baselines.strong.common import ROOT,PROJECT,OLD,out,write,read,digest,initialize,resource

COMMIT='742c753c0d4a5d87076c8f69e5628c79a8cc5488'

def main():
    if (out('act')/'technical_repair.json').exists():
        assert str(ROOT)==read(out('act')/'technical_repair.json')['corrected_server_root'],'Use the corrected STRONG_BASELINE_RUN_ROOT; preserve the original attempt.'
    initialize();original=ROOT/'vendor/act';port=ROOT/'vendor/act_port'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip()==COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=original).strip()
    if not port.exists():
        port.mkdir();shutil.copytree(original/'detr',port/'detr')
        for name in ['policy.py','LICENSE']:shutil.copyfile(original/name,port/name)
        path=port/'detr/models/detr_vae.py';text=path.read_text()
        assert text.count('self.input_proj_robot_state = nn.Linear(14, hidden_dim)')==2
        text=text.replace('self.input_proj_robot_state = nn.Linear(14, hidden_dim)','self.input_proj_robot_state = nn.Linear(768, hidden_dim)')
        assert text.count('self.encoder_action_proj = nn.Linear(14, hidden_dim)')==1
        text=text.replace('self.encoder_action_proj = nn.Linear(14, hidden_dim)','self.encoder_action_proj = nn.Linear(7, hidden_dim)')
        text=text.replace('self.encoder_joint_proj = nn.Linear(14, hidden_dim)','self.encoder_joint_proj = nn.Linear(768, hidden_dim)')
        text=text.replace('state_dim = 14 # TODO hardcode','state_dim = 7 # canonical OSC action dimension',1)
        assert text.count('additional_pos_embed.weight)[0]')==1
        text=text.replace('additional_pos_embed.weight)[0]','additional_pos_embed.weight)[-1]')
        path.write_text(text)
    patch=subprocess.run(['diff','-u',str(original/'detr/models/detr_vae.py'),str(port/'detr/models/detr_vae.py')],capture_output=True,text=True)
    (out('act')/'official_io.patch').write_text(patch.stdout)
    write(out('act')/'official_source.json',dict(url='https://github.com/tonyzhaozh/act',commit=COMMIT,license='MIT (ACT), upstream DETR notices retained',
        original=str(original),port=str(port),original_clean=True,
        differences=['technical repair 1: use the final of seven decoder outputs; upstream selected the first intermediate output',
                     'robot-state conditioning slot takes frozen CLIP instruction embedding, not measured qpos',
                     '14D joint actions replaced by 7D canonical OSC; action encoder input and action output adapted',
                     'single 128x128 camera from frozen existing pack',
                     'training anchors and whole-demonstration splits inherited; update budget instead of epoch budget',
                     'explicit valid forecast mask instead of treating an exactly-zero action as absent'],
        core_preserved=['official ResNet18+DETR CVAE','all seven configured decoder layers now receive gradients','latent dimension 32','posterior action encoder','L1+10*KL loss','zero latent inference','exponential temporal aggregation k=0.01'],
        patched_source_sha256=digest(port/'detr/models/detr_vae.py'),official_policy_sha256=digest(port/'policy.py')))
    from baselines.strong.data import prepare
    prepare('act')
    write(out('act')/'preflight_resources.json',resource())

if __name__=='__main__':main()
