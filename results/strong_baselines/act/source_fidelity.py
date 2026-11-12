"""Read-only source parity for the actual pinned ACT core and its one repair."""
from pathlib import Path
from baselines.strong.common import ROOT,out,read,write,digest

def main():
    source=ROOT/'vendor/act';port=ROOT/'vendor/act_port';rows=[]
    expected=Path('detr/models/detr_vae.py')
    paths=[Path('policy.py')]+[p.relative_to(source) for p in (source/'detr').rglob('*.py')]
    for rel in sorted(paths):
        original=(source/rel).read_text();actual=(port/rel).read_text()
        if rel==expected:
            transformed=original.replace('self.input_proj_robot_state = nn.Linear(14, hidden_dim)','self.input_proj_robot_state = nn.Linear(768, hidden_dim)')
            transformed=transformed.replace('self.encoder_action_proj = nn.Linear(14, hidden_dim)','self.encoder_action_proj = nn.Linear(7, hidden_dim)')
            transformed=transformed.replace('self.encoder_joint_proj = nn.Linear(14, hidden_dim)','self.encoder_joint_proj = nn.Linear(768, hidden_dim)')
            transformed=transformed.replace('state_dim = 14 # TODO hardcode','state_dim = 7 # canonical OSC action dimension',1)
            transformed=transformed.replace('additional_pos_embed.weight)[0]','additional_pos_embed.weight)[-1]')
            assert transformed==actual
        else:assert original==actual,rel
        rows.append(dict(path=str(rel),unchanged=original==actual,original_sha256=digest(source/rel),executed_port_sha256=digest(port/rel)))
    assert sum(not r['unchanged'] for r in rows)==1
    write(out('act')/'source_fidelity.json',dict(status='PASS',official_commit=read(out('act')/'official_source.json')['commit'],
        files=rows,only_changed_core_file=str(expected),
        exact_declared_dimensional_and_decoder_output_edits=True,official_policy_loss_unchanged=True,
        no_policy_inference=True,no_training=True,no_physics=True))
    print('PASS: all ACT core source files match the pinned original except the exact declared dimensional edits and last-decoder-output repair.')

if __name__=='__main__':main()
