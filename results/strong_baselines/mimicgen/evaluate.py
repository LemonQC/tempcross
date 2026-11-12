"""One selected downstream visual policy, two tasks, 80 exposed Panda episodes."""
from common import ROOT, OUT, PROJECT, SOURCE, TASKS, digest, read, write


def main():
    import baselines.common.evaluator as evaluator
    from baselines.common.validate import validate_job
    import h5py
    import numpy as np
    selection=read(OUT/'selection.json')
    assert not selection['used_physical_success']
    assert digest(selection['path'])==selection['sha256']
    assert read(ROOT/'downstream/training/status.json')['status']=='COMPLETE'
    evaluator.ROOT=ROOT/'evaluation';evaluator.ROOT.mkdir(exist_ok=True)
    original=evaluator.write_json
    def role_write(path,value):
        if isinstance(value,dict) and value.get('label')=='mimicgen_bc_transformer':
            value.update(role='EXPOSED_REFERENCE',mimicgen_freeze_commit=read(ROOT/'downstream_freeze.json')['remote_commit'],
                policy_information=['rgb','verbatim instruction'],extra_generated_training_data=True,
                checkpoint_selected_by_validation=True,physical_success_used_for_selection=False)
        original(path,value)
    evaluator.write_json=role_write
    frozen=read(PROJECT/'configs/evaluation/baseline_conditions_60_99.json')
    jobs=[]
    for task in TASKS:
        job=evaluator.evaluate('mimicgen_bc_transformer','robomimic',selection['path'],task,'panda')
        path=evaluator.ROOT/'formal/mimicgen_bc_transformer'/task/'panda/job.json'
        v=validate_job(path)
        for line in (path.parent/'episodes.jsonl').read_text().splitlines():
            r=__import__('json').loads(line)
            expected=next(e for e in frozen if e['task']==task and e['robot']=='panda' and e['official_index']==r['official_condition_index'])
            with h5py.File(r['telemetry_path']) as f:
                assert np.array_equal(f['baseline/restored_initial_state'][0],np.asarray(expected['initial_state'],dtype=np.float64))
        v['exact_original_initial_states_verified']=True;write(path.parent/'validation.json',v)
        jobs.append(job)
        state=read(OUT/'status.json');state['evaluated_episode_ids']=[r['condition_id'] for j in jobs for r in j['conditions']]
        write(OUT/'status.json',state)
    write(ROOT/'evaluation/complete.json',dict(status='COMPLETE',episodes=80,jobs=jobs,role='EXPOSED_REFERENCE'))


if __name__=='__main__':main()
