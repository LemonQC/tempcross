"""Evaluate only the new selected checkpoint on exposed reference conditions.

The existing evaluator, original frozen states, controller, adapter and success
function are reused. Only the output root and honest evaluation-role metadata
are set here; the old 1280 experiments and their ledgers remain immutable.
"""
import argparse
import shutil
from common import OUT, ROOT, digest, environment, read, setup_source, write


def main():
    import os
    os.environ.update(environment())
    setup_source()
    import baselines.common.evaluator as evaluator
    from baselines.common.validate import validate_job
    from baselines.common.probe import validate_freeze
    marker=validate_freeze()
    selected=read(OUT/'selection.json');model=read(OUT/'model_index.json')
    assert not selected['used_physical_success'] and selected['step']==model['selected_step']
    assert digest(model['merged_checkpoint'])==model['merged_sha256']
    evaluator.ROOT=ROOT/'physical'
    evaluator.ROOT.mkdir(exist_ok=True)
    original_write=evaluator.write_json
    def role_write(path,value):
        if isinstance(value,dict) and value.get('label')=='new_barx_joint_target_adaptation':
            value.update(role='EXPOSED_FROZEN_REFERENCE_NOT_INDEPENDENT_TEST',
                         adaptation_frozen_commit=read(ROOT/'frozen_publication.json')['remote_commit'],
                         checkpoint_selected_by_validation=True,physical_success_used_for_selection=False)
        original_write(path,value)
    evaluator.write_json=role_write
    for task in ['pnp_counter_to_sink','pnp_sink_to_counter']:
        for robot in ['panda','ur5e']:
            if shutil.disk_usage(ROOT).free<25*1024**3:
                raise OSError('Frozen disk floor reached before physical job')
            job=evaluator.evaluate('new_barx_joint_target_adaptation','barx',model['merged_checkpoint'],task,robot)
            path=evaluator.ROOT/'formal'/job['label']/task/robot/'job.json'
            validation=validate_job(path)
            # The reused validator reserves its frozen-state check for its old
            # role label. Check exact initial-state values explicitly here.
            import h5py,numpy as np
            frozen=read(evaluator.PROJECT/'configs/evaluation/baseline_conditions_60_99.json')
            records=[__import__('json').loads(s) for s in (path.parent/'episodes.jsonl').read_text().splitlines()]
            for row in records:
                expected=next(r for r in frozen if r['task']==task and r['robot']==robot and r['official_index']==row['official_condition_index'])
                with h5py.File(row['telemetry_path'],'r') as f:
                    assert np.array_equal(f['baseline/restored_initial_state'][0],np.asarray(expected['initial_state'],dtype=np.float64))
            validation['exact_original_frozen_initial_states_verified']=True
            write(path.parent/'validation.json',validation)
    write(ROOT/'physical/complete.json',dict(status='COMPLETE',episodes=160,old_models_rerun=0,
          independent_unseen_test=False,role='EXPOSED_REFERENCE',selected_step=selected['step']))


if __name__=='__main__':main()
