"""Evaluate TOCAT's identifiability and same-evidence decision computation."""
from collections import Counter,defaultdict
import csv
import hashlib
from pathlib import Path
import pickle

from method_v0.common import PROJECT,OUT,ROOT,TASKS,guard,read,write,digest,ndarray_digest
from method_v0.organize import load_rows
from method_v0.contracts import (Edge,EvidenceGraph,continuation_contract,witnessed_task_planner,
    physical_feasibility_interval,symbolic_nonidentifiability_witness)


def equal(a,b):
    import numpy as np
    if hasattr(a,'detach') and hasattr(b,'detach'):return np.array_equal(a.detach().cpu().numpy(),b.detach().cpu().numpy())
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) and isinstance(b,type(a)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def handoffs():
    import numpy as np
    episodes=read(PROJECT/'results/mechanism/analysis.json')['episodes']
    groups=defaultdict(list)
    for r in episodes:groups[r['condition']].append(r)
    result=[]
    for condition,rr in sorted(groups.items()):
        assert len(rr)==3
        has=[r['handoff_action_index'] is not None for r in rr]
        if not any(has):continue
        assert all(has) and len({r['handoff_action_index'] for r in rr})==1
        directories=[Path(r['history_path']).parent for r in rr]
        loaded=[np.load(d/'handoff.npz',allow_pickle=False) for d in directories]
        keys=loaded[0].files
        assert all(set(z.files)==set(keys) for z in loaded)
        max_error=0.
        for key in keys:
            assert all(np.array_equal(loaded[0][key],z[key]) for z in loaded[1:])
            assert all(np.isfinite(z[key]).all() for z in loaded)
            if loaded[0][key].size:
                max_error=max(max_error,max(float(np.abs(loaded[0][key].astype(float)-z[key].astype(float)).max()) for z in loaded[1:]))
        prefix=[np.load(d/'prefix.npz',allow_pickle=False) for d in directories]
        assert all(np.array_equal(prefix[0]['actions'],z['actions']) and np.array_equal(prefix[0]['states'],z['states']) for z in prefix[1:])
        # Only compare declared safe keys; never serialize private policy configuration.
        runtimes=[]
        for d in directories:
            with (d/'handoff_runtime.pkl').open('rb') as f:runtimes.append(pickle.load(f))
        rng_keys=['python_rng','numpy_rng','torch_rng','cuda_rng','policy_queue','model_training']
        rng_equal={k:all(equal(runtimes[0][k],v[k]) for v in runtimes[1:]) for k in rng_keys}
        assert all(rng_equal.values())
        initials=[read(d/'initial.json') for d in directories]
        assert len({v['model_xml_sha256'] for v in initials})==1
        anchor=rr[0]['handoff_action_index']
        values=[dict(method=r['method'],success=r['task_success'],total_steps=r['steps'],
            suffix_steps=r['steps']-anchor,remaining_benchmark_steps=600-anchor,
            failure_stage=r['failure_stage'],early_stop_censored_for_benchmark=not r['task_success'] and r['steps']<600,
            history_path=r['history_path'],telemetry_path=r['telemetry_path']) for r in rr]
        result.append(dict(condition=condition,task=TASKS[0],robot='panda',anchor_state_index=anchor,
            numeric_fields_compared=len(keys),max_numeric_error=max_error,prefix_actions_and_states_exact=True,
            recorded_rng_checks=rng_equal,model_xml_sha256=initials[0]['model_xml_sha256'],
            snapshot_sha256={r['method']:digest(d/'handoff.npz') for r,d in zip(rr,directories)},
            suffixes=values,opposite_factual_outcomes=len({r['success'] for r in values})>1,
            role='EXISTING_OUTCOME_SELECTED_DIAGNOSTIC_ONLY_NOT_A_TRAIN_OR_TEST_SET'))
        for z in loaded+prefix:z.close()
    assert len(result)==5
    write(OUT/'matched_continuation_witnesses.json',result)
    return result


def graph(rows,matched):
    edges={};goals=set();queries=[]
    for r in rows:
        anchor=r['first_bilateral_contact_state']
        index=anchor if anchor is not None else 0
        root=r['uid']+'/anchor';end=r['uid']+'/observed-end'
        duration=r['steps']-index
        if duration==0:
            if r['official_success']:goals.add(root)
            edges[root]=[]
        else:
            edges[root]=[Edge(r['behavior'],end,duration,r['uid'])]
            if r['official_success']:goals.add(end)
        queries.append(dict(root=root,budget=r['benchmark_horizon']-index,corpus=r['corpus'],
                            anchor_type='BILATERAL_CONTACT' if anchor is not None else 'INITIAL_NO_CONTACT'))
    for r in matched:
        root=f"verified-mechanism/{r['condition']}";edges[root]=[]
        for s in r['suffixes']:
            end=root+'/'+s['method']+'/end'
            edges[root].append(Edge(s['method'],end,s['suffix_steps'],root+'/'+s['method']))
            if s['success']:goals.add(end)
        queries.append(dict(root=root,budget=600-r['anchor_state_index'],corpus='mechanism_existing_matched',anchor_type='CERTIFIED_RECORDED_HANDOFF'))
    g=EvidenceGraph(edges,goals);g.validate();return g,queries


def comparison(rows,matched):
    g,queries=graph(rows,matched);n=0;mismatch=[];full=[]
    for q in queries:
        for budget in range(q['budget']+1):
            a=continuation_contract(g,q['root'],budget);b=witnessed_task_planner(g,q['root'],budget)
            n+=1
            if a!=b:mismatch.append(dict(root=q['root'],budget=budget,contract=sorted(a),planner=sorted(b)))
        choices=continuation_contract(g,q['root'],q['budget'])
        full.append(dict(**q,supported_options=sorted(choices),physical_feasibility_interval=physical_feasibility_interval(choices)))
    write(OUT/'same_information_decisions.json',full)
    write(OUT/'planner_equivalence.json',dict(queries=len(queries),all_budget_comparisons=n,disagreements=mismatch,
        supported_edge_source='only actually recorded complete action suffixes; merge only verified mechanism handoffs',
        comparator='independent forward exhaustive witnessed-path planner, not a continuous TAMP implementation',
        exact_same_information=True,new_physics_steps=0,trained_models=0))
    assert not mismatch
    return full,n


def cross_embodiment(rows):
    import numpy as np
    selected=[r for r in rows if r['corpus']=='baseline_1280' and r['behavior']=='A_barx_joint']
    pairs=[]
    for task in TASKS:
        for i in range(60,100):
            source=next(r for r in selected if r['task']==task and r['robot']=='ur5e' and r['condition']==i)
            target=next(r for r in selected if r['task']==task and r['robot']=='panda' and r['condition']==i)
            q1=np.asarray(source['object_initial_quaternion']);q2=np.asarray(target['object_initial_quaternion'])
            angle=2*np.arccos(np.clip(abs(np.dot(q1,q2))/(np.linalg.norm(q1)*np.linalg.norm(q2)),0,1))
            pairs.append(dict(task=task,condition=i,object_asset_same=source['object_asset']==target['object_asset'],
                initial_object_position_distance_m=float(np.linalg.norm(np.asarray(source['object_initial_position'])-target['object_initial_position'])),
                initial_object_rotation_distance_rad=float(angle),source_success=source['official_success'],target_success=target['official_success'],
                source_condition_id=source['condition_id'],target_condition_id=target['condition_id'],
                interpretation='DESCRIPTIVE_POLICY_OUTCOME_NOT_INTRINSIC_CAPABILITY_OR_CAUSAL_PAIR',
                source_history=source['uid'],target_history=target['uid']))
    write(OUT/'cross_embodiment_descriptive_pairs.json',pairs)
    return pairs


def main():
    guard();rows=load_rows('factual_continuations');matched=handoffs()
    decisions,n=comparison(rows,matched);pairs=cross_embodiment(rows)
    by_corpus=defaultdict(list)
    for r in rows:by_corpus[r['corpus']].append(r)
    coverage=[]
    for name,rr in sorted(by_corpus.items()):
        contact=[r for r in rr if r['first_bilateral_contact_state'] is not None]
        coverage.append(dict(corpus=name,episodes=len(rr),official_successes=sum(r['official_success'] for r in rr),
            bilateral_contact_anchors=len(contact),contact_then_success=sum(r['official_success'] for r in contact),
            contact_then_not_complete=sum(not r['official_success'] for r in contact),
            early_stop_before_benchmark=sum(r['truncated_before_benchmark_horizon'] for r in rr),
            certified_intrinsic_infeasibility_labels=0))
    opposite=[r for r in matched if r['opposite_factual_outcomes']]
    invalid=sum(not s['success'] for r in opposite for s in r['suffixes'])
    abstract=symbolic_nonidentifiability_witness()
    observations=abstract['observations']
    assert all(abstract['world_a'][k]==v==abstract['world_b'][k] for k,v in observations.items())
    positive=sum(bool(r['supported_options']) for r in decisions)
    result=dict(status='COMPLETE_EVIDENCE_NEGATIVE_METHOD_GATE',candidate='TOCAT_v0',
        primary_factual_episodes=len(rows),demonstrations=1900,coverage=coverage,
        matched_handoff_groups=len(matched),matched_groups_with_opposite_outcomes=len(opposite),
        failed_suffixes_at_physically_witnessed_feasible_handoffs=invalid,
        source_target_policy_outcome_disagreements=sum(r['source_success']!=r['target_success'] for r in pairs),
        cross_embodiment_intrinsic_capability_differences_identified=False,
        graph_queries=len(decisions),supported_positive_queries=positive,unknown_feasibility_queries=len(decisions)-positive,
        all_budget_planner_comparisons=n,meaningfully_different_same_information_decisions=0,
        certified_physical_infeasibility_labels=0,
        symbolic_nonidentifiability=dict(two_models_fit_identical_observations=True,unobserved_option_feasibility_differs=True,
            type='LOGICAL_PROOF_NOT_ROBOT_COUNTERFACTUAL_SUPERVISION'),
        learned_policy_or_new_mechanism_implemented=False,training_updates=0,new_physics_steps=0,old_baseline_reruns=0,
        decision='NO_GO_LEARNED_METHOD_IDENTIFIABILITY_AND_NOVELTY_NOT_ESTABLISHED',
        stage_b_scope='factual label extraction, candidate specification checking and planner equivalence; no neural method implemented after gate failure',
        independent_unseen_test=False)
    write(OUT/'summary.json',result)
    with (OUT/'coverage.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(coverage[0]),lineterminator='\n');w.writeheader();w.writerows(coverage)
    print(__import__('json').dumps(result,indent=2))


if __name__=='__main__':main()
