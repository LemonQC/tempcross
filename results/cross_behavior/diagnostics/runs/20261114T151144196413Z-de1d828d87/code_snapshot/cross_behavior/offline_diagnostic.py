"""Programmatic diagnostics from indexed saved evidence, without a simulator.

Privileged telemetry is consumed here only, never supplied to any policy.
JSONL is the supported trajectory schema. Unsupported/missing evidence fails
explicitly; unavailable ground truth remains null rather than being invented.
"""
import argparse
import csv
import io
import json
import math
import signal
from collections import Counter
from pathlib import Path
import numpy as np
from cross_behavior.handoff import HandoffError, atomic, read, write, sha, now, safe_text

PAUSE=False


def request_pause(*_):
    global PAUSE
    PAUSE=True


def vector(value):
    if value is None:return None
    v=np.asarray(value,dtype=float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise HandoffError('INVALID_3D_VECTOR')
    return v


def norm(value):
    return float(np.linalg.norm(value))


def optional_list(value):
    return None if value is None else np.asarray(value).tolist()


def projection_from_rgbd(xyz):
    """Fit calibrated projective mapping to recorded pixel/world ray pairs.

    This does not infer or synthesize object labels. Fit error is retained;
    unsuccessful calibration prevents overlays rather than fabricating pixels.
    """
    h,w=xyz.shape[:2]
    ys=np.linspace(0,h-1,min(h,24),dtype=int);xs=np.linspace(0,w-1,min(w,32),dtype=int)
    xx,yy=np.meshgrid(xs,ys)
    points=xyz[np.ix_(ys,xs)].reshape(-1,3);pixels=np.column_stack((xx.ravel(),yy.ravel()))
    mask=np.isfinite(points).all(1)&(np.linalg.norm(points,axis=1)<30)
    points,pixels=points[mask],pixels[mask]
    if len(points)<20:return None,None
    center=points.mean(0);scale=max(float(points.std()),1e-8)
    X=(points-center)/scale;uv=(pixels-np.array([w/2,h/2]))/max(w,h)
    design=np.zeros((2*len(X),11));rhs=uv.ravel()
    extended=np.column_stack((X,np.ones(len(X))))
    design[::2,:4]=extended;design[1::2,4:8]=extended
    design[::2,8:]=-uv[:,0,None]*X;design[1::2,8:]=-uv[:,1,None]*X
    coefficients,_,rank,_=np.linalg.lstsq(design,rhs,rcond=None)
    if rank<11:return None,None
    P=np.r_[coefficients,1.].reshape(3,4)
    TX=np.eye(4);TX[:3,:3]/=scale;TX[:3,3]=-center/scale
    Tuv=np.array([[max(w,h),0,w/2],[0,max(w,h),h/2],[0,0,1.]])
    P=Tuv@P@TX
    projected=np.column_stack((points,np.ones(len(points))))@P.T
    fitted=projected[:,:2]/projected[:,2,None]
    rms=float(np.sqrt(np.mean(np.sum((fitted-pixels)**2,axis=1))))
    return (P if rms<.5 else None),rms


def project(P,point):
    if P is None or point is None:return None
    p=P@np.r_[point,1.]
    if abs(p[2])<1e-10:return None
    return (p[:2]/p[2]).tolist()


def timeline(rows,identity,hz):
    phases=[]
    for row in rows:
        phase=row['decision'].get('phase','UNIDENTIFIED')
        if not phases or phases[-1]['phase']!=phase:
            phases.append(dict(episode_id=identity,phase=phase,start_step=row['step'],end_step=row['step']))
        else:phases[-1]['end_step']=row['step']
    for phase in phases:
        phase.update(control_steps=phase['end_step']-phase['start_step']+1,
                     start_seconds=(phase['start_step']-1)/hz,end_seconds=phase['end_step']/hz)
    return phases


def episode_diagnostic(episode,contract,hz):
    file=lambda name:Path(episode['files'][name]['path'])
    data=read(file('episode.json'));condition=read(file('condition.json'))
    rows=[json.loads(line) for line in file('telemetry.jsonl').read_text().splitlines() if line.strip()]
    if not rows or [r['step'] for r in rows]!=list(range(1,len(rows)+1)):
        raise HandoffError('NONCONTIGUOUS_OR_EMPTY_TELEMETRY: '+episode['id'])
    if data.get('steps')!=len(rows):raise HandoffError('EPISODE_LENGTH_MISMATCH: '+episode['id'])
    for row in rows:
        if not all(k in row for k in ('decision','metrics','action')):raise HandoffError('UNSUPPORTED_JSONL_SCHEMA')
        if len(row['action'])!=7 or not np.isfinite(row['action']).all():raise HandoffError('INVALID_SAVED_ACTION')
    first,last=rows[0],rows[-1];metrics=[r['metrics'] for r in rows]
    predictions=[(r['step'],r['decision']['perceived']) for r in rows if isinstance(r['decision'].get('perceived'),dict)]
    perceived=predictions[0][1] if predictions else {}
    target_errors=[float(r['decision']['translation_error_m']) for r in rows if r['decision'].get('translation_error_m') is not None]
    if not all(math.isfinite(v) for v in target_errors):raise HandoffError('NONFINITE_TRACKING_ERROR')
    tcp=vector(last['metrics']['tcp'])
    target=vector(last['decision'].get('target_tcp'))
    native=[r['step'] for r in rows if r['metrics'].get('native_success') is True]
    phases=timeline(rows,episode['id'],hz)
    independent=[];crosscheck=[];post_errors=[]
    for i,row in enumerate(rows):
        target_i=vector(row['decision'].get('target_tcp'))
        if target_i is None:continue
        post_errors.append(norm(target_i-vector(row['metrics']['tcp'])))
        if i:
            measured=norm(target_i-vector(rows[i-1]['metrics']['tcp']));independent.append(measured)
            if row['decision'].get('translation_error_m') is not None:
                crosscheck.append(abs(measured-float(row['decision']['translation_error_m'])))
    result=dict(episode_id=episode['id'],task=episode['task'],robot=episode['robot'],condition=episode['condition'],
        steps=len(rows),duration_seconds=len(rows)/hz,recorded_full_success=bool(data['success']),
        native_predicate_available=episode['task']!='push',first_native_success_step=native[0] if native else None,
        final_native_success=bool(last['metrics'].get('native_success')),final_tcp_position=tcp.tolist(),
        final_executed_phase=last['decision'].get('phase'),
        mean_reported_pre_action_tracking_error_m=float(np.mean(target_errors)) if target_errors else None,
        max_reported_pre_action_tracking_error_m=max(target_errors) if target_errors else None,
        mean_independently_recomputed_pre_action_tracking_error_m=float(np.mean(independent)) if independent else None,
        max_tracking_error_crosscheck_difference_m=max(crosscheck) if crosscheck else None,
        mean_post_action_target_error_m=float(np.mean(post_errors)) if post_errors else None,
        final_target_to_post_action_tcp_error_m=norm(target-tcp) if target is not None else None,
        target_estimate_available_steps=len(predictions),total_steps=len(rows),
        timed_out=len(rows)>=contract.get('max_steps',len(rows)+1),
        maximum_stable_control_steps=max(m.get('stable_control_steps',0) for m in metrics),
        privileged_values_used_only_offline=True)
    perception=dict(episode_id=episode['id'],first_estimate_step=predictions[0][0] if predictions else None,
        first_recorded_estimate=perceived,estimate_semantics='Latched initial estimate; availability is not fresh detection or visibility at every subsequent step',
        state_alignment='RGB estimate precedes action; logged truth is post-action of the same control step. Initial comparison is approximate, not strict causal pairing.')
    if episode['task']=='push':
        initial=vector(first['metrics']['object_position']);final=vector(last['metrics']['object_position'])
        goal=vector(first['metrics']['goal_center']);estimate=vector(perceived.get('start'));goal_estimate=vector(perceived.get('target'))
        shift=final-initial;intended=goal[:2]-initial[:2]
        length=norm(intended);axis=intended/max(length,1e-12);lateral=np.array([-axis[1],axis[0]])
        angle=None if norm(shift[:2])<1e-6 else math.degrees(math.atan2(float(shift[:2]@lateral),float(shift[:2]@axis)))
        result.update(perceived_object_position=optional_list(estimate),perceived_goal_center=optional_list(goal_estimate),
            true_object_reference_position=initial.tolist(),true_goal_center=goal.tolist(),
            true_object_reference_step=first['step'],
            object_localization_error_m=norm(estimate-initial) if estimate is not None else None,
            goal_localization_error_m=norm(goal_estimate-goal) if goal_estimate is not None else None,
            goal_localization_xy_error_m=norm(goal_estimate[:2]-goal[:2]) if goal_estimate is not None else None,
            final_object_position=final.tolist(),observed_displacement_vector=shift.tolist(),
            final_recorded_displacement_m=float(last['metrics']['displacement_m']),
            observed_reference_displacement_xy_m=norm(shift[:2]),push_direction_error_degrees=angle,
            lateral_displacement_m=float(shift[:2]@lateral),final_goal_xy_error_m=norm(final[:2]-goal[:2]))
        touch=[r['step'] for r in rows if r['metrics'].get('gripper_object_contact')]
        result['contact_control_steps']=len(touch)
        result['failure_class']='NO_TARGET_ESTIMATE' if not predictions else (
            'OFF_GOAL_AFTER_CONTACT' if touch else 'APPROACH_NOT_COMPLETED_VISUAL_OR_CONTROL_AMBIGUOUS')
        perception.update(object_estimate=optional_list(estimate),goal_estimate=optional_list(goal_estimate),
                          object_truth=initial.tolist(),goal_truth=goal.tolist())
    elif episode['task']=='stack':
        final_object=vector(last['metrics']['object_position'])
        release=[r['step'] for r in rows if isinstance(r['decision'].get('phase'),int) and r['decision']['phase']>=6 and r['metrics'].get('released')]
        distances=[norm(vector(r['metrics']['tcp'])-vector(r['metrics']['object_position'])) for r in rows]
        result.update(first_release_step=release[0] if release else None,
            remaining_seconds_after_release=(len(rows)-release[0])/hz if release else None,
            final_retreat_distance_m=norm(tcp-final_object),
            retreat_threshold_m=contract.get('stack',{}).get('retreat_distance_m'),
            final_retreat_pass=last['metrics'].get('retreat'),final_released=last['metrics'].get('released'),
            final_xy_error_m=last['metrics'].get('xy_error_m'),final_height_error_m=last['metrics'].get('height_error_m'))
        result['failure_class']=('NATIVE_STACK_BUT_RETREAT_INCOMPLETE' if native and not last['metrics'].get('retreat') else
                                 'STABILITY_INCOMPLETE' if native else 'NATIVE_STACK_NOT_REACHED')
        result['timeout_reason']=result['failure_class'] if result['timed_out'] else 'NOT_A_TIMEOUT'
        perception.update(object_estimate=perceived.get('start'),target_estimate=perceived.get('target'),
                          object_truth=first['metrics'].get('object_position'),target_truth=first['metrics'].get('object_B_position'))
    elif episode['task']=='drawer':
        touches=[r['step'] for r in rows if r['metrics'].get('handle_contact')]
        longest=current=0
        for r in rows:
            current=current+1 if r['metrics'].get('handle_contact') else 0;longest=max(longest,current)
        opening=max(max(m['drawer_normalized_joint'].values()) for m in metrics)
        normal=vector(perceived.get('normal'));handle=vector(perceived.get('handle'))
        result.update(perceived_handle_position=optional_list(handle),perceived_handle_normal=optional_list(normal),
            normal_tilt_from_horizontal_degrees=math.degrees(math.asin(min(1.,abs(normal[2])/max(norm(normal),1e-12)))) if normal is not None else None,
            first_handle_contact_step=touches[0] if touches else None,contact_control_steps=len(touches),
            total_handle_contact_seconds=len(touches)/hz,longest_continuous_contact_seconds=longest/hz,
            maximum_normalized_opening=float(opening),handle_visibility='NO_ESTIMATE' if not predictions else 'ESTIMATE_AVAILABLE_VISIBILITY_NOT_CERTIFIED',
            failure_class='VISUAL_NO_ESTIMATE' if not predictions else ('CONTACT_WITH_INSUFFICIENT_OPENING' if touches else 'APPROACH_VISUAL_OR_CONTROL_AMBIGUOUS'))
        perception.update(handle_estimate=optional_list(handle),normal_estimate=optional_list(normal),true_handle_pose=None,
            true_handle_pose_reason='Exact dynamic handle pose was not exported in the JSONL schema; no simulator or counterfactual labels are generated.')
    else:raise HandoffError('UNSUPPORTED_TASK_DIAGNOSTIC: '+episode['task'])
    return result,phases,perception


def overlay(episode,perception,output,max_width):
    import cv2
    names=episode['files']
    if 'initial.png' not in names or 'initial_rgbd.npz' not in names:return dict(status='UNAVAILABLE_INDEXED_RGBD')
    image=cv2.imread(names['initial.png']['path'])
    if image is None:raise HandoffError('UNREADABLE_INDEXED_IMAGE')
    with np.load(names['initial_rgbd.npz']['path'],allow_pickle=False) as data:
        xyz=np.asarray(data['xyz'])
    P,rms=projection_from_rgbd(xyz)
    audit=dict(projection_reprojection_rms_pixels=rms,projection_matrix=None if P is None else P.tolist(),
               approximation='Initial image only. World-point projection is fitted from recorded RGB-D rays; not simulator rendering.')
    h,w=image.shape[:2];canvas=cv2.copyMakeBorder(image,0,100,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
    points=[('object_estimate',(0,0,255)),('goal_estimate',(255,255,0)),('object_truth',(0,255,0)),
            ('goal_truth',(255,0,255)),('handle_estimate',(0,255,255)),('target_estimate',(255,255,0)),('target_truth',(255,0,255))]
    audit['points']={};line=0
    for key,color in points:
        point=vector(perception.get(key))
        if point is None:continue
        pixel=project(P,point)
        inside=pixel is not None and 0<=pixel[0]<w and 0<=pixel[1]<h
        distances=np.linalg.norm(xyz.reshape(-1,3)-point,axis=1)
        finite=distances[np.isfinite(distances)]
        minimum=float(finite.min()) if len(finite) else None
        audit['points'][key]=dict(pixel=pixel,inside_initial_fov=inside,nearest_recorded_depth_surface_distance_m=minimum,
                                  visibility_certified=False)
        if inside:
            xy=tuple(int(round(v)) for v in pixel);cv2.drawMarker(canvas,xy,color,cv2.MARKER_CROSS,9,1)
        cv2.putText(canvas,key+('' if inside else ' [off-frame/unresolved]'),(4,h+12+line*13),cv2.FONT_HERSHEY_SIMPLEX,.30,color,1,cv2.LINE_AA);line+=1
    if canvas.shape[1]<max_width:
        scale=min(2.,max_width/canvas.shape[1]);canvas=cv2.resize(canvas,None,fx=scale,fy=scale)
    output.parent.mkdir(parents=True,exist_ok=True)
    ok,data=cv2.imencode('.jpg',canvas,[cv2.IMWRITE_JPEG_QUALITY,88])
    if not ok:raise HandoffError('OVERLAY_ENCODING_FAILED')
    atomic(output,data.tobytes());audit.update(path=str(output.name),status='GENERATED',width=canvas.shape[1],height=canvas.shape[0])
    return audit


def csv_output(path,records):
    keys=list(dict.fromkeys(k for row in records for k in row))
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
    for row in records:
        writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    atomic(path,stream.getvalue().encode())


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--state',required=True)
    parser.add_argument('--config',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-episodes',type=int)
    args=parser.parse_args(argv)
    state_path=Path(args.state);state=read(state_path);output=Path(state['output']);cfg=read(args.config)
    signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
    try:
        manifest=read(args.manifest)
        # Use the frozen source contract, not a later edited live config.
        contract_file=output/'code_snapshot/configs/tasks/cross_behavior_contracts.json'
        contract=read(contract_file) if contract_file.exists() else {'max_steps':600}
        metrics=[];phases=[];audits=[];processed=0
        selected=Counter();overlay_count=0
        for episode in manifest['episodes']:
            identity=episode['id'];result_file=output/'episode_diagnostics'/f'{identity}.json'
            if PAUSE or (args.stop_after_episodes is not None and processed>=args.stop_after_episodes):
                state.update(analysis_status='INTERRUPTED',error='Clean episode-boundary interruption',updated_at=now());write(state_path,state);return 75
            key=(episode['task'],episode['robot'])
            make_overlay=selected[key]<2 and overlay_count<cfg['maximum_overlay_count']
            if identity in state['episodes']:
                if sha(result_file.read_bytes())!=state['episodes'][identity]['sha256']:raise HandoffError('COMPLETED_EPISODE_OUTPUT_CHANGED')
                result=read(result_file)
            else:
                metric,phase,audit=episode_diagnostic(episode,contract,float(cfg['control_hz']))
                if make_overlay:
                    audit['overlay']=overlay(episode,audit,output/'overlays'/f'{identity}.jpg',cfg['maximum_overlay_width'])
                result=dict(metrics=metric,phases=phase,perception=audit)
                write(result_file,result)
                state['episodes'][identity]=dict(path=str(result_file),sha256=sha(result_file.read_bytes()),status='COMPLETE')
                write(state_path,state)
            if make_overlay:
                selected[key]+=1
                if result['perception'].get('overlay',{}).get('status')=='GENERATED':overlay_count+=1
            metrics.append(result['metrics']);phases.extend(result['phases']);audits.append(result['perception']);processed+=1
        csv_output(output/'episode_metrics.csv',metrics);csv_output(output/'phase_timeline.csv',phases)
        atomic(output/'overlays/README.md',
               b'# Initial-image diagnostic overlays\n\nOnly indexed saved RGB/RGB-D is used. Cross markers are recorded estimates or logged truth projected through a fitted camera matrix. Off-frame or occluded points are not certified visible. Missing RGB-D is reported in perception_audit.json; no image or label is fabricated.\n')
        write(output/'perception_audit.json',dict(episodes=audits,privileged_state_only_offline=True,
              visibility_limit='Latched estimates and initial-image geometric proxies do not establish dynamic visual observability or causal failure attribution.'))
        groups=[]
        for task,robot in sorted({(m['task'],m['robot']) for m in metrics}):
            rows=[m for m in metrics if m['task']==task and m['robot']==robot]
            groups.append(dict(task=task,robot=robot,episodes=len(rows),full_successes=sum(m['recorded_full_success'] for m in rows),
                native_predicate_ever=sum(m['first_native_success_step'] is not None for m in rows) if task!='push' else None,
                failure_classes=dict(Counter(m['failure_class'] for m in rows))))
        summary=dict(schema_version=1,analysis_status='COMPLETE',run_id=state['run_id'],input_fingerprint=state['fingerprint'],
            episodes=len(metrics),actual_control_steps=sum(m['steps'] for m in metrics),groups=groups,
            new_physics_steps=0,training_updates=0,baseline_reruns=0,overlay_count=overlay_count,
            undefined_values_are_null=True,phase_timing_source='Recorded executed decision.phase, not post-step expert phase',
            report_scope='Offline diagnostics of the supplied indexed episodes, not formal benchmark success or causal identification')
        write(output/'diagnostic_summary.json',summary)
        lines=['# Cross-behavior Stage A offline diagnostic', '',f"Run `{state['run_id']}`; input fingerprint `{state['fingerprint']}`.",
            '',f"Computed from {len(metrics)} indexed trajectories and {summary['actual_control_steps']} saved control steps. No physics, training or demonstration collection was started.",
            '', '| Task | Robot | Episodes | Recorded full successes | Native predicate ever |', '|---|---|---:|---:|---:|']
        for g in groups:lines.append(f"| {g['task']} | {g['robot']} | {g['episodes']} | {g['full_successes']} | {g['native_predicate_ever'] if g['native_predicate_ever'] is not None else 'N/A'} |")
        lines+=['','## Episode diagnostics','',
            'All numbers below are calculated from saved JSONL. The control-step 1 object state is the first logged post-action state, not a fabricated t=0 value. Perception was computed before the action, so perception/truth comparisons are explicitly approximate in time. A latched estimate is not fresh per-frame detection.',
            '', 'Native predicate events are separated from frozen full-contract outcomes. Tracking errors are geometric diagnostics, not evidence that perception or mechanics caused failure.', '']
        for m in metrics:
            lines += [f"### {m['episode_id']}",'',f"Steps: {m['steps']}; last executed phase: {m['final_executed_phase']}; diagnostic: `{m['failure_class']}`."]
            if m['task']=='push':
                lines.append(f"Object localization error: {m['object_localization_error_m']} m; goal XY localization error: {m['goal_localization_xy_error_m']} m; recorded displacement: {m['final_recorded_displacement_m']} m; lateral displacement: {m['lateral_displacement_m']} m; direction error: {m['push_direction_error_degrees']} degrees.")
            elif m['task']=='stack':
                lines.append(f"First native success: step {m['first_native_success_step']}; first recorded release in release phase: {m['first_release_step']}; final TCP: {m['final_tcp_position']}; retreat distance: {m['final_retreat_distance_m']} m; threshold: {m['retreat_threshold_m']} m; observed time remaining after release: {m['remaining_seconds_after_release']} s; timeout reason: `{m['timeout_reason']}`.")
            else:
                lines.append(f"Handle estimate available for {m['target_estimate_available_steps']}/{m['steps']} steps (latched); first contact: {m['first_handle_contact_step']}; total/longest contact: {m['total_handle_contact_seconds']}/{m['longest_continuous_contact_seconds']} s; maximum normalized opening: {m['maximum_normalized_opening']}; visual flag: `{m['handle_visibility']}`. Exact dynamic handle ground truth is unavailable in this schema.")
            lines.append(f"Mean reported pre-action tracking error: {m['mean_reported_pre_action_tracking_error_m']} m; final target-to-post-action TCP error: {m['final_target_to_post_action_tcp_error_m']} m.")
            lines.append('')
        lines+=['## Evidence and reuse limits','',
            '- `input_manifest.json` records absolute paths, bytes, SHA256, presence, read status and expected-checksum matches. Raw MP4/NPZ/PKL/HDF5/weights are not uploaded.',
            '- `code_context.json`, `code_snapshot/` and `source_changes.patch` preserve the relevant executed-source context and uncommitted changes without staging unrelated files.',
            '- `phase_timeline.csv`, `episode_metrics.csv`, `perception_audit.json` and bounded initial-image overlays expose the calculations for review.',
            '- Missing detections/ground truth are null, not zero. Visual/control ambiguity remains unresolved where saved evidence cannot distinguish it.',
            '- The next scientific decision remains expert observability and trajectory timing. These offline diagnostics neither authorize another physical repair nor establish algorithm novelty.','']
        atomic(output/'CROSS_BEHAVIOR_STAGE_A_OFFLINE_DIAGNOSTIC.md','\n'.join(lines).encode())
        state.update(analysis_status='COMPLETE',analyzed_at=now(),error=None);write(state_path,state)
        return 0
    except (ValueError,KeyError,OSError,HandoffError) as error:
        state.update(analysis_status='FAILED_ANALYSIS',error=safe_text(str(error)));write(state_path,state)
        print(safe_text(str(error)),file=__import__('sys').stderr);return 1
