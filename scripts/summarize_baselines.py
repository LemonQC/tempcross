#!/usr/bin/env python3
"""Generate tables and figures from actual closed episodes and audited budgets."""
import collections,csv,json,math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.common.paths import PROJECT,ROOT,TASKS,write_json
OUT=PROJECT/'results/baselines'
NAMES={'A_barx_no_reps':'BARX No Reps source','A_barx_joint':'BARX Joint source','A_bc_transformer':'BC-Transformer source-only','A_diffusion_policy':'Diffusion Policy source-only','B_bc_gmm':'BC-GMM target-only','B_bc_rnn':'BC-RNN target-only','B_bc_transformer':'BC-Transformer target-only','B_diffusion_policy':'Diffusion Policy target-only','C_bc_transformer':'BC-Transformer source+target','C_diffusion_policy':'Diffusion Policy source+target','C_barx_no_reps_6000':'BARX No Reps legacy LoRA6k','C_barx_joint_96000':'BARX Joint legacy LoRA96k'}

def write(name,lines):
 (PROJECT/'docs'/name).write_text('\n'.join(lines)+'\n')

def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','/') for x in row)+' |' for row in rows]
def cell(c):
 if c is None:return '未完成（不填0%）'
 lo,hi=c['wilson95'];return f"{c['successes']}/{c['episodes']} ({c['success_rate']:.1%}; 95%CI {lo:.1%}–{hi:.1%})"
def paired_p(a,b):
 n=a+b
 return min(1.,2*sum(math.comb(n,i) for i in range(min(a,b)+1))/2**n) if n else 1.

def plots(cells,budgets):
 if not cells:return
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'tempcross-baseline','font.size':9})
 labels=[x for x in NAMES if any(c['label']==x and c['robot']=='panda' for c in cells)]
 fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
 for ax,task in zip(axes,TASKS):
  data={c['label']:c for c in cells if c['robot']=='panda' and c['task']==task};x=list(range(len(labels)))
  means=[data[k]['success_rate'] if k in data else float('nan') for k in labels];lo=[means[i]-data[k]['wilson95'][0] if k in data else 0 for i,k in enumerate(labels)];hi=[data[k]['wilson95'][1]-means[i] if k in data else 0 for i,k in enumerate(labels)]
  ax.bar(x,means,color=[{'A':'#5478a3','B':'#519470','C':'#b58147'}[k[0]] for k in labels],yerr=[lo,hi],capsize=2)
  ax.set_xticks(x,labels,rotation=75,ha='right');ax.set_ylim(0,1);ax.set_title(task+' / Panda / original60–99');ax.set_ylabel('Official success fraction; Wilson 95% CI');ax.grid(axis='y',alpha=.2)
 fig.suptitle('Data regimes and legacy budgets differ; descriptive comparison only',fontsize=11);fig.tight_layout();fig.savefig(OUT/'official_success.svg',metadata={'Date':None});plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));new=[r for r in budgets if r['method']!='barx']
 for r in new:
  p=OUT/'training'/r['label']/'status.json'
  s=json.loads(p.read_text());history=s.get('validation_history',[])
  if history:ax.plot([h['anchor_positions'] for h in history],[h['value']/s['initial_validation']['value'] for h in history],label=r['label'])
 ax.set_xlabel('Anchor positions (target for B/C; source for A)');ax.set_ylabel('Fixed-validation metric / its initial value');ax.set_title('Within-model validation progress; different losses are not comparable');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'validation_progress.svg',metadata={'Date':None});plt.close(fig)
 for path in [OUT/'official_success.svg',OUT/'validation_progress.svg']:
  path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

def main():
 s=json.loads((OUT/'status.json').read_text());analysis=json.loads((OUT/'analysis.json').read_text());cells=analysis['cells'];episodes=analysis['episodes'];budgets=s['training'];by={(c['label'],c['task'],c['robot']):c for c in cells};complete=s['status']=='COMPLETE'
 lines=['# Baseline真实执行结果', '',f"状态：**{s['status']}**。已关闭 {s['formal_complete_cells']} 个正式评测格、{s['formal_episodes']} 个episode。训练日志设置 {s['models_with_training_logs']}/8，选模完成 {s['models_with_selected_checkpoints']}/8。",'', '只统计新冻结官方60–99条件；每任务每机器人每设置40次。旧P0–P3和开发smoke不加入成功率。技术错误另列，不填0%。原始小型结果见[逐episode结果](../results/baselines/episodes.csv)、[主表](../results/baselines/cells.csv)、[训练预算](../results/baselines/training_budgets.csv)、[checkpoint索引](../results/baselines/checkpoint_index.csv)。']
 if not complete:lines+=['','本文件是执行中的真实快照，不能作为完整Benchmark结论。']
 for group,title in [('A','零目标示范迁移'),('B','仅目标学习'),('C','源+目标学习')]:
  labels=[k for k in NAMES if k.startswith(group)]
  lines+=['',f'## Group {group}：{title}','']+table(['设置','Counter→Sink','Sink→Counter'],[[NAMES[k],cell(by.get((k,TASKS[0],'panda'))),cell(by.get((k,TASKS[1],'panda')))] for k in labels])
 lines+=['','Group C历史BARX使用原LoRA权重，其目标划分、初始化、分辨率、样本量与新BC/DP不同，不构成等预算排名。','', '## UR5e源具身参考','']+table(['Source-only设置','Counter→Sink','Sink→Counter'],[[NAMES[k],cell(by.get((k,TASKS[0],'ur5e'))),cell(by.get((k,TASKS[1],'ur5e')))] for k in NAMES if k.startswith('A')])
 lines+=['','## 真实训练预算','', '新模型共享100条Panda示范池（每任务50），实际90条训练、10条整示范验证；源池1800条为IIWA/Kinova3/UR5e每任务每机器人300条，实际1620/180。B/C架构配对的目标采样序列、更新次数和目标暴露相同，C另加同量source，因此总样本/计算不相等。anchor不等于展开后的图像/标签总数；精确覆盖见各training目录的demonstration_coverage.csv与consumption_summary.json。','']
 rows=[]
 for r in budgets:
  if r['method']=='barx':continue
  coverage=OUT/'training'/r['label']/'consumption_summary.json';cov=json.loads(coverage.read_text()) if coverage.exists() else {}
  rows.append([r['label'],r['target_positions'],r['source_positions'],r['optimizer_updates'],cov.get('unique_demonstrations',{}),r.get('actual_batch'),f"{r.get('parameter_count',0)/1e6:.2f}",f"{r.get('gpu_occupied_wall_seconds',0):.1f}",f"{r.get('peak_gpu_allocated_bytes',0)/1024**3:.2f}",r.get('budget_stop_reason') or r['status']])
 lines+=table(['设置','target anchor','source anchor','实际更新','实际独立demo','实际batch','参数M','单卡训练循环秒','峰值GiB','终止/状态'],rows)
 matched=OUT/'matched_data_audit.json'
 if matched.exists():
  audit=json.loads(matched.read_text())
  lines+=['','B/C实际匹配审计见[matched_data_audit.json](../results/baselines/matched_data_audit.json)：已通过的配对初始化序列化文件SHA完全相同，实际消费target ID/anchor的有序哈希及更新次数相同。随机crop配置相同，但不强制每个target样本使用同一个随机crop；C混合batch的随机draw及batch统计也不同。']
  if audit.get('blocked_pairs'):lines+=['','未完成匹配的配对：`'+json.dumps(audit['blocked_pairs'],ensure_ascii=False)+'`，不据此作配对数据效应结论。']
 diagnostics=[]
 for r in budgets:
  if r['method']=='barx':continue
  state=json.loads((OUT/'training'/r['label']/'status.json').read_text());history=state['validation_history'];best=min(history,key=lambda h:h['value'])
  diagnostics.append([r['label'],history[-1]['metric'],f"{state['initial_validation']['value']:.6f}",f"{best['value']:.6f} @ {best['anchor_positions']}",f"{history[-1]['value']:.6f}"])
 lines+=['','固定验证曲线的只读诊断（不同loss不直接比较；未据此更换权重）：','']+table(['设置','指标','初始','历史最小值及anchor','选定末步值'],diagnostics)
 lines+=['','验证平台表示最近窗口未继续改善，不保证全局充分收敛；若末步较历史验证最小值恶化，也原样保留末步规则和历史checkpoint。训练CSV的learning_rate字段记录scheduler更新后的下一步LR；DP首个实际optimizer步骤LR为0，未回写原始CSV。','', '目标HDF每条ep_meta中的robot0_agentview_left与冻结评测barx_panda_agentview名称不同，但全部100条的基座相对位置、旋转、45°FOV及180×320分辨率均核对一致，见camera_calibration_audit.json；未用默认相机名称臆断视角不一致。']
 lines+=['','GPU秒为单卡训练循环墙钟时间：包含循环内数据等待和周期验证，不含模型构建/加载、初始验证、checkpoint保存和哈希，也不是CUDA kernel累计时间。验证每12k固定256 anchors，最终checkpoint来自预先冻结的验证延长/末步规则，未用最终测试挑选。预算到顶仍改善须标为BUDGET_LIMITED_NOT_CONVERGED；验证平台也不证明全局充分收敛。','', 'BARX source prior复用官方50k更新权重，不把其原始大规模预训练当作本轮训练。No Reps6k旧LoRA为3000更新/batch2、3068 target+2932 source；逐图像SHA回溯全部3068 target实际覆盖18条示范（95条eligible池）；2932 source样本完整回溯为19条三具身源示范。Joint96k旧LoRA为12000更新/batch8、48412 target+47588 source，实际100 target及115 source demos，source三具身均有。旧LoRA不是官方充分全参数适配。','', '## 同条件配对与稳定性','']
 pairs=s['paired_comparisons'];rows=[]
 for p in pairs:
  a,b=len(p['rescued']),len(p['regressed']);rows.append([p['candidate']+' vs '+p['baseline'],p['task'],a,b,p['net_success_change'],f'{paired_p(a,b):.4f}',p['interpretation']])
 lines+=table(['配对','任务','救回','退化','净增','精确双侧McNemar p（未校正）','解释'],rows) if rows else ['配对正式结果尚未完成。']
 lines+=['','新BC/DP每设置只有一个训练seed，40条件CI衡量本次测试集不确定性，不能据此宣称跨训练seed的稳定排序或跨场景泛化。配对检验为描述性、未做多重校正；加入source的结论只适用于该共享动作/观测、目标暴露匹配且C计算更多的设定。','', '## 本次观测到的最佳值','']
 for group in ['A','B','C']:
  for task in TASKS:
   candidates=[c for c in cells if c['group']==group and c['task']==task and c['robot']=='panda' and not (group=='C' and c['legacy_setting'])]
   if candidates:
    k=max(c['successes'] for c in candidates);best=[c for c in candidates if c['successes']==k];lines.append(f"- {group} / {task}："+'、'.join(NAMES[c['label']] for c in best)+f'，{k}/40；仅本次观测最大值。')
 lines+=['','BARX历史LoRA在C表单独保留，不能和新C模型忽略预算差异宣称总体最强。','', '## 推理与资产','', '独占GPU、同一旧开发图像的16次warmup+80次策略调用见[latency](../results/baselines/latency/)。平均每控制步和fresh-query时间分开，chunk8缓存不会被当作每步完整推理；正式并发评测耗时不作独占延迟比较。','', f'服务器执行根目录：`{ROOT}`。完整视频、原生输出、12维实际控制、TCP、夹爪、物体轨迹及接触均在formal/，仓库[视频索引](../results/baselines/video_index.csv)和[所有外部文件SHA256](../results/baselines/external_files.csv)保留绝对路径。','', '![Official success](../results/baselines/official_success.svg)','', '![Validation](../results/baselines/validation_progress.svg)']
 latency=[]
 for path in sorted((OUT/'latency').glob('*.json')):
  d=json.loads(path.read_text());latency.append(dict(label=d['label'],mean_step_ms=1000*d['mean_policy_step_seconds'],p95_step_ms=1000*d['p95_policy_step_seconds'],mean_fresh_query_ms=1000*d['mean_fresh_query_seconds'],peak_gpu_GiB=d['peak_gpu_allocated_bytes']/1024**3,parameters=d['parameter_count']))
 if latency:
  with (OUT/'latency_summary.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(latency[0]));w.writeheader();w.writerows(latency)
  lines+=['','独占GPU实测推理（ms）：','']+table(['设置','均值/控制步','p95/控制步','均值/新query','峰值GiB'],[[d['label'],f"{d['mean_step_ms']:.2f}",f"{d['p95_step_ms']:.2f}",f"{d['mean_fresh_query_ms']:.2f}",f"{d['peak_gpu_GiB']:.2f}"] for d in latency])
  lines+=['','显存列采用PyTorch max_memory_allocated口径，不包含MuJoCo图形渲染和驱动分配；推理计时期间仅有一个基准策略进程，无并发训练或仿真。']
 write('BASELINE_RESULTS.md',lines)
 stages=['approached_center_8cm_proxy','issued_close','physical_jaw_closure_5mm','bilateral_contact','stable_contact_lift','transport_10cm_contact_proxy','placement_official_subpredicate','official_task_success']
 failure=['# Baseline阶段诊断与技术异常','', '官方任务success是唯一主成功判据。到达8cm、接触后抬升3cm持续10个状态、接触搬运10cm只是预先声明的诊断代理。阶段代理不构成严格递进链，物体中心距离也不等于抓取可达性；不以代理失败否定官方成功。放置及撤离子判据直接调用官方定义，不送入policy。','']
 rows=[]
 for c in cells:
  es=[e for e in episodes if (e['label'],e['task'],e['robot'])==(c['label'],c['task'],c['robot'])];rows.append([c['label'],c['task'],c['robot']]+[f"{sum(bool(e[k]) for e in es)}/{len(es)}" for k in stages])
 failure+=table(['设置','任务','机器人','到达代理','闭爪命令','实测闭爪5mm','双指接触','稳定抬升','搬运代理','官方放置','官方success'],rows)
 counts=collections.Counter(e['observed_failure_stage'] for e in episodes if e['robot']=='panda' and not e['success'])
 failure+=['','Panda失败阶段计数：`'+json.dumps(dict(counts),ensure_ascii=False)+'`。这是观察到的断点分类，不能从单次轨迹直接认定视觉、时间建模、动作表征或迁移是根因。','', '每episode的最大抬升、接触持续、阶段标志及视频路径见[失败索引](../results/baselines/failure_index.csv)，原始物理轨迹支持复查。闭爪命令与实测指垫中心距离减少≥5mm分别记录，后者只说明确有闭爪运动，不声称完全闭合或抓住物体。观察到的策略失败只有在完整合法rollout结束后才进入分母。','', '技术分类记录checkpoint loading、frozen environment/condition restore、network inference、action interface、simulation step、telemetry/success audit等实际出错阶段。详见technical_status.csv及原job traceback；技术异常与未运行方法均不填0%。训练不足在训练预算表单列，不能从失败episode倒推。','', '首次DP探针的三个失败属于记录器对zero-LR warmup首步的参数变化断言错误，日志原样保留；第二步核验确有参数变化，探针权重丢弃，正式算法和warmup规则不变。目标HDF初次打包的慢迭代技术尝试也保留，训练只使用完整hash验证后的包。','', '本轮不重新执行P0–P3。B1、标准OSC/IK、物体相对位姿迁移和A3仍是历史PRIVILEGED_GEOMETRY_REFERENCE/DIAGNOSTIC_ONLY，未混入RGB主表。','', 'KITE有公开官方代码，阻碍是与固定旧栈的依赖及BARX接触意图/decoder数据接口；CrossZero尚未核实可执行官方仓库与本任务资产，且原点云输入超出RGB共同边界。BARX Single Rep/ECoT未核实匹配预训练权重，单卡从头源VLA训练未授权预算；它们是BLOCKED而非0%。RoboCasa新版DP不是本固定任务上的第二个独立算法。']
 write('BASELINE_FAILURE_ANALYSIS.md',failure)
 plots(cells,budgets)
 ready=complete and not s['formal_technical_cells'] and s['models_with_selected_checkpoints']==8 and s['formal_complete_cells']==32
 write_json(OUT/'platform_readiness.json',dict(engineering_benchmark_ready=ready,formal_cells=s['formal_complete_cells'],formal_episodes=s['formal_episodes'],single_training_seed=True,independent_training_seed_repeats=False,unlimited_convergence_not_demonstrated=True,evaluation_loop_reusable_for_new_policies=ready,new_policy_factory_registration_required=True,current_policy_kinds=['barx','robomimic'],publication_claim='Runnable benchmark platform with reusable evaluation loop; new policy implementations require factory/CLI registration. T-RO scientific novelty, multiple-training-seed robustness and generalization beyond two tasks are not established by engineering completion'))
 print(json.dumps(dict(reports_generated=True,formal_episodes=s['formal_episodes'],platform_ready=ready)))
if __name__=='__main__':main()
