"""Publish setting/task cells and information budgets without completion inflation."""
import json
from baselines.strong.common import PROJECT,METHODS,out,read
from baselines.strong.matrix import main as basic

def main():
    basic();path=PROJECT/'docs/STRONG_BASELINE_MASTER_MATRIX.md';text=path.read_text()
    # Multiple trained settings are separate models, not one pooled success rate.
    multiple=set()
    for method in METHODS:
        p=out(method)/'results.json'
        if p.exists() and len({c['group'] for c in read(p).get('cells',[])})>1:multiple.add(method)
    revised=[]
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)>=8 and fields[1].strip() in multiple:
            fields[5]=' 分设置见下表 ';line='|'.join(fields)
        revised.append(line)
    text='\n'.join(revised)+'\n'
    lines=['','## 按设置与任务分列的真实评测','',
           '下表只读取已经存在的结果单元；未执行单元不填0%，不同设置不合并伪装成一个模型。', '',
           '| 方法 | 设置 | 任务 | 官方成功/总次数 | 成功率 | 结果作用域 |',
           '|---|---|---|---:|---:|---|']
    any_cells=False
    for method in METHODS:
        p=out(method)/'results.json'
        if not p.exists():p=out(method)/'partial_results.json'
        if not p.exists():continue
        result=read(p)
        for c in result.get('cells',[]):
            n=c['episodes'];k=c['successes'];assert 0<=k<=n and n>0
            assert c.get('role','EXPOSED_REFERENCE')=='EXPOSED_REFERENCE'
            any_cells=True
            lines.append(f"| {method} | {c['group']} | {c['task']} | {k}/{n} | {100*k/n:.1f}% | {result['status']} / EXPOSED_REFERENCE |")
    if not any_cells:lines.append('| — | — | 尚无通过最终验收的新方法单元 | N/A | N/A | 训练与smoke不计正式结果 |')
    lines+=['','## 数据和信息条件','',
        '数据量、预训练、目标示范、原生传感器及额外监督必须随各方法分列；不默认等数据或等信息。', '',
        '| 方法 | 本轮源具身 | 部署观测（未完成者仅列原生需求） | Panda训练示范/任务 | 协议 |',
        '|---|---|---|---|---|']
    for method in METHODS:
        p=out(method)/'protocol.json'
        if not p.exists():
            lines.append(f'| {method} | 尚未核对 | 尚未核对 | N/A | NOT_RUN |');continue
        cfg=read(p)
        source=cfg.get('source_robots','见协议')
        obs=cfg.get('observation',cfg.get('deployment_observation','见协议'))
        number=cfg.get('train_target_demonstrations_per_task',cfg.get('target_demonstrations','见协议'))
        render=lambda value:json.dumps(value,ensure_ascii=False) if isinstance(value,(list,dict)) else str(value)
        label='未冻结边界' if cfg.get('training_budget_frozen') is False or cfg.get('status','').startswith('NOT_FROZEN') else '冻结协议'
        lines.append(f"| {method} | {render(source)} | {render(obs)} | {render(number)} | [{label}](../results/strong_baselines/{method}/protocol.json) |")
    lines+=['',
        '旧BC/DP源池包含IIWA、Kinova3和UR5e。新增ACT及VQ-BeT依本轮UR5e/Panda设置仅取UR5e的600条源示范，保留原有身份划分；旧结果不重训、不覆盖，也不把跨源池比较称为等数据预算比较。BARX预训练/适配及MimicGen生成监督的来源仍以各自原协议为准。',
        '', 'ACT唯一一次decoder输出修复及原版24000更新尝试另列归档。VQ-BeT唯一一次TF32精度修复发生在正式训练前，原四次组件探针更新和四条smoke保留，修复版独立目录重新验收。不能用原技术尝试替代修复版训练完成量；技术尝试、探针和正式评测成本分别记账。']
    statuses={method:read(out(method)/'status.json') for method in METHODS}
    completed=sum(s['status'].startswith('COMPLETE_') for s in statuses.values())
    partial=sum(s['status']=='PARTIAL' for s in statuses.values())
    blocked=[m for m,s in statuses.items() if s['status']=='BLOCKED']
    lines += ['','## 完整性边界','',
        f'六项中完整Baseline为{completed}/6；PARTIAL为{partial}项；BLOCKED为{len(blocked)}项。只要未达到6/6，就不能称六项Baseline已补齐。',
        '', 'MimicGen v2的有界下游组件完成仍保持PARTIAL；其新策略总成功数2/80，与复用旧人类BC的2/80相同，不能用单一任务的计数变化声称总体增益。',
        '', 'VQ-BeT表中的60000为两阶段训练迭代数，对应84000次正式optimizer调用；探针和废弃技术尝试在各自资源账本另列，不混成相同计算预算。']
    if blocked:
        lines += ['', '| 未完成方法 | 已记录阻碍 | 结论边界 |','|---|---|---|']
        labels={'kite':'原生扩展已可用；一次修复后，数据检查器的导入路径错误使目标几何/解码部署未闭环',
            'crosszero':'未核实作者实现/权重；固定OSC下的原生joint-IK部署桥接未建立',
            'object_flow':'一次启动修复后遇到保存相机名与重建模型不一致；尚无认证流标签、学习预测器或动作连接'}
        for method in blocked:
            lines.append(f"| {method} | {labels.get(method,statuses[method].get('reason_code','见报告'))} | [BLOCKED报告](../results/strong_baselines/{method}/RESULTS.md)；N/A成功率，不是算法失败 |")
    path.write_text(text+'\n'.join(lines)+'\n')

if __name__=='__main__':main()
