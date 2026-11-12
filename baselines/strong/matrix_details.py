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
        '| 方法 | 本轮源具身 | 部署观测（冻结协议） | Panda训练示范/任务 | 协议 |',
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
        lines.append(f"| {method} | {render(source)} | {render(obs)} | {render(number)} | [冻结协议](../results/strong_baselines/{method}/protocol.json) |")
    lines+=['',
        '旧BC/DP源池包含IIWA、Kinova3和UR5e。新增ACT及VQ-BeT依本轮UR5e/Panda设置仅取UR5e的600条源示范，保留原有身份划分；旧结果不重训、不覆盖，也不把跨源池比较称为等数据预算比较。BARX预训练/适配及MimicGen生成监督的来源仍以各自原协议为准。',
        '', 'ACT唯一一次decoder输出修复及原版24000更新尝试另列归档。VQ-BeT唯一一次TF32精度修复发生在正式训练前，原四次组件探针更新和四条smoke保留，修复版独立目录重新验收。不能用原技术尝试替代修复版训练完成量；技术尝试、探针和正式评测成本分别记账。']
    path.write_text(text+'\n'.join(lines)+'\n')

if __name__=='__main__':main()
