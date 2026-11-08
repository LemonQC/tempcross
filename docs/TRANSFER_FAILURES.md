# COFT-P2 完整失败审计

**P1 80次完成：15次成功、65次失败。65次失败均有完整状态/动作指标与视频抽帧核查。P0 Panda 的重复失败也已核查，但不加入65次独立计数。**

主阶段：62次 GRASP_ALIGNMENT（包含未包围/对齐与闭合时机失败）、2次 OBJECT_DROP、1次 PLACEMENT_ERROR。它们是可观察阶段，不是对策略、视觉或控制根因的唯一因果判断。没有仅凭 success=false 判为不可达、碰撞或相机遮挡。

| 方法/机器人 | GRASP_ALIGNMENT | OBJECT_DROP | PLACEMENT_ERROR |
|---|---:|---:|---:|
| no_reps/ur5e | 15 | 0 | 0 |
| no_reps/panda | 19 | 0 | 0 |
| joint_reps/ur5e | 9 | 2 | 0 |
| joint_reps/panda | 19 | 0 | 1 |

## 证据、例外与未知项

- **抓取获取失败**：完整轨迹中的 EEF—物体距离、指间接触、夹爪命令与物体运动配合视频核查。源 No Reps/Joint 失败中7/15、3/11从未闭爪；目标为16/19、17/20。升高或双指碰到物体本身不代表稳定抓持。
- **joint_reps/ur5e/006：OBJECT_DROP**。胡萝卜双指接触步280–404，闭爪命令保持时失去接触，之后在415步达到约4cm的最高升高量，再落回台面，空手继续向水槽移动。属于短暂抬升/滑脱，不能说已完成稳定长距离搬运。
- **joint_reps/ur5e/011：OBJECT_DROP**。香蕉接触步236–377，升高约7.3cm，尚未到槽时提前开爪，约400步落回台面。区分于上一例闭爪滑脱。
- **joint_reps/panda/001：PLACEMENT_ERROR**。黄瓜抓持后上升11.1cm并向水槽搬运、释放；退离距离最多8.43cm、结束8.326cm，未达到官方>10cm。单独的入槽布尔值不可用，不伪造“入槽已通过”；退离不足本身已足以使任务为假。简单退离控制是合理待检验解释。
- **no_reps/panda/004：有效成功参考**。真实 seed1009，389步完成官方任务；晚期视频与状态核查已保存。这反驳“Panda整体仿真坏了”或“所有目标条件均不可执行”的说法。
- **同条件精确重放**：P0与P1的Joint Reps首条两臂动作/qpos/EEF/物体逐项差为0。该重复只用于工程复现，不增加样本量。

全部80条的策略动作—官方夹爪变换—12D执行动作映射误差为0；未确认 POLICY_ACTION_MISMATCH。每个目标方法只有2个失败接近臂限位，且目标成功例也接近过限位；不将限位出现当成失败原因。接触日志包含正常夹持、手指内部/本体接触，未确认 COLLISION 致因不代表全程无碰撞。无独立IK或控制干预，无法确认 REACHABILITY_OR_EXECUTION 主因。

视频核查使用带帧号样本和完整状态/动作时序；不是逐帧人工标注。`video_reviews.json` 给出每例路径、检查帧、证据及分类。65个P1失败及P0失败均有审查记录；其余成功仅核查代表例，不冒称所有成功视频均逐例人工审查。完整视频的每一帧都经过了解码完整性检查。

官方没有独立连续任务进度；合理预抓6D姿态、力闭合、碰撞致因和动态可见面积没有经过验证的独立真值。因此记录阶段证据与 UNKNOWN 根因，不将诊断阈值升级成成功标签。旧COFT投影只检查中心视锥，不能判断遮挡或 Eye-in-Hand 观测失效。

## 全部65条失败

真实 seed 从冻结条件读取；官方 nominal seed 不用于匹配。所有失败均达到600步。每条的完整证据与外部视频路径可按 Case ID 在 [analysis.json](../results/simulation/closed_loop/analysis.json)、[video_reviews.json](../results/simulation/closed_loop/video_reviews.json) 和 [episodes.csv](../results/simulation/closed_loop/episodes.csv) 中查询。

| Case ID | 真实seed | 步数 | 主失败阶段 | 接触步数 | 最大升高m | 闭爪命令占比 |
|---|---:|---:|---|---:|---:|---:|
| no_reps/ur5e/000 | 1000 | 600 | GRASP_ALIGNMENT | 0 | 0.0010 | 0.227 |
| no_reps/ur5e/001 | 1002 | 600 | GRASP_ALIGNMENT | 0 | 0.0007 | 0.173 |
| no_reps/ur5e/003 | 1005 | 600 | GRASP_ALIGNMENT | 0 | 0.0005 | 0.213 |
| no_reps/ur5e/004 | 1009 | 600 | GRASP_ALIGNMENT | 0 | 0.0009 | 0.000 |
| no_reps/ur5e/005 | 1010 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/ur5e/008 | 1013 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.213 |
| no_reps/ur5e/009 | 1014 | 600 | GRASP_ALIGNMENT | 0 | 0.0117 | 0.427 |
| no_reps/ur5e/010 | 1015 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/ur5e/012 | 1017 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/ur5e/014 | 1024 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/ur5e/015 | 1025 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.013 |
| no_reps/ur5e/016 | 1026 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.027 |
| no_reps/ur5e/017 | 1028 | 600 | GRASP_ALIGNMENT | 0 | 0.0052 | 0.000 |
| no_reps/ur5e/018 | 1029 | 600 | GRASP_ALIGNMENT | 18 | 0.0033 | 0.347 |
| no_reps/ur5e/019 | 1030 | 600 | GRASP_ALIGNMENT | 0 | 0.0001 | 0.000 |
| no_reps/panda/000 | 1000 | 600 | GRASP_ALIGNMENT | 0 | 0.0009 | 0.000 |
| no_reps/panda/001 | 1002 | 600 | GRASP_ALIGNMENT | 0 | 0.0013 | 0.053 |
| no_reps/panda/002 | 1004 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/panda/003 | 1005 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/panda/005 | 1010 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/panda/006 | 1011 | 600 | GRASP_ALIGNMENT | 0 | 0.0046 | 0.000 |
| no_reps/panda/007 | 1012 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/panda/008 | 1013 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/panda/009 | 1014 | 600 | GRASP_ALIGNMENT | 0 | 0.0005 | 0.000 |
| no_reps/panda/010 | 1015 | 600 | GRASP_ALIGNMENT | 0 | 0.0148 | 0.253 |
| no_reps/panda/011 | 1016 | 600 | GRASP_ALIGNMENT | 0 | 0.0149 | 0.000 |
| no_reps/panda/012 | 1017 | 600 | GRASP_ALIGNMENT | 0 | 0.0202 | 0.013 |
| no_reps/panda/013 | 1019 | 600 | GRASP_ALIGNMENT | 0 | 0.0522 | 0.000 |
| no_reps/panda/014 | 1024 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| no_reps/panda/015 | 1025 | 600 | GRASP_ALIGNMENT | 0 | 0.0031 | 0.000 |
| no_reps/panda/016 | 1026 | 600 | GRASP_ALIGNMENT | 0 | 0.0625 | 0.000 |
| no_reps/panda/017 | 1028 | 600 | GRASP_ALIGNMENT | 0 | 0.0021 | 0.000 |
| no_reps/panda/018 | 1029 | 600 | GRASP_ALIGNMENT | 0 | 0.0008 | 0.000 |
| no_reps/panda/019 | 1030 | 600 | GRASP_ALIGNMENT | 0 | 0.0255 | 0.000 |
| joint_reps/ur5e/004 | 1009 | 600 | GRASP_ALIGNMENT | 0 | 0.0015 | 0.003 |
| joint_reps/ur5e/006 | 1011 | 600 | OBJECT_DROP | 123 | 0.0399 | 0.288 |
| joint_reps/ur5e/010 | 1015 | 600 | GRASP_ALIGNMENT | 0 | 0.0003 | 0.000 |
| joint_reps/ur5e/011 | 1016 | 600 | OBJECT_DROP | 142 | 0.0727 | 0.507 |
| joint_reps/ur5e/012 | 1017 | 600 | GRASP_ALIGNMENT | 0 | 0.0267 | 0.333 |
| joint_reps/ur5e/013 | 1019 | 600 | GRASP_ALIGNMENT | 0 | 0.0219 | 0.000 |
| joint_reps/ur5e/014 | 1024 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| joint_reps/ur5e/015 | 1025 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.613 |
| joint_reps/ur5e/016 | 1026 | 600 | GRASP_ALIGNMENT | 0 | 0.0257 | 0.507 |
| joint_reps/ur5e/017 | 1028 | 600 | GRASP_ALIGNMENT | 0 | 0.0029 | 0.240 |
| joint_reps/ur5e/018 | 1029 | 600 | GRASP_ALIGNMENT | 0 | 0.0014 | 0.387 |
| joint_reps/panda/000 | 1000 | 600 | GRASP_ALIGNMENT | 0 | 0.0006 | 0.000 |
| joint_reps/panda/001 | 1002 | 600 | PLACEMENT_ERROR | 218 | 0.1105 | 0.373 |
| joint_reps/panda/002 | 1004 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| joint_reps/panda/003 | 1005 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| joint_reps/panda/004 | 1009 | 600 | GRASP_ALIGNMENT | 0 | 0.0014 | 0.000 |
| joint_reps/panda/005 | 1010 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| joint_reps/panda/006 | 1011 | 600 | GRASP_ALIGNMENT | 0 | 0.0039 | 0.000 |
| joint_reps/panda/007 | 1012 | 600 | GRASP_ALIGNMENT | 0 | 0.0010 | 0.000 |
| joint_reps/panda/008 | 1013 | 600 | GRASP_ALIGNMENT | 0 | 0.0002 | 0.000 |
| joint_reps/panda/009 | 1014 | 600 | GRASP_ALIGNMENT | 0 | 0.0359 | 0.040 |
| joint_reps/panda/010 | 1015 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| joint_reps/panda/011 | 1016 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| joint_reps/panda/012 | 1017 | 600 | GRASP_ALIGNMENT | 0 | 0.0373 | 0.000 |
| joint_reps/panda/013 | 1019 | 600 | GRASP_ALIGNMENT | 0 | 0.0091 | 0.000 |
| joint_reps/panda/014 | 1024 | 600 | GRASP_ALIGNMENT | 0 | 0.0000 | 0.000 |
| joint_reps/panda/015 | 1025 | 600 | GRASP_ALIGNMENT | 0 | 0.0059 | 0.000 |
| joint_reps/panda/016 | 1026 | 600 | GRASP_ALIGNMENT | 0 | 0.0062 | 0.000 |
| joint_reps/panda/017 | 1028 | 600 | GRASP_ALIGNMENT | 0 | 0.0020 | 0.000 |
| joint_reps/panda/018 | 1029 | 600 | GRASP_ALIGNMENT | 0 | 0.0532 | 0.227 |
| joint_reps/panda/019 | 1030 | 600 | GRASP_ALIGNMENT | 0 | 0.0385 | 0.000 |
