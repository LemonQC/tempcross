# COFT-P2 失败分类与案例状态

**P0_BLOCKED；有效 rollout=0；操作失败案例=0。** 这里的 0 是“没有可分类轨迹”，不是“没有策略失效”或“两个机器人全部失败”。`results/simulation/status.json` 的 failure_cases 为真实空列表。

| 固定分类 | 分类所需证据 | 本轮可用性 |
|---|---|---|
| REACHABILITY_OR_EXECUTION | EEF/目标误差、关节极限、控制命令与状态响应，排除策略从未发出正确动作等混淆；不能仅凭末端未到达推断不可达 | 无轨迹，N/A |
| GRASP_ALIGNMENT | 物体与手指相对姿态、接近／闭合时刻、接触状态和视频 | 无轨迹，N/A |
| COLLISION | 具体接触几何、接触时刻与动作中断／任务损失；日常抓持接触不能自动算碰撞失败 | 无轨迹，N/A |
| OBJECT_DROP | 已确认抓持并抬升，随后物体运动／接触丢失及视频证据 | 无轨迹，N/A |
| PLACEMENT_ERROR | 已抓取搬运后未满足官方容器／释放判据的状态与视频 | 无轨迹，N/A |
| POLICY_ACTION_MISMATCH | 原始策略动作、归一化／夹爪变换、环境动作与真实状态响应的不匹配证据 | 未完成数值推理，N/A |
| UNKNOWN | 有失败 rollout 但证据不足，或多种解释不能分离 | 没有 rollout，不人为生成 UNKNOWN episode |

接近、预抓取、抓取并抬升、掉落／碰撞、入槽并释放、动作偏差／极限等阶段均未测，不能从安装日志猜测。没有视频可进行视觉核查，也不能把 checkpoint 下载错误归入上述操作类别。

实际技术失败链：官方 CUDA wheel 下载超时 → 轮1官方 VQ Git EOF/index-pack → 轮2固定 commit 获取成功，但 uv VQ checkout/reset 无法解析对象 → 正式环境单元测试缺依赖 → gate 拒绝启动任何物理 episode。原始日志存 `results/simulation/`，完整未删节日志位于 `/home/users/lemon/coft-p2/`。

观察记录器为未来有效运行预备了独立 HDF5/JSONL：qpos/qvel、EEF/夹爪等观测、全 body pose、target pose、接触几何、策略动作块及环境动作；保留官方视频，不修改输入。它尚未通过完整集成验证，不能说这些数据本轮已经采集。

旧 COFT 中心投影诊断只允许使用实际模拟相机和物体状态，作用是辅助定位几何因素；不估计动态遮挡、不给成功标签，也不证明 Eye-in-Hand 耦合。本轮未执行该诊断。
