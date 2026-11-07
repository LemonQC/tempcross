# 参数来源、坐标与证据等级

本次使用 URDF XML 作为公开数据读取；没有安装、启动或调用 ROS。

| 参数 | 来源 | 已验证内容 | 未验证内容 |
|---|---|---|---|
| SR5 六关节 origin/axis/position limits | [ROKAE 官方源](https://github.com/RokaeRobot/rokae_ros2/blob/d6a508c38513d4c39b2f10f14275d0e26df7fb43/rokae_description/urdf/xMateSR5.urdf.xacro) | 源文件入库；逐关节测试；v0 转录与新模型随机 FK 一致 | 现场 SR5-5/0.9C 序列号、版本、零位、控制器限制与此文件的对应 |
| Panda 七关节与固定 joint8 | [Franka 官方源](https://github.com/frankarobotics/franka_ros/blob/ddd2fffd9de44b02ad15b4bbb2bfa2cec4d60d98/franka_description/robots/common/franka_arm.xacro) | 源文件与 joint_limits.yaml 入库；独立矩阵指数 FK；link7→link8 平移 0.107 m | 具体实机标定及负载参数 |
| SR5 产品型号 | [官方数据表](https://static.rokae.com/Downloads/DataSheet/ROKAE_xMate%20SR_EN.pdf)、[硬件手册](https://static.rokae.com/Downloads/Manual/SR5-5%200.9C%20Hardware%20Installation%20Manual.pdf) | 公开产品为六轴，标称工作半径 919 mm；关节范围与公开链相容 | 这些粗规格不等价于实机 FK 校验 |
| 法兰→TCP | 假设：平移 [0,0,0.12] m，无旋转 | 两机器人相同，排除工具差异的控制对照；变换逆运算已测试 | 不是 PGIA 或 Franka Hand 的实际 TCP |
| 法兰→光学相机 A | 假设：平移 [.065,0,.025] m，无旋转 | 完整 SE(3) 连乘；光学 +z 前、+x 右、+y 下的 pinhole 约定 | 手眼实测、安装朝向、RGB/depth 对齐与实际支架 |
| 法兰→光学相机 B | 假设：平移 [-.065,.025,.025] m，无旋转 | 同一本体换外参的受控反事实 | 不是已知 Panda 相机安装值 |
| 相机内参 | 假设 640×480，fx=fy=420，cx=320，cy=240，depth [.25,2] m | 软件投影与裁剪边界 | 不是 D435i 实测标定或有效深度保证 |
| 两指手 | 共同代理：掌部盒中心 [0,0,.045]、半尺寸 [.045,.045,.025]；手指中心 [0,±.034,.095]、半尺寸 [.04,.008,.025]，单位 m | 用于工具碰撞与视线遮挡；明确两指 | 不是 PGIA-140-80-W 或 Franka Hand CAD；未核对开口、指尖与抓取接触 |
| 机器人碰撞与遮挡 | 关节中心连线、半径 .025 m 的代理；间隔 <=.025 m 的球，半径膨胀 .0125 m | 覆盖代理中心线 capsule；检查非近邻自碰撞与环境 | 不是官方机器人外形保守包络；相邻／次相邻链段排除，可能漏检或误判 |
| 场景、物体 | 固定生成分布，全部假设，未筛除不利样本 | 随机种子、每个 AABB、姿态、完整候选轨迹入库 | 场景实测、摩擦、质量、接触稳定性、形状不对称造成的抓持差异 |

世界坐标 = 机器人基座坐标；两机器人基座置于同一原点、同一朝向，非实机安装事实。`T_A_B` 将 B 坐标中的点映射到 A。`T_world_tcp = FK(q) T_flange_tcp`，`T_world_optical = FK(q) T_flange_optical`，目标法兰位姿为 `T_world_tcp inverse(T_flange_tcp)`。SR5 输出帧 `xMateSR5_link6`；Panda 输出帧 `panda_link8`。Panda 默认 hand 安装角与真实 hand TCP 不隐含在这里；两个本体均使用明确共同的代理工具。

这些外参是合法刚体变换，但“数学合法”不等于“真实标定正确”。完整要求中的两种真实夹爪、实测外参与官方碰撞网格验证尚未达到；因此本轮不能升级为已验证真实跨具身操作，也不能凭代理几何否定该假设。

源文件保留原版权/许可；模型提取脚本仅支持这两个已固定文件的语法，不是通用 xacro 解释器。使用 `scripts/build_models.py` 重建后必须通过测试，禁止凭商品工作半径重造关节链。
