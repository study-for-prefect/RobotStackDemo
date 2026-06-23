# Tools

这里仅放可执行工具；可复用逻辑应优先放入 `robot_scene_pipeline/`。

较大的命令文件只保留稳定入口，具体实现放在同目录下的功能包中。例如
`stack_demo_pipeline.py` 对应 `stack_demo/`，`moveit_plan_preview.py`
对应 `moveit_preview/`。

| 目录 | 内容 |
| --- | --- |
| `calibration/` | hover-only 标定、XY 系统偏差采集与拟合 |
| `data/` | RealSense 数据集采集 |
| `diagnostics/` | 静态场景和感知稳定性诊断 |
| `monitoring/` | 实时 YOLO/深度监控 |
| `planning/` | 场景决策到机器人执行计划 |
| `robot/` | TF 查询、MoveIt 执行、夹爪运行时 |
| `workflows/` | 两阶段抓取、闭环堆叠等完整流程 |

从项目根目录执行脚本，以保证相对配置和模型路径正确。
