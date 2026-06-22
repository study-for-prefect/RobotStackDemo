# RobotStackDemo

UR5 + RealSense D435i 的积木识别、抓取与堆叠演示项目。

## 目录结构

```text
RobotStackDemo/
├── config/                  # 检测器、机械臂预备位等配置
├── docs/                    # 标定与工作流说明
├── models/                  # 模型权重路径和训练结果
├── robot_scene_pipeline/    # 可复用的感知、几何、TF 与规划逻辑
├── tests/                   # 不依赖真实机器人运行的测试
├── tools/
│   ├── calibration/         # hover/TCP/XY 偏差标定
│   ├── data/                # RealSense 数据采集
│   ├── diagnostics/         # 感知稳定性诊断
│   ├── monitoring/          # 实时检测监控
│   ├── planning/            # 决策到执行计划的转换
│   ├── robot/               # ROS 2、TF、MoveIt、夹爪接口
│   └── workflows/           # 完整抓取、堆叠工作流
└── training/                # 数据转换和 YOLO 训练脚本
```

## 常用入口

```bash
# 单帧感知
python3 -m robot_scene_pipeline.snapshot_pipeline --skip-llm

# hover 标定
python3 tools/calibration/hover_tool_offset_calibration.py --label green

# XY 偏差采集和分析
python3 tools/calibration/xy_bias_diagnosis.py --help

# 两阶段视觉抓取
python3 tools/workflows/two_stage_visual_pick.py --help

# 闭环堆叠
python3 tools/workflows/stack_demo_pipeline.py --help

# 测试
python3 -m unittest discover -s tests
```

涉及真实机械臂运动的命令默认只规划或采集；确认 UR5、MoveIt、TF、相机和夹爪状态后，再显式添加 `--execute`。

详细说明见：

- [Hover 与 XY 标定](docs/HOVER_XY_CALIBRATION.md)
- [LLM 堆叠入口](docs/LLM_STACK_BLOCKS.md)
- [感知流水线](robot_scene_pipeline/README.md)
