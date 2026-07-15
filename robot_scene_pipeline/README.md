# Robot Scene Pipeline

本目录只负责共享感知、几何和底层安全能力，不再包含开放式 LLM/VLM 动作规划。任务决策在
`tools/workflows/stack_demo/` 中完成：代码先生成 `PhysicalActionEdge`，Qwen 只选择候选 ID。

## 主要模块

- `snapshot_pipeline.py`：单帧 RGB-D 感知编排，只输出观测产物。
- `perception_server.py` / `perception_runtime.py`：供工作流获取最新观测的常驻感知服务。
- `ros_topic_capture.py`：从已运行的 ROS 2 相机话题获取同步 RGB-D。
- `realsense_capture.py`：显式请求时使用的旧 direct RealSense 路径。
- `detector_runtime.py`：YOLO 检测。
- `depth_geometry.py` / `tabletop_geometry.py`：三维中心、尺寸、yaw 和桌面估计。
- `tf_transform.py`：读取最新 TF，将 optical-frame 点变换到 `base_link`。
- `object_tracking.py`：跨 revision 的显式 track 重绑定；不把 detector ID 当持久身份。
- `grasp_yaw_search.py`：连续抓取 yaw 和安全区间的底层几何能力。
- `tool_swept_volume.py`：GF225 指尖/掌部的分段扫掠检查。
- `task_geometry.py`：共享的 footprint 和区域几何谓词。
- `visual_color.py` / `object_semantics.py`：感知语义归一化。
- `ollama_policy_client.py`：无状态、有界 Ollama 单轮调用基础设施。

已删除快照 LLM reasoning、开放式 action/stack policy、task contract、grounded action adapter、
旧 replan/schema 和 action compiler。`snapshot_pipeline.py` 不接收 instruction，也不生成机器人
动作。

## 坐标、TF 与身份

D435i 安装在末端。在线观测必须读取当前 `base_link <- camera_color_optical_frame` TF；场景、
机械臂、相机或物体发生变化后不得复用旧坐标。默认对齐深度话题是：

```text
/camera/camera/color/image_raw
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/color/camera_info
```

对齐深度反投影点位于 `camera_color_optical_frame`，应使用 direct optical-frame TF。只有有意
走旧 camera-link 路径时才使用 optical-to-camera-link 转换。

detector `object_id` 只在当前 scene revision 中有效。跨帧状态只能使用 `track_id` 和显式
重绑定结果；歧义绑定不能进入动作候选。

手眼标定的静态 TF 发布由已有外部 ROS 系统负责，本仓库的工作流不会启动或重复启动驱动。
可在驱动已运行时只读验证 TF：

```bash
python -m robot_scene_pipeline.tf_transform \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame
```

## 离线感知

只读本地图像、不会访问相机或机器人：

```bash
conda run -n scene_graph_benchmark \
  python -m robot_scene_pipeline.snapshot_pipeline \
  --image-in input_dir/1.jpg \
  --output-dir /tmp/robot_scene_pipeline_test \
  --device cpu
```

快照产物包括原图、detector overlay、候选、三维对象、private scene state、TF 状态和可选桌面
几何。它们都是感知事实，不包含 LLM 决策或可执行动作。

## 在线感知

RealSense 驱动必须由操作者在本工作流之外维护。驱动已运行后，可获取一帧并使用最新 TF：

```bash
conda run -n scene_graph_benchmark \
  python -m robot_scene_pipeline.snapshot_pipeline \
  --use-tf \
  --base-frame base_link \
  --camera-frame camera_color_optical_frame \
  --estimate-tabletop \
  --output-dir /tmp/robot_scene_geometry
```

若要人工确认后再拍照，增加 `--capture-trigger enter`。不同 namespace 可用实际
`--color-topic`、`--depth-topic` 和 `--camera-info-topic` 覆盖。在线 ROS topic 捕获的 Python
环境必须能导入 `rclpy`、`sensor_msgs`。

工作流默认通过感知服务请求 fresh observation；显式允许时才回退到快照子进程：

```bash
python3 -m robot_scene_pipeline.perception_server \
  --host 127.0.0.1 \
  --port 8765
```

上面的服务启动命令仅是文档；本次架构重构没有启动相机、机器人或服务进程。

## 与 Stack Demo 的边界

感知层提供 RGB-D、内参、实时 TF、对象几何、track、邻接和底层碰撞能力。它不决定：

- 当前优先处理哪个对象；
- 抓取 yaw、推动方向/距离或放置坐标；
- 颜色整理是否完成；
- 房屋角色绑定、姿态或结构是否完成。

这些由 stack demo 的代码状态、物理边生成器和任务 planner 负责。Qwen 的无状态候选选择协议、
抓取后重新观测门控和运行命令见 `tools/workflows/stack_demo/README.md`。

## 安全说明

MoveIt plan-only 成功不是 GF225 无碰撞证明。所有动作仍必须经过代码侧指尖、掌部、下降、
抬升、运输、释放、退回或推动分段扫掠检查。工作区和 GF225 参数由 stack demo 的统一配置
加载，不应在感知模块重复硬编码。
