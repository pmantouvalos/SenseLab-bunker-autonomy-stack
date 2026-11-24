# Bunker Mini 2.0 – ROS 2 Autonomy Stack

![Status](https://img.shields.io/badge/Status-Under%20Development-yellow)
![ROS 2](https://img.shields.io/badge/ROS%202-Jazzy-blue)
![Gazebo](https://img.shields.io/badge/Gazebo-Harmonic-orange)
![Ubuntu](https://img.shields.io/badge/Ubuntu-24.04-brightgreen)

A modular ROS 2 autonomy stack for the **AgileX Bunker Mini 2.0** tracked UGV, developed at the **SenseLab Space Informatics** research laboratory (Technical University of Crete, director: Prof. Panagiotis Partsinevelos). The long-term goal is autonomous operation in GNSS-denied, subterranean-style environments, inspired by the DARPA SubT Challenge.

The stack currently runs end to end **in simulation** (Gazebo Harmonic, SubT-style cave world with a simulated OAK-D Lite camera). Deployment on the physical robot is the next milestone.

## Demo

▶ **[Watch the simulation demo (video, 54 s)](docs/media/demo.mp4)**: the rover follows the lane, detects rocks and drives around them in AUTO mode, while the dashboard shows the camera views, the lane mask and telemetry.

## What it does

1. **Perceives the lane.** A YOLOv8n-seg network segments the two lane lines. Mask pixels are projected onto the ground plane using the camera pose (TF) to obtain the lane centre in metres.
2. **Follows it.** A Stanley controller turns the lane centre into linear and angular velocity commands.
3. **Detects artifacts.** YOLO plus depth gives 3D detections (survivor, backpack, fire extinguisher, ...) in the map frame. Repeated detections are merged and each artifact is reported once.
4. **Arbitrates behaviour.** A state machine (`STANDBY`, `MANUAL`, `AUTO_*`, `ESTOP`) is the single publisher of `/cmd_vel`.
5. **Exposes a web dashboard** with camera views, manual driving, a software e-stop and the list of found artifacts.

```
camera ─► lane_detector ─► /lane/center_path ─► stanley_controller ─► /auto/cmd_vel ─┐
      └─► artifact_detector ─► artifact_reporter ─► /artifacts/report ───────────────┤
dashboard ─► /autonomy/command, /manual/cmd_vel ─────────────────────────────────────┤
                                                                                      ▼
                                                           autonomy_manager ─► /cmd_vel ─► robot
```

Only `autonomy_manager` writes to `/cmd_vel`; every other node publishes proposals on its own topic.

## Repository layout

| Path | Description |
|---|---|
| `src/mine_rover_core` | Autonomy nodes: lane detection, Stanley controller, artifact detection/reporting, obstacle detection, slalom planner, state machine, web dashboard, and dataset/training scripts. |
| `src/bunker_simulation` | Gazebo Harmonic world, robot URDF, OAK-D Lite model, launch files and track/mesh generators. |
| `src/bunker_ros2` | Base driver package for the Bunker (AgileX). |
| `src/ugv_sdk` | AgileX `ugv_sdk` (Apache-2.0), vendored third-party code, not written by this project. |

More detail is in [`src/mine_rover_core/docs/`](src/mine_rover_core/docs) (`AUTONOMY.md`, `GUIDE.md`, `YOLO_TRAINING.md`).

## Getting started

Requirements: Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic (`ros_gz_sim`, `ros_gz_bridge`), and a Python virtual environment with [`ultralytics`](https://github.com/ultralytics/ultralytics) (the launch file looks for `~/venvs/yolo`).

```bash
git clone https://github.com/pmantouvalos/SenseLab-bunker-autonomy-stack.git rover_ws
cd rover_ws
colcon build --packages-select bunker_simulation mine_rover_core
source install/setup.bash
ros2 launch bunker_simulation autonomy.launch.py    # add headless:=true to hide the Gazebo window
```

Open <http://localhost:8080> for the dashboard. Gazebo takes about a minute to load. Cave segments and artifact models are downloaded from [Gazebo Fuel](https://app.gazebosim.org/OpenRobotics) on first run.

The pretrained `models/lane_seg.pt` is included. To retrain it from the simulator:

```bash
python3 src/mine_rover_core/training/collect_lane_dataset.py --count 700 --out ~/datasets/lane_sim
~/venvs/yolo/bin/python src/mine_rover_core/training/train_lane_seg.py --epochs 40
```

Labels are generated automatically (colour thresholds on the simulated image), so no manual annotation is needed. On the real robot, new training images from the real camera will be required.

## Safety

The dashboard server listens on `127.0.0.1` only. The dashboard e-stop is **software only**; on the real Bunker Mini always keep the physical emergency stop within reach.

## Roadmap

- [x] ROS 2 Jazzy workspace and AgileX base drivers
- [x] Gazebo simulation: SubT-style cave world, robot URDF, OAK-D Lite
- [x] Lane perception (YOLOv8-seg) and Stanley lane keeping
- [x] 3D artifact detection and reporting
- [x] State machine and web dashboard
- [ ] Obstacle avoidance / slalom planner: validation
- [ ] LiDAR-based local planning and topological exploration
- [ ] SLAM integration and benchmarking in DARPA SubT-inspired worlds
- [ ] Hardware deployment and real-world testing

## References

- Zhang, Ji, et al. *Falco: Fast Likelihood-based Collision Avoidance...* JFR, 2020.
- Dang, Tung, et al. *Graph-based subterranean exploration path planning...* JFR, 2020.
- Scherer, Sebastian, et al. *Resilient and Modular Subterranean Exploration...* Field Robotics, 2022.

---

**Author:** Panagiotis Mantouvalos · Undergraduate Research Assistant, SenseLab Space Informatics, Technical University of Crete
