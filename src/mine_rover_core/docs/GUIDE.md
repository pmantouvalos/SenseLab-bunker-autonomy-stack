# System guide: what it does and what to change on the real robot

## 1. What the system does today

A virtual Bunker Mini (in Gazebo, cave world) with an OAK-D Lite camera:

1. **Sees** the two white lane lines with a neural network (YOLOv8n-seg) and computes the lane centre in metres.
2. **Drives** by following the centre with the Stanley control law.
3. **Recognises artifacts** (person, backpack, ...) with YOLO, localises them in 3D and reports each one once.
4. **Decides** with a state machine (STANDBY / MANUAL / AUTO / ESTOP).
5. **Displays** everything on a web dashboard (http://localhost:8080): cameras, buttons, manual driving, e-stop, list of findings.

Start: `ros2 launch bunker_simulation autonomy.launch.py`. Stop: Ctrl+C.

Data flow:

```
camera ─► lane_detector ─► /lane/center_path ─► stanley_controller ─► /auto/cmd_vel ─┐
      └─► artifact_detector ─► artifact_reporter ─► /artifacts/report ───────────────┤
dashboard ─► /autonomy/command, /manual/cmd_vel ─────────────────────────────────────┤
                                                                                     ▼
                                                               autonomy_manager ─► /cmd_vel ─► robot
```

**Rule:** the decision "which velocity command passes" is made only in `autonomy_manager`.
(One small exception: `dashboard_server.py` also sends zero velocity to `/cmd_vel` on
EMERGENCY STOP, see line ~152. It is an extra safety measure, not normal driving.)

## 2. The central file of the autonomy

**`src/mine_rover_core/mine_rover_core/autonomy_manager.py`**

The state machine lives here. To add a behaviour (return to base, obstacle avoidance, ...):

1. add the state name to `STATES` (and to `MODE_OF`) at the top of the file,
2. write a `_tick_<name>` method that returns the `Twist` (or `None` to stop) and calls `self.enter(...)` to change state,
3. if the operator selects it, add a command in `on_command` (and in `COMMANDS` in `dashboard_server.py`).

States: `STANDBY` (stationary) · `MANUAL` (dashboard passes, dead-man 0.5 s) · `AUTO_ACQUIRE` (waits to see the lane) · `AUTO_FOLLOW` (Stanley passes) · `AUTO_LANE_LOST` (stationary, searching) · `AUTO_ARTIFACT_HOLD` (optional stop at a new artifact) · `ESTOP` (locked, leave only with RESET).

> Note: the old `state_manager.py` is something else: a simple timer-based sequence
> "forward 4 s, turn 180°, return", without sensors. It is not used by
> `autonomy.launch.py`. The real autonomy is written in `autonomy_manager.py`.

## 3. What each file does

Folder `src/mine_rover_core/mine_rover_core/`:

| File | What it does | Listens to | Publishes |
|---|---|---|---|
| `autonomy_manager.py` | state machine, selects which command reaches the robot | `/autonomy/command`, `/manual/cmd_vel`, `/auto/cmd_vel`, `/lane/status`, `/artifacts/report` | `/cmd_vel` (20 Hz), `/autonomy/state` |
| `stanley_controller.py` | Stanley control law: turns the lane centre into linear and angular velocity | `/lane/center_path` | `/auto/cmd_vel`, `/auto/stanley_debug` |
| `lane_detector.py` | runs the lane network, projects pixels onto the ground, finds the centre | `/oak/rgb/image_raw`, `/oak/rgb/camera_info` | `/lane/center_path`, `/lane/status`, `/lane/debug_image` |
| `lane_utils.py` | pure math (white mask, ground projection, centre fit). No ROS | | |
| `artifact_detector.py` | object YOLO + depth, gives 3D detections | `/oak/rgb/camera_info`, image and depth | `/oak/nn/spatial_detections`, `/oak/nn/detections_image` |
| `artifact_reporter.py` | merges repeated detections of the same artifact and reports it once | `/oak/nn/spatial_detections` | `/artifacts/report`, `/artifacts/markers` |
| `oak_points_relay.py` | relays the camera point cloud with the correct frame_id | `/oak/points_raw` | `/oak/points` |
| `dashboard_server.py` | web server + dashboard (folder `dashboard/`) | state, reports, images, `/odom` | `/autonomy/command`, `/manual/cmd_vel` |
| `state_manager.py` | OLD timer-based demo (see above) | | `/cmd_vel` |

Other:

| File | What it does |
|---|---|
| `bunker_simulation/launch/autonomy.launch.py` | starts everything (sim, detection, lane, Stanley, manager, dashboard) |
| `bunker_simulation/launch/sim.launch.py`, `detection.launch.py` | Gazebo and artifact detection |
| `bunker_simulation/scripts/generate_lane.py` | generates the lane in the world (simulation only) |
| `mine_rover_core/training/collect_lane_dataset.py`, `train_lane_seg.py` | automatic image collection from the sim and training of `models/lane_seg.pt` |
| `docs/AUTONOMY.md`, `docs/YOLO_TRAINING.md` | documentation |

## 4. What to change/check on the REAL robot

Parameters are declared with `declare_parameter` at the top of each node. You can change them in
the code or in the launch file (`parameters=[{'k': 1.0}]`), without recompiling when using the launch file.

### Definitely need a change or a check

| What | Where | Why |
|---|---|---|
| `use_sim_time` | `autonomy.launch.py` (`sim_time = {'use_sim_time': True}`) | the real robot has no `/clock` from Gazebo: set `False` or remove it. Otherwise the nodes will not advance |
| **Lane network** | `models/lane_seg.pt` | trained on simulation images. It needs images from the real camera and a new training run. Otherwise it falls back to colour thresholds (`lane_utils.white_mask`) that need tuning for real lighting |
| `lane_width` | `lane_detector.py` | real lane width (m). Used when only one line is visible |
| `x_min`, `x_max` | `lane_detector.py` | how far ahead of the robot you trust the floor. Depends on camera height and tilt |
| Camera height and tilt (TF) | URDF / `sim.launch.py` | the pixel → ground projection uses the camera pose. If it does not match the real robot, the lane centre will be wrong |
| `v_max`, `v_min`, `max_omega` | `stanley_controller.py` (0.5 / 0.2 / 1.0) | start much lower in the first tests |
| `k`, `k_soft` | `stanley_controller.py` (1.2 / 0.3) | control gains: large `k` = more aggressive correction, oscillations. Tune these experimentally |
| `wheelbase`, `front_axle_x`, `max_steer` | `stanley_controller.py` (0.5 / 0.3 / 0.6) | the Bunker is tracked, so `wheelbase` is virtual. Set it so turns look natural |
| `max_manual_lin`, `max_manual_ang` | `autonomy_manager.py` (0.8 / 1.5) | manual driving limits |
| `host` | `dashboard_server.py` (`127.0.0.1`) | if you open the dashboard from another device with `0.0.0.0`, anyone on the network can drive the robot |
| Physical e-stop | hardware | the dashboard ESTOP is software. Always keep the physical button |

### State machine timeouts (`autonomy_manager.py`)

| Parameter | Default | Meaning |
|---|---|---|
| `manual_timeout` | 0.5 s | no manual command → stop (dead-man) |
| `auto_timeout` | 0.5 s | no Stanley command → stop |
| `acquire_time` | 0.5 s | how long the lane must be visible before starting |
| `acquire_timeout` | 10 s | how long it waits to find a lane before giving up |
| `lane_lost_time` | 1.0 s | no lane for this long → `AUTO_LANE_LOST` |
| `lane_lost_timeout` | 6.0 s | lost for this long → `STANDBY` |
| `artifact_hold_s` | 0 | >0: stops for this many seconds at each new artifact |

On the real robot, with a slower or noisier camera, you may need to increase the `lane_lost_*` values.

### Artifact detection

| What | Where | Meaning |
|---|---|---|
| `model` | `artifact_detector.py` (`yolov8n.pt`) | COCO recognises only `person` well. Backpack and fire extinguisher need fine-tuning (`docs/YOLO_TRAINING.md`) |
| `classes` | launch argument | which classes are reported |
| `confidence` | `artifact_detector.py` (0.5), `lane_detector.py` (0.3) | confidence threshold |
| `roi_scale` | `artifact_detector.py` (0.3) | what fraction of the box is used for the depth measurement |
| `merge_radius` | `artifact_reporter.py` (1.0 m) | two detections closer than this are considered the same artifact |
| `min_observations`, `min_score` | `artifact_reporter.py` (5 / 0.5) | how many times and with what confidence it must be seen to be reported (false-positive filter) |
| `fixed_frame` | `artifact_reporter.py` (`odom`) | the coordinate frame in which positions are stored. `odom` drifts over time |
| Camera topics | `/oak/rgb/...`, `/oak/points` | on the real robot the names depend on the OAK-D driver. Check with `ros2 topic list` |

### Simulation only (not transferable)

`generate_lane.py`, the world, the floor limit at x≈12.5 m, the automatic dataset collection (robot teleportation).

## 5. Test order on the real robot

1. `MANUAL` only, from the dashboard, with low speed limits. Check that the robot stops when you release the key and on ESTOP.
2. Look at the cameras and `/lane/debug_image`: does the network find the lines?
3. `AUTO` with a very low `v_max` (e.g. 0.15) and someone next to the physical e-stop.
4. Tune `k` and `k_soft` according to the oscillations.
5. Then enable the artifacts.

## Obstacle avoidance (slalom)

| File | What it does |
|---|---|
| `obstacle_detector.py` | Depth image → obstacles in metres (`/obstacles`). Finds the floor plane with RANSAC (works when the robot tilts too). `blocking` = height ≥ 0.25 m and width ≥ 0.35 m. |
| `slalom_planner.py` | Remembers the rocks (only measurements from 0.6–2.6 m, the closest measurement wins), computes the passage and publishes the curve `/avoid/path` + `/avoid/status`. Parameter `pose_topic`: in the sim = `/ground_truth/odom`. |
| `stanley_controller.py` | Started twice: once for the lane lines, once (`stanley_avoid`) for the avoidance curve. |
| `autonomy_manager.py` | New states `AUTO_OBSTACLE_STOP` → `AUTO_AVOID` → `AUTO_FOLLOW` and `AUTO_BLOCKED`. In AVOID it limits the speed to 0.25 m/s (`avoid_speed`). |

On the real robot: set `pose_topic` to an odometry that fuses wheels + IMU (wheel odometry turns incorrectly on tracks).
