# Rover autonomy: lane, Stanley controller, state machine, dashboard

## Quick start (one command)

```bash
cd rover_ws
colcon build --packages-select bunker_simulation mine_rover_core
source install/setup.bash
ros2 launch bunker_simulation autonomy.launch.py          # add headless:=true to hide the Gazebo window
```

Open **http://localhost:8080**. Gazebo takes about a minute to load.
The launch file locates the YOLO virtualenv on its own (`~/venvs/yolo`).

## Data flow

```
OAK-D Lite ─► lane_detector (YOLOv8-seg) ─► /lane/center_path ─► stanley_controller ─► /auto/cmd_vel ─┐
           └► artifact_detector (YOLO) ─► artifact_reporter ─► /artifacts/report ──────────────┐      │
                                                                                                ▼      ▼
dashboard ─► /autonomy/command, /manual/cmd_vel ───────────────────────────────────────► autonomy_manager ─► /cmd_vel ─► robot
   ▲                                                                                        │
   └────────────────────────────── /autonomy/state ◄────────────────────────────────────────┘
```

**Rule:** only `autonomy_manager` writes to `/cmd_vel`. Every other node publishes
proposals on its own topic, and the manager decides which one passes depending on the state.

## The lane

`bunker_simulation/scripts/generate_lane.py` creates two continuous white lines (lane width
1.8 m) that start straight, bend gently to the left (1.2 m) and straighten again. The length is
12 m: the floor of the start area ends at ~12.5 m (there is a gap after that), so the lane must
not exceed it. The lines are visual only (no collision). To change the shape, edit the constants
at the top of the script and run `python3 scripts/generate_lane.py --write`. The artifacts
(backpack, fire extinguisher, survivor) stand just **outside** the lines so the lane stays free.

## The lane network

* Model: **YOLOv8n-seg** with a single class (`lane_line`). It outputs a pixel-wise mask.
* The mask pixels are projected onto the ground using the camera pose (TF), giving the left
  line, the right line and the lane centre in metres. If only one line is visible, the centre is
  obtained by offsetting it by half the lane width.
* **Training (automatic, no manual labelling):** with the simulator running,

```bash
python3 src/mine_rover_core/training/collect_lane_dataset.py --count 700 --out ~/datasets/lane_sim
~/venvs/yolo/bin/python src/mine_rover_core/training/train_lane_seg.py --epochs 40
colcon build --packages-select mine_rover_core     # installs lane_seg.pt
```

  The collector teleports the robot to random positions and angles around the lane and
  generates the labels with colour thresholds (`lane_utils.white_mask`). The network learns from
  them and ends up working where the thresholds fail. On the **real robot** you will need images
  from the real camera (e.g. labelled with Roboflow) and a new training run.
* If `lane_seg.pt` is missing, `lane_detector` falls back to colour thresholds (it says so in
  the log and in the dashboard).

## Stanley controller

`δ = ψ_e + atan(k·e / (k_soft + v))`, where `e` is the lateral distance of the front axle from
the lane centre and `ψ_e` is the angle between the robot heading and the lane. δ is converted to
an angular velocity `ω = v·tan(δ)/L` (the robot is tracked, so `L` is a virtual wheelbase).
Parameters are in `stanley_controller.py` (`k`, `k_soft`, `wheelbase`, `v_max`, `max_steer`, ...).

## State machine (`autonomy_manager.py`)

| State | What it does | How you leave it |
|---|---|---|
| `STANDBY` | stationary | MANUAL / AUTO |
| `MANUAL` | passes the dashboard command (dead-man 0.5 s) | AUTO / STANDBY |
| `AUTO_ACQUIRE` | stationary, waits to see the lane for 0.5 s | found → `AUTO_FOLLOW`, 10 s without → `STANDBY` |
| `AUTO_FOLLOW` | passes the Stanley command | lane lost for 1 s → `AUTO_LANE_LOST` |
| `AUTO_LANE_LOST` | stationary | lane seen again → `AUTO_FOLLOW`, 6 s → `STANDBY` |
| `AUTO_ARTIFACT_HOLD` | stops for `artifact_hold_s` seconds at each new finding (optional, default 0 = disabled) | time elapsed → `AUTO_FOLLOW` |
| `ESTOP` | zero velocity at 20 Hz, locked | **RESET only** → `STANDBY` |

`ESTOP` is reachable from every state. To add a new behaviour (e.g. return to base, obstacle
avoidance): add a state name, a `_tick_<name>` method that returns the Twist and, if the operator
selects it, a command in `on_command`.

## Dashboard

Buttons MANUAL / AUTO / Stop, **EMERGENCY STOP** (always visible at the bottom; Space key) and
RESET. Manual driving with W A S D / arrow keys or with the buttons (hold to drive, release to
stop). Two camera views (object YOLO, and the lane with the network mask and centre line). When an
artifact is found, a flashing red "FOUND: ..." banner appears with a sound, and the finding is
added to the list.

Safety: the server listens on `127.0.0.1` only. With `host:=0.0.0.0` you can open it from another
device, but then anyone on the network can drive the robot. The dashboard emergency stop is
**software**; on the real Bunker Mini also keep the physical e-stop within reach.

## Files

| File | Role |
|---|---|
| `bunker_simulation/scripts/generate_lane.py` | generates the lane inside the world |
| `bunker_simulation/launch/autonomy.launch.py` | starts everything |
| `mine_rover_core/lane_utils.py` | math: mask, ground projection, centre fit |
| `mine_rover_core/lane_detector.py` | lane network → `/lane/center_path` |
| `mine_rover_core/stanley_controller.py` | Stanley law → `/auto/cmd_vel` |
| `mine_rover_core/autonomy_manager.py` | state machine, sole publisher of `/cmd_vel` |
| `mine_rover_core/dashboard_server.py`, `dashboard/index.html` | web dashboard |
| `training/collect_lane_dataset.py`, `training/train_lane_seg.py` | dataset and training |
