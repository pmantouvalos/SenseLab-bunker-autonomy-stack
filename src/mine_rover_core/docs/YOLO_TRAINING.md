# YOLO artifact detection: install, run, train, deploy

A "rescue rover in a dark cave" scenario: the Bunker Mini with an OAK-D Lite camera finds
"artifacts" (survivor, backpack, fire extinguisher, helmet) as in DARPA SubT, and shows on screen
what it found and where.

```
OAK-D Lite (RGB + depth)
      │
      ▼
artifact_detector      YOLO + depth → 3D position relative to the camera
  (Gazebo only; on the real robot the camera itself does this)
      │  /oak/nn/spatial_detections  (vision_msgs/Detection3DArray)
      ▼
artifact_reporter      camera → map (TF), merging, confirmation after 5 detections
      │  /artifacts/markers, /artifacts/report
      ▼
RViz
```

---

## 1. Install YOLO (once)

The system has no system-wide `pip`. YOLO (the `ultralytics` package) goes into a separate
virtualenv so it does not interfere with the ROS packages.

```bash
sudo apt install python3-pip python3-venv
python3 -m venv ~/venvs/yolo
~/venvs/yolo/bin/pip install ultralytics "numpy<2"
```

> `numpy<2` is required: `cv_bridge` in ROS Jazzy is built against numpy 1.x and breaks with numpy 2.

In every terminal that runs the detector, add the venv packages to ROS's Python:

```bash
export PYTHONPATH=$PYTHONPATH:$HOME/venvs/yolo/lib/python3.12/site-packages
```

The GPU is used automatically if `torch` sees CUDA:
`~/venvs/yolo/bin/python -c "import torch; print(torch.cuda.is_available())"`.

---

## 2. Run the demo in simulation

```bash
# Terminal 1: cave + robot (the open-loop moves it 2 m forward, towards the artifacts)
source install/setup.bash
ros2 launch bunker_simulation sim.launch.py

# Terminal 2: detection + RViz
source install/setup.bash
export PYTHONPATH=$PYTHONPATH:$HOME/venvs/yolo/lib/python3.12/site-packages
ros2 launch bunker_simulation detection.launch.py rviz:=true
```

Keyboard driving, with `run_open_loop:=false` in terminal 1:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

The first time, `yolov8n.pt` is downloaded automatically. It is pretrained on COCO (80
categories), so it **already finds `person` and `backpack`**. It does **not** know fire
extinguishers: that is the motivation for your own model (step 3).

What to look at:
- `ros2 topic echo /artifacts/report`: one line for each confirmed artifact
- RViz: the image with the boxes, the robot, the point cloud and the artifact markers

---

## 3. Train your own model

### 3.1 Choose classes
Few and specific, using objects you will actually have at hand, e.g.:
`survivor` (mannequin or person), `backpack`, `fire_extinguisher`, `helmet`.

### 3.2 Collect images
- **From the real OAK-D Lite** (most important): 150–300 images per class.
- Variety: distances 0.5–6 m, angles, partial occlusion, and **darkness with only the robot's
  floodlight**, as in the target scenario.
- Add 10–20% images **without** objects (reduces false positives).
- Optionally, images from Gazebo (`/oak/rgb/image_raw`), so the same model works in simulation too.

### 3.3 Label the boxes
With [Roboflow](https://roboflow.com) (free for small projects) or CVAT: draw a box around each
object and export in **"YOLOv8"** format. Split 80% train / 20% validation. The structure you get:

```
dataset/
├── data.yaml          # paths + class names
├── train/images  train/labels
└── valid/images  valid/labels
```

```yaml
# data.yaml
path: ~/datasets/subt_artifacts
train: train/images
val: valid/images
names: [survivor, backpack, fire_extinguisher, helmet]
```

### 3.4 Training (on a CUDA GPU)
```bash
source ~/venvs/yolo/bin/activate
yolo detect train data=dataset/data.yaml model=yolov8n.pt imgsz=640 epochs=100 batch=16 device=0
```
- `yolov8n` (nano): the smallest model, suitable for the OAK-D Lite chip.
- You start from the pretrained model (transfer learning), so a few hundred images are enough.
- Result: `runs/detect/train/weights/best.pt`. Check `mAP50` in `runs/detect/train/results.png`
  (above ~0.8 is good for a demo).

### 3.5 Test in simulation
```bash
ros2 launch bunker_simulation detection.launch.py rviz:=true \
  model:=$HOME/runs/detect/train/weights/best.pt \
  classes:='survivor,backpack,fire_extinguisher,helmet'
```
If the dataset contains the class `survivor` instead of `person`, that is the name that will appear.

---

## 4. Running on the OAK-D Lite (real robot)

The OAK-D Lite has its own processor (RVC2) and runs YOLO **inside the camera**. It also
computes the 3D position of each box from the stereo depth, so the laptop does not need a GPU.
The model must first be converted:

1. **Convert** `best.pt` with **Luxonis Hub** (the tool Luxonis recommends today; the old
   tools.luxonis.com is now "legacy"): platform **RVC2**, input size a multiple of 32 (e.g.
   `640x352` or `416x416`). It produces an NN Archive (`.tar.xz`) for DepthAI v3.
   Instructions: <https://docs.luxonis.com/software-v3/ai-inference/conversion>
2. **Run with 3D (spatial) detection** on the camera, publishing to the **same** topic
   `/oak/nn/spatial_detections`. This way `artifact_reporter` and RViz work without any change.
   This part will be set up once the camera is connected, so it can be tested in practice.

On the real robot you therefore do **not** run `artifact_detector`. You run the camera driver
and `artifact_reporter`.

---

## Files

| File | What it does |
|---|---|
| `bunker_simulation/urdf/oak_d_lite.urdf.xacro` | Camera model: specs, pose, Gazebo sensor |
| `bunker_simulation/config/bridge.yaml` | `/oak/rgb/...`, `/oak/stereo/...`, `/oak/points` |
| `bunker_simulation/worlds/cave_circuit_practice_01.sdf` | 3 SubT artifacts in the cave |
| `mine_rover_core/artifact_detector.py` | YOLO + depth → 3D (simulation only) |
| `mine_rover_core/artifact_reporter.py` | 3D → map, confirmation, markers (sim + real) |
| `bunker_simulation/launch/detection.launch.py` | Starts detector, reporter, RViz |
| `bunker_simulation/config/demo.rviz` | The RViz view |
