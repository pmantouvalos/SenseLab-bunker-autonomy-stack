#!/usr/bin/env python3
"""
Collects a labelled YOLO dataset of the 5 cartoon figures in the simulator, fully automatic.

The simulator knows where every figure is, so the labels are exact (no hand labelling):
    1. put the 5 figures at random places of the track (gz set_pose on the fig_* models),
    2. put the rover (teleport) at a random distance/angle from one of them, looking at it,
    3. take the camera image and project the 3D box of every figure into it -> YOLO boxes.
Images where a figure is only partly visible or hidden behind a rock are dropped (an unlabelled
figure would teach the network that it is background).

    # Gazebo must be running (ros2 launch bunker_simulation sim.launch.py headless:=true)
    python3 collect_figure_dataset.py --images 600 --out ~/datasets/figures

Writes <out>/images/{train,val}, <out>/labels/{train,val} and <out>/data.yaml, ready for
train_figures.py. Needs the same environment as the nodes (source install/setup.bash).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'mine_rover_core'))
from lane_utils import quat_to_matrix  # noqa: E402

WORLD = 'cave_circuit_practice_01'
CLASSES = ['mickey', 'homer', 'spiderman', 'pikachu', 'shrek']


def load_config():
    for p in (os.path.join(HERE, '..', '..', 'bunker_simulation', 'config', 'figures.json'),):
        if os.path.exists(p):
            return json.load(open(p))
    raise SystemExit('config/figures.json not found: run bunker_simulation/scripts/generate_figures.py --write')


def load_boulders():
    import importlib.util
    p = os.path.join(HERE, '..', '..', 'bunker_simulation', 'scripts', 'generate_track.py')
    spec = importlib.util.spec_from_file_location('generate_track', p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BOULDERS


def set_pose(name, x, y, z, yaw):
    try:
        _set_pose(name, x, y, z, yaw)
    except subprocess.TimeoutExpired:
        print('gz service timed out, continuing', flush=True)


def _set_pose(name, x, y, z, yaw):
    req = (f'name: "{name}" position {{x: {x:.3f} y: {y:.3f} z: {z:.3f}}} '
           f'orientation {{z: {math.sin(yaw / 2):.5f} w: {math.cos(yaw / 2):.5f}}}')
    subprocess.run(['gz', 'service', '-s', f'/world/{WORLD}/set_pose', '--reqtype', 'gz.msgs.Pose',
                    '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)


class Grabber(Node):
    def __init__(self):
        super().__init__('figure_collector')
        self.rgb = self.depth = self.K = self.gt = None
        self.create_subscription(Image, '/oak/rgb/image_raw', lambda m: setattr(self, 'rgb', (time.time(), m)), 1)
        self.create_subscription(Image, '/oak/stereo/image_raw', lambda m: setattr(self, 'depth', (time.time(), m)), 1)
        self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.on_info, 1)
        self.create_subscription(Odometry, '/ground_truth/odom', lambda m: setattr(self, 'gt', m), 1)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def on_info(self, m):
        self.K = list(m.k)

    def spin(self, seconds):
        t = time.time()
        while time.time() - t < seconds:
            rclpy.spin_once(self, timeout_sec=0.05)

    def fresh_images(self, after, timeout=6.0):
        t = time.time()
        while time.time() - t < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.rgb and self.depth and self.rgb[0] > after and self.depth[0] > after:
                return self.rgb[1], self.depth[1]
        return None

    def cam_pose_world(self):
        """(R, t) of the camera optical frame in the world, from ground truth + the static TF."""
        if self.gt is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform('base_footprint', 'oak_rgb_camera_optical_frame', Time())
        except Exception:
            return None
        q, p = self.gt.pose.pose.orientation, self.gt.pose.pose.position
        Rb, tb = quat_to_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])
        qc, pc = tf.transform.rotation, tf.transform.translation
        Rc, tc = quat_to_matrix(qc.x, qc.y, qc.z, qc.w), np.array([pc.x, pc.y, pc.z])
        return Rb @ Rc, Rb @ tc + tb


def project_box(K, R, t, centre, size):
    """Image box (x1,y1,x2,y2), centre depth and 'all in front' flag for an upright 3D box."""
    cx, cy, cz = centre
    sx, sy, sz = size
    corners = np.array([[cx + dx * sx / 2, cy + dy * sy / 2, cz + dz * sz / 2]
                        for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)])
    cam = (corners - t) @ R                      # world -> camera (R maps camera -> world)
    if (cam[:, 2] < 0.3).any():
        return None
    u = K[0] * cam[:, 0] / cam[:, 2] + K[2]
    v = K[4] * cam[:, 1] / cam[:, 2] + K[5]
    c = (np.array(centre) - t) @ R
    return (u.min(), v.min(), u.max(), v.max()), float(c[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', type=int, default=600)
    ap.add_argument('--out', default=os.path.expanduser('~/datasets/figures'))
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    cfg = load_config()
    boulders = load_boulders()
    rnd = random.Random(a.seed)
    for sub in ('images/train', 'images/val', 'labels/train', 'labels/val'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)

    rclpy.init()
    g = Grabber()
    print('waiting for the simulator ...')
    while g.K is None or g.gt is None or g.rgb is None:
        g.spin(0.5)
    saved = dropped = 0
    layout = {}
    frames_in_layout = 0

    def rock_near(x, y, margin):
        return any(abs(x - b['x']) < b['size'][0] / 2 + margin and abs(y - b['y']) < b['size'][1] / 2 + margin
                   for b in boulders)

    it = 0
    t_start = time.time()
    while saved < a.images:
        it += 1
        if it <= 12 or it % 20 == 0:
            print(f'iter {it} t={time.time() - t_start:.0f}s saved={saved} dropped={dropped}', flush=True)
        if frames_in_layout == 0:
            layout = {}
            for n in CLASSES:
                for _ in range(1000):
                    x, y = rnd.uniform(1.0, 10.8), rnd.uniform(-0.8, 0.8)
                    if not rock_near(x, y, 0.3) and all(math.hypot(x - p[0], y - p[1]) > 0.5 for p in layout.values()):
                        layout[n] = (x, y, rnd.uniform(-math.pi, math.pi))
                        break
                set_pose(f'fig_{n}', *layout[n][:2], 0.0, layout[n][2])
        frames_in_layout = (frames_in_layout + 1) % 8

        # a rover pose looking at a random figure from a random side
        n = rnd.choice(CLASSES)
        fx, fy, _ = layout[n]
        d = rnd.uniform(0.8, 3.3)
        theta = rnd.uniform(-0.6, 0.6) + (math.pi if rnd.random() < 0.25 else 0.0)
        rx, ry = fx - d * math.cos(theta), fy - d * math.sin(theta)
        if not (-0.3 <= rx <= 11.5 and abs(ry) <= 1.1) or rock_near(rx, ry, 0.5):
            continue
        yaw = math.atan2(fy - ry, fx - rx) + rnd.uniform(-0.25, 0.25)
        set_pose('bunker_mini', rx, ry, 0.1, yaw)
        g.spin(0.9)                               # let it settle
        t0 = time.time()
        got = g.fresh_images(t0)
        pose = g.cam_pose_world()
        if got is None or pose is None:
            continue
        rgb_msg, depth_msg = got
        R, t = pose
        img = np.frombuffer(rgb_msg.data, np.uint8).reshape(rgb_msg.height, rgb_msg.width, -1)[:, :, ::-1].copy()
        depth = np.frombuffer(depth_msg.data, np.float32).reshape(depth_msg.height, depth_msg.width)
        h, w = img.shape[:2]

        labels, bad, why = [], False, []
        for ci, name in enumerate(CLASSES):
            x, y, _ = layout[name]
            sx, sy, sz = cfg[name]['size']
            pr = project_box(g.K, R, t, (x, y, sz / 2), (sx, sy, sz))
            if pr is None:                       # (partly) behind the camera
                if math.hypot(x - t[0], y - t[1]) < 1.0 and (np.array([x, y, 0.15]) - t) @ R[:, 2] > -0.2:
                    bad = True                    # right next to the camera, in front: partly seen
                    why.append(f'{name}:near')
                continue
            (x1, y1, x2, y2), dist = pr
            full = max(1.0, (x2 - x1) * (y2 - y1))
            cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            vis = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1) / full
            if vis < 0.02:
                continue                          # not in the picture
            if (x2 - x1) < 10 or (y2 - y1) < 16:
                continue                          # too small to matter (background)
            if vis < 0.55:
                bad = True                        # mostly cut by the image border: ambiguous
                why.append(f'{name}:cut{vis:.2f}')
                continue
            # hidden behind something closer? (the figure itself is the nearest thing in its box)
            ys = slice(int(cy1 + 0.2 * (cy2 - cy1)), int(cy2 - 0.1 * (cy2 - cy1)))
            xs = slice(int(cx1 + 0.25 * (cx2 - cx1)), int(cx2 - 0.25 * (cx2 - cx1)))
            roi = depth[ys, xs]
            roi = roi[np.isfinite(roi)]
            if roi.size == 0:
                bad = True
                continue
            closer = float((roi < dist - 0.45).mean())      # share of pixels in front of the figure
            if closer > 0.85:
                continue                          # fully hidden behind a rock: correctly unlabelled
            if closer > 0.03:
                bad = True                        # partly hidden: ambiguous
                why.append(f'{name}:partly_hidden{closer:.2f}')
                continue
            labels.append(f'{ci} {(cx1 + cx2) / 2 / w:.6f} {(cy1 + cy2) / 2 / h:.6f} '
                          f'{(cx2 - cx1) / w:.6f} {(cy2 - cy1) / h:.6f}')
        if bad:
            dropped += 1
            if dropped <= 15:
                print('drop', why, flush=True)
            continue
        split = 'val' if rnd.random() < 0.15 else 'train'
        stem = f'fig_{a.seed}_{saved:05d}'
        cv2.imwrite(os.path.join(a.out, 'images', split, stem + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(a.out, 'labels', split, stem + '.txt'), 'w') as f:
            f.write('\n'.join(labels) + ('\n' if labels else ''))
        saved += 1
        if saved % 25 == 0 or saved < 4:
            print(f'{saved}/{a.images} saved, {dropped} dropped', flush=True)

    with open(os.path.join(a.out, 'data.yaml'), 'w') as f:
        f.write(f'path: {a.out}\ntrain: images/train\nval: images/val\nnames:\n')
        for i, n in enumerate(CLASSES):
            f.write(f'  {i}: {n}\n')
    print(f'done: {saved} images in {a.out} ({dropped} dropped)')
    set_pose('bunker_mini', 0, 0, 0.1, 0)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
