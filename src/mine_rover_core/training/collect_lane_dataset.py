#!/usr/bin/env python3
"""
Builds a lane-segmentation dataset from the Gazebo simulation, with no hand labelling.

How it works
  1. The simulation must be running (ros2 launch bunker_simulation sim.launch.py ...).
  2. For every sample the robot is *teleported* (gz set_pose service) to a random pose
     on and around the lane: random position along it, sideways offset and heading.
  3. The camera image is saved and the white paint is segmented with colour
     thresholds (lane_utils.white_mask). The outline of every white blob is written as
     a YOLO-segmentation polygon, class 0 = lane_line.
  4. train_lane_seg.py then trains a neural network on these images. The network sees
     the whole picture and learns what a lane line *looks like*, so at run time it also
     works where thresholds fail (shadows, glare, partly worn paint, ...).

Usage (from the workspace root, after `source install/setup.bash`):
    python3 src/mine_rover_core/training/collect_lane_dataset.py --count 800 \
        --out ~/datasets/lane_sim
"""
import argparse
import math
import os
import random
import subprocess
import sys

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from mine_rover_core.lane_utils import mask_to_polygons, white_mask  # noqa: E402

sys.path.insert(0, os.path.join(get_package_share_directory('bunker_simulation'), 'scripts'))
import generate_lane as lane  # noqa: E402  (single source of truth for the lane shape)

WORLD = 'cave_circuit_practice_01'


def set_pose(x, y, yaw, z=0.12):
    req = (f'name: "bunker_mini" position {{x: {x:.3f} y: {y:.3f} z: {z}}} '
           f'orientation {{z: {math.sin(yaw / 2):.5f} w: {math.cos(yaw / 2):.5f}}}')
    r = subprocess.run(
        ['gz', 'service', '-s', f'/world/{WORLD}/set_pose', '--reqtype', 'gz.msgs.Pose',
         '--reptype', 'gz.msgs.Boolean', '--timeout', '3000', '--req', req],
        capture_output=True, text=True)
    return 'true' in r.stdout


class Grabber(Node):
    def __init__(self):
        super().__init__('lane_dataset_collector', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.bridge = CvBridge()
        self.last = None
        self.create_subscription(Image, '/oak/rgb/image_raw', self.cb, 2)

    def cb(self, msg):
        self.last = msg

    def fresh_image(self, after_sim_s, timeout=15.0):
        """First image whose timestamp is later than `after_sim_s` (simulation seconds)."""
        end = self.get_clock().now().nanoseconds / 1e9 + timeout
        while rclpy.ok() and self.get_clock().now().nanoseconds / 1e9 < end:
            rclpy.spin_once(self, timeout_sec=0.2)
            if self.last is not None:
                t = self.last.header.stamp.sec + self.last.header.stamp.nanosec / 1e9
                if t > after_sim_s:
                    return self.bridge.imgmsg_to_cv2(self.last, 'bgr8')
        return None

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9


def random_pose(rng):
    x = rng.uniform(lane.X_START + 0.5, lane.X_END - 2.5)
    h = 1e-3
    slope = (lane.centre(x + h) - lane.centre(x - h)) / (2 * h)
    tangent = math.atan(slope)
    offset = rng.gauss(0.0, 0.25)
    offset = max(-0.65, min(0.65, offset))
    y = lane.centre(x) + offset
    yaw = tangent + rng.gauss(0.0, 0.25)
    yaw = max(tangent - 0.7, min(tangent + 0.7, yaw))
    return x, y, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--count', type=int, default=800)
    ap.add_argument('--out', default=os.path.expanduser('~/datasets/lane_sim'))
    ap.add_argument('--val-fraction', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(args.out, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(args.out, split, 'labels'), exist_ok=True)

    rclpy.init()
    node = Grabber()
    while node.now_s() == 0.0:   # wait for the first /clock message
        rclpy.spin_once(node, timeout_sec=0.5)
    saved = skipped = failures = 0
    for i in range(args.count):
        x, y, yaw = random_pose(rng)
        if not any(set_pose(x, y, yaw) for _ in range(3)):   # a service call can time out
            failures += 1
            if failures >= 8:
                print('set_pose keeps failing - is the simulation running?')
                break
            continue
        failures = 0
        t_set = node.now_s()
        img = node.fresh_image(t_set + 0.8)   # let the robot settle, then take a new frame
        if img is None:
            failures += 1
            if failures >= 8:
                print('no camera image')
                break
            continue
        mask = white_mask(img)
        polys = mask_to_polygons(mask)
        if not polys:
            skipped += 1
            continue
        h, w = img.shape[:2]
        split = 'val' if rng.random() < args.val_fraction else 'train'
        name = f'lane_s{args.seed}_{i:05d}'
        cv2.imwrite(os.path.join(args.out, split, 'images', name + '.jpg'), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(args.out, split, 'labels', name + '.txt'), 'w') as f:
            for p in polys:
                coords = ' '.join(f'{px / w:.5f} {py / h:.5f}' for px, py in p)
                f.write(f'0 {coords}\n')
        saved += 1
        if saved % 25 == 0:
            print(f'{saved} saved ({skipped} without lane skipped)  [{i + 1}/{args.count}]')

    with open(os.path.join(args.out, 'data.yaml'), 'w') as f:
        f.write(f'path: {args.out}\ntrain: train/images\nval: val/images\nnames:\n  0: lane_line\n')
    set_pose(0.0, 0.0, 0.0)  # put the robot back at the start
    print(f'done: {saved} images in {args.out}')
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
