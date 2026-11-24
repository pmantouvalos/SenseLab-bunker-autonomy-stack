"""
obstacle_detector: depth image -> big obstacles (boulders) measured in metres.

Camera only (the stereo depth of the OAK-D counts as camera). No neural network:
the decision "can the rover drive over it?" is geometric, so it works for any
object and needs no training data.

    /oak/stereo/image_raw   depth image, 32FC1, metres, optical frame
    /oak/rgb/camera_info    intrinsics (pinhole model)
    TF                      camera -> base_footprint (camera height / tilt)

How it works
  1. Every 4th pixel of the depth image becomes a 3D point with the pinhole model
        X = (u-cx)*Z/fx   Y = (v-cy)*Z/fy
     and is moved into the robot frame (x forward, y left, z up, z=0 = floor).
  2. Points higher than `min_height` above the floor are "something standing there"
     (gravel is ~2 cm, bricks ~4 cm -> ignored).
  3. Those points are dropped into a top-view grid (cells of `cell` m) and joined
     into blobs (connected components) = one blob per object.
  4. For every blob: height, lateral width (y extent), depth extent (x extent).
        height >= block_height and width >= block_width  -> BLOCKING     (boulder)
        otherwise                                         -> TRAVERSABLE  (small thing)

Output
    /obstacles   vision_msgs/Detection3DArray in the robot frame (base_footprint).
                 bbox.center = middle of the blob, bbox.size = (x extent, y extent, height),
                 results[0].hypothesis.class_id = 'blocking' | 'traversable',
                 score = number of points / 100 (a rough confidence).
    /obstacles/debug   sensor_msgs/Image: depth image with the blobs coloured.

Note: the camera only sees the FRONT face of an obstacle, so the x extent is
what is visible, not the real depth of the rock. The planner keeps the rock in memory.
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from mine_rover_core import lane_utils as lu


class ObstacleDetector(Node):

    def __init__(self):
        super().__init__('obstacle_detector')
        p = self.declare_parameter
        p('base_frame', 'base_footprint')
        p('stride', 4)             # use every Nth pixel (speed)
        p('min_depth', 0.3)        # m, ignore closer (OAK-D Lite MinZ ~0.2-0.35)
        p('max_depth', 5.0)        # m, ignore further (stereo noise grows with Z^2)
        p('min_height', 0.10)      # m above floor: lower = floor / gravel / bricks
        p('block_height', 0.25)    # m: at least this tall -> cannot be crossed
        p('block_width', 0.35)     # m: at least this wide -> cannot be crossed
        p('cell', 0.05)            # m, top-view grid cell
        p('min_points', 25)        # a blob needs this many points (noise filter)
        p('max_rate', 8.0)         # Hz, drop frames above this (CPU)
        p('x_max', 4.0)            # m, only report obstacles closer than this
        p('y_half', 1.6)           # m, only look this far to each side (lane is +-0.9)

        self.base = self.get_parameter('base_frame').value
        self.stride = int(self.get_parameter('stride').value)
        self.min_depth = self.get_parameter('min_depth').value
        self.max_depth = self.get_parameter('max_depth').value
        self.min_height = self.get_parameter('min_height').value
        self.block_h = self.get_parameter('block_height').value
        self.block_w = self.get_parameter('block_width').value
        self.cell = self.get_parameter('cell').value
        self.min_points = int(self.get_parameter('min_points').value)
        self.min_period = 1.0 / max(self.get_parameter('max_rate').value, 0.1)
        self.x_max = self.get_parameter('x_max').value
        self.y_half = self.get_parameter('y_half').value

        self.bridge = CvBridge()
        self.K = None
        self.rng = np.random.default_rng(1)
        self.last = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.on_info, 10)
        self.create_subscription(Image, '/oak/stereo/image_raw', self.on_depth, 1)
        self.pub = self.create_publisher(Detection3DArray, '/obstacles', 10)
        self.pub_dbg = self.create_publisher(Image, '/obstacles/debug', 1)
        self.get_logger().info('obstacle_detector ready (depth image -> /obstacles)')

    def on_info(self, msg):
        if self.K is None:
            self.K = list(msg.k)

    def camera_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.base, 'oak_rgb_camera_optical_frame', Time())
        except TransformException as e:
            self.get_logger().warn(f'No TF to the camera yet: {e}', throttle_duration_sec=5.0)
            return None
        q, t = tf.transform.rotation, tf.transform.translation
        return lu.quat_to_matrix(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z])

    # ------------------------------------------------------------------ geometry
    def points_in_base(self, depth, R, t):
        """Nx3 points (x fwd, y left, z up) + their pixel coordinates (u, v)."""
        s = self.stride
        fx, cx, fy, cy = self.K[0], self.K[2], self.K[4], self.K[5]
        z = depth[::s, ::s]
        v, u = np.mgrid[0:depth.shape[0]:s, 0:depth.shape[1]:s]
        ok = np.isfinite(z) & (z >= self.min_depth) & (z <= self.max_depth)
        z, u, v = z[ok], u[ok], v[ok]
        cam = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1)
        return cam @ R.T + t, u, v

    def floor_plane(self, pts):
        """
        Floor plane z = a*x + b*y + c in the robot frame, fitted with RANSAC.

        The TF says "the floor is z=0", but when the robot climbs a brick or tilts on the
        gravel, the whole floor tilts in the camera view (5 deg at 2 m = 17 cm!) and would
        look like a wall. The plane through most of the near points is the floor, even when a
        rock fills part of the image (a vertical wall has fewer points than the floor).
        Returns (a, b, c); (0, 0, 0) if there are too few points.
        """
        near = pts[(pts[:, 0] > 0.5) & (pts[:, 0] < 2.5) & (np.abs(pts[:, 1]) < 1.2)
                   & (pts[:, 2] < 0.5) & (pts[:, 2] > -0.5)]
        if len(near) < 150:
            return 0.0, 0.0, 0.0
        if len(near) > 600:
            near = near[self.rng.choice(len(near), 600, replace=False)]
        best, best_n = (0.0, 0.0, 0.0), 0
        for _ in range(60):
            tri = near[self.rng.choice(len(near), 3, replace=False)]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if abs(n[2]) < 1e-6:
                continue
            a, b = -n[0] / n[2], -n[1] / n[2]
            if math.hypot(a, b) > 0.36:        # > ~20 deg: not a floor
                continue
            c = tri[0, 2] - a * tri[0, 0] - b * tri[0, 1]
            cnt = int((np.abs(near[:, 2] - (a * near[:, 0] + b * near[:, 1] + c)) < 0.03).sum())
            if cnt > best_n:
                best, best_n = (a, b, c), cnt
        if best_n < 0.3 * len(near):
            return 0.0, 0.0, 0.0
        a, b, c = best                       # refine with all inliers
        m = np.abs(near[:, 2] - (a * near[:, 0] + b * near[:, 1] + c)) < 0.03
        A = np.c_[near[m, 0], near[m, 1], np.ones(int(m.sum()))]
        sol = np.linalg.lstsq(A, near[m, 2], rcond=None)[0]
        return float(sol[0]), float(sol[1]), float(sol[2])

    def find_blobs(self, pts):
        """Joins tall points into blobs in the top view. Returns a list of dicts."""
        a, b, c = self.floor_plane(pts)
        pts = pts.copy()
        pts[:, 2] -= a * pts[:, 0] + b * pts[:, 1] + c      # height above the floor
        tall = pts[:, 2] >= self.min_height
        tall &= (pts[:, 2] <= 2.0) & (pts[:, 0] <= self.x_max) & (np.abs(pts[:, 1]) <= self.y_half)
        idx = np.nonzero(tall)[0]
        if idx.size == 0:
            return []
        c = self.cell
        gx = int(self.x_max / c) + 2
        gy = int(2 * self.y_half / c) + 2
        ix = np.clip((pts[idx, 0] / c).astype(int), 0, gx - 1)
        iy = np.clip(((pts[idx, 1] + self.y_half) / c).astype(int), 0, gy - 1)
        grid = np.zeros((gx, gy), np.uint8)
        grid[ix, iy] = 255
        # close small holes (the depth is sampled every few pixels)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, labels = cv2.connectedComponents(grid, connectivity=8)
        lab = labels[ix, iy]
        blobs = []
        for k in range(1, n):
            sel = idx[lab == k]
            if sel.size < self.min_points:
                continue
            P = pts[sel]
            # percentiles instead of min/max: a few noisy pixels must not widen the blob
            x0, x1 = np.percentile(P[:, 0], [2, 98])
            y0, y1 = np.percentile(P[:, 1], [2, 98])
            h = float(np.percentile(P[:, 2], 98))
            width = max(y1 - y0, x1 - x0)
            blocking = h >= self.block_h and width >= self.block_w
            blobs.append(dict(x0=x0, x1=x1, y0=y0, y1=y1, h=h, n=int(sel.size),
                              blocking=bool(blocking), sel=sel))
        return sorted(blobs, key=lambda b: b['x0'])

    # ------------------------------------------------------------------ callback
    def on_depth(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.K is None or now - self.last < self.min_period:
            return
        pose = self.camera_pose()
        if pose is None:
            return
        self.last = now
        R, t = pose
        depth = self.bridge.imgmsg_to_cv2(msg)
        pts, u, v = self.points_in_base(depth, R, t)
        blobs = self.find_blobs(pts) if pts.size else []

        out = Detection3DArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.base
        for b in blobs:
            d = Detection3D()
            d.header = out.header
            d.bbox.center.position.x = float(0.5 * (b['x0'] + b['x1']))
            d.bbox.center.position.y = float(0.5 * (b['y0'] + b['y1']))
            d.bbox.center.position.z = float(0.5 * b['h'])
            d.bbox.size.x = float(b['x1'] - b['x0'])
            d.bbox.size.y = float(b['y1'] - b['y0'])
            d.bbox.size.z = float(b['h'])
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'blocking' if b['blocking'] else 'traversable'
            hyp.hypothesis.score = float(min(1.0, b['n'] / 100.0))
            d.results.append(hyp)
            out.detections.append(d)
        self.pub.publish(out)

        if self.pub_dbg.get_subscription_count():
            self.publish_debug(depth, u, v, blobs, msg.header)

    def publish_debug(self, depth, u, v, blobs, header):
        d = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        vis = cv2.applyColorMap(np.clip(d / self.max_depth * 255, 0, 255).astype(np.uint8),
                                cv2.COLORMAP_JET)
        for b in blobs:
            color = (0, 0, 255) if b['blocking'] else (0, 255, 0)
            vis[v[b['sel']], u[b['sel']]] = color
            cv2.putText(vis, f"{b['x0']:.1f}m w{max(b['y1']-b['y0'], b['x1']-b['x0']):.2f} h{b['h']:.2f}",
                        (int(u[b['sel']].min()), int(max(12, v[b['sel']].min() - 4))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        img = self.bridge.cv2_to_imgmsg(vis, 'bgr8')
        img.header = header
        self.pub_dbg.publish(img)


def main():
    rclpy.init()
    node = ObstacleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
