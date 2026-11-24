"""
lane_detector: neural network that finds the white lane lines and the lane centre.

    /oak/rgb/image_raw  --> YOLOv8-seg (lane_seg.pt) --> mask of the white lines
                        --> mask pixels projected on the floor with the camera pose (TF)
                        --> left line / right line / lane centre in metres

The network is trained in the simulator (training/collect_lane_dataset.py and
training/train_lane_seg.py). If the weights are missing the node falls back to plain
colour thresholds, so the rest of the chain can still be tested.

Outputs
    /lane/center_path   nav_msgs/Path   centre of the lane in base_footprint (x fwd, y left),
                                        empty when no lane is seen. Input of the Stanley controller.
    /lane/status        std_msgs/String JSON: {"valid", "left", "right", "points", "source"}
    /lane/debug_image   sensor_msgs/Image  camera image with mask and lane centre (dashboard)

Parameters
    model        weights file; a bare name is looked up in share/mine_rover_core/models
    confidence   minimum network confidence
    lane_width   distance between the two line centres (m): used when only one line is visible
    x_min, x_max floor range (m ahead of base_footprint) that is trusted
    device       '' (auto), 'cpu' or '0' (GPU)
"""
import json
import os

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from mine_rover_core import lane_utils as lu


class LaneDetector(Node):

    def __init__(self):
        super().__init__('lane_detector')
        self.declare_parameter('model', 'lane_seg.pt')
        self.declare_parameter('confidence', 0.3)
        self.declare_parameter('lane_width', 1.8)
        self.declare_parameter('x_min', 0.5)
        self.declare_parameter('x_max', 6.0)
        self.declare_parameter('device', '')
        self.declare_parameter('base_frame', 'base_footprint')
        self.conf = self.get_parameter('confidence').value
        self.lane_width = self.get_parameter('lane_width').value
        self.x_min = self.get_parameter('x_min').value
        self.x_max = self.get_parameter('x_max').value
        self.device = self.get_parameter('device').value or None
        self.base = self.get_parameter('base_frame').value

        self.model = self.load_model(self.get_parameter('model').value)
        self.source = 'neural_network' if self.model else 'colour_threshold'

        self.bridge = CvBridge()
        self.K = None
        self.prev_centre = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.on_info, 10)
        self.create_subscription(Image, '/oak/rgb/image_raw', self.on_image, 2)
        self.pub_path = self.create_publisher(Path, '/lane/center_path', 5)
        self.pub_status = self.create_publisher(String, '/lane/status', 5)
        self.pub_debug = self.create_publisher(Image, '/lane/debug_image', 2)
        self.get_logger().info(f'Lane detector ready (source: {self.source})')

    # ------------------------------------------------------------------ model
    def load_model(self, name):
        path = name
        if not os.path.isabs(path) or not os.path.exists(path):
            try:
                path = os.path.join(get_package_share_directory('mine_rover_core'), 'models', name)
            except Exception:
                path = name
        if not os.path.exists(path):
            self.get_logger().warn(
                f'Lane weights "{name}" not found: using colour thresholds instead. '
                'Train them with training/train_lane_seg.py (see docs/AUTONOMY.md).')
            return None
        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().error("'ultralytics' is not installed: using colour thresholds")
            return None
        self.get_logger().info(f'Loading lane network: {path}')
        return YOLO(path)

    def on_info(self, msg):
        if self.K is None:
            self.K = list(msg.k)

    # ------------------------------------------------------------------ main loop
    def segment(self, img):
        """White-line mask (uint8 0/255) from the network, or from thresholds."""
        if self.model is None:
            return lu.white_mask(img)
        res = self.model(img, conf=self.conf, device=self.device, verbose=False)[0]
        mask = np.zeros(img.shape[:2], np.uint8)
        if res.masks is not None:
            for poly in res.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
        mask[:lu.HORIZON_ROW, :] = 0
        return mask

    def camera_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.base, 'oak_rgb_camera_optical_frame', Time())
        except TransformException as e:
            self.get_logger().warn(f'No TF to the camera yet: {e}', throttle_duration_sec=5.0)
            return None
        q, p = tf.transform.rotation, tf.transform.translation
        return lu.quat_to_matrix(q.x, q.y, q.z, q.w), np.array([p.x, p.y, p.z])

    def on_image(self, msg):
        if self.K is None:
            return
        pose = self.camera_pose()
        if pose is None:
            return
        R, t = pose
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        mask = self.segment(img)

        rows = range(lu.HORIZON_ROW + 2, img.shape[0], 4)
        runs = lu.scanline_runs(mask, rows)
        geo = lu.lane_centre_from_runs(runs, self.K, R, t, self.lane_width,
                                       self.x_min, self.x_max, self.prev_centre)
        fit = lu.fit_centre(geo['centre'])

        path = Path()
        path.header.stamp = msg.header.stamp
        path.header.frame_id = self.base
        if fit is not None:
            coef, x0, x1 = fit
            self.prev_centre = coef if len(coef) == 3 else np.array([0.0, coef[0], coef[1]])
            xs = np.arange(0.0, x1 + 1e-6, 0.25)
            for x in xs:
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(x)
                ps.pose.position.y = float(np.polyval(coef, x))
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
        else:
            self.prev_centre = None
        self.pub_path.publish(path)

        status = dict(valid=fit is not None, left=bool(len(geo['left']) >= 4),
                      right=bool(len(geo['right']) >= 4), points=int(len(geo['centre'])),
                      source=self.source)
        self.pub_status.publish(String(data=json.dumps(status)))
        self.publish_debug(img, mask, fit, status, msg.header, R, t)

    def publish_debug(self, img, mask, fit, status, header, R, t):
        out = img.copy()
        out[mask > 0] = (0.5 * out[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        if fit is not None:
            coef, _, x1 = fit
            xs = np.arange(0.5, x1, 0.2)
            uv, ok = lu.ground_to_pixels(np.stack([xs, np.polyval(coef, xs)], axis=1), self.K, R, t)
            pts = uv[ok].astype(np.int32)
            if len(pts) >= 2:
                cv2.polylines(out, [pts.reshape(-1, 1, 2)], False, (0, 165, 255), 3)
        label = ('LANE ' + ('L' if status['left'] else '-') + ('R' if status['right'] else '-')
                 if status['valid'] else 'NO LANE')
        cv2.putText(out, f'{label}  [{self.source}]', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if status['valid'] else (0, 0, 255), 2)
        m = self.bridge.cv2_to_imgmsg(out, 'bgr8')
        m.header = header
        self.pub_debug.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
