"""
artifact_detector: YOLO + depth -> 3D detections (simulation stand-in for the OAK-D Lite).

On the real robot the OAK-D Lite runs YOLO *on the camera* (RVC2 chip) and
computes the 3D position of every box from its stereo depth ("spatial
detection"). In Gazebo there is no such chip, so this node does the same job
on the PC:

    /oak/rgb/image_raw      (colour image)  --+
    /oak/stereo/image_raw   (depth, metres) --+--> YOLO --> 3D point per box
    /oak/rgb/camera_info    (intrinsics)    --+

Outputs:
    /artifacts/crop             std_msgs/String, JSON {name, score, w, h, jpeg (base64)}: a portrait
                                (upper part of the box, from the full image) of what was found,
                                at most twice a second per class; the dashboard keeps the best one
    /oak/nn/spatial_detections  vision_msgs/Detection3DArray (camera optical frame)
    /oak/nn/detections_image    sensor_msgs/Image with boxes drawn (for the demo screen)

Parameters:
    model          YOLO weights. 'yolov8n.pt' = pretrained on COCO (detects
                   person, backpack, ...). Replace with your trained 'best.pt'.
    confidence     minimum YOLO confidence (0..1)
    classes        comma-separated class names to keep, e.g. 'person,backpack';
                   empty = keep every class of the model
    roi_scale      fraction of the box (centre) used to read depth, like the
                   OAK "bounding box scale factor"
    device         '' (auto), 'cpu' or '0' (first NVIDIA GPU)

Needs the 'ultralytics' Python package (see docs/YOLO_TRAINING.md).
"""
import base64
import json
import math
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

MIN_DEPTH = 0.2   # OAK-D Lite MinZ (m)
MAX_DEPTH = 12.0  # end of the ideal depth range (m)


def box_to_point(depth, box, fx, fy, cx, cy, roi_scale=0.3):
    """
    3D point (x, y, z) in the camera optical frame for one bounding box.

    depth : HxW float array in metres (NaN/inf = no measurement)
    box   : (x1, y1, x2, y2) in pixels
    Uses the median depth of a small window in the centre of the box
    (robust against background pixels at the box edges), then the pinhole
    camera model:  X = (u - cx) * Z / fx,  Y = (v - cy) * Z / fy.
    Returns None if there is no valid depth.
    """
    x1, y1, x2, y2 = box
    u = 0.5 * (x1 + x2)
    v = 0.5 * (y1 + y2)
    hw = max(1.0, 0.5 * (x2 - x1) * roi_scale)
    hh = max(1.0, 0.5 * (y2 - y1) * roi_scale)
    h, w = depth.shape[:2]
    r0, r1 = int(max(0, v - hh)), int(min(h, v + hh + 1))
    c0, c1 = int(max(0, u - hw)), int(min(w, u + hw + 1))
    roi = depth[r0:r1, c0:c1]
    valid = roi[np.isfinite(roi) & (roi >= MIN_DEPTH) & (roi <= MAX_DEPTH)]
    if valid.size == 0:
        return None
    z = float(np.median(valid))
    return ((u - cx) * z / fx, (v - cy) * z / fy, z)


class ArtifactDetector(Node):

    def __init__(self):
        super().__init__('artifact_detector')
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('confidence', 0.5)
        self.declare_parameter('classes', '')
        self.declare_parameter('roi_scale', 0.3)
        self.declare_parameter('device', '')

        self.conf = self.get_parameter('confidence').value
        self.keep = {c.strip() for c in self.get_parameter('classes').value.split(',') if c.strip()}
        self.roi_scale = self.get_parameter('roi_scale').value

        try:
            from ultralytics import YOLO
        except ImportError:
            self.get_logger().fatal(
                "Python package 'ultralytics' is not installed. "
                'See src/mine_rover_core/docs/YOLO_TRAINING.md (step 1).')
            raise SystemExit(1)

        model_path = self.get_parameter('model').value
        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.device = self.get_parameter('device').value or None
        names = self.model.names
        self.get_logger().info(f'Model classes: {list(names.values())}')
        missing = [c for c in self.keep if c not in names.values()]
        if missing:
            self.get_logger().warn(
                f'These classes are not in the model and will never be detected: {missing}')

        self.bridge = CvBridge()
        self.K = None
        self.create_subscription(CameraInfo, '/oak/rgb/camera_info', self.on_info, 10)

        # colour and depth images arrive separately: pair them by timestamp
        sync = ApproximateTimeSynchronizer(
            [Subscriber(self, Image, '/oak/rgb/image_raw'),
             Subscriber(self, Image, '/oak/stereo/image_raw')],
            queue_size=5, slop=0.05)
        sync.registerCallback(self.on_images)

        self.pub_det = self.create_publisher(Detection3DArray, '/oak/nn/spatial_detections', 10)
        self.pub_img = self.create_publisher(Image, '/oak/nn/detections_image', 2)
        self.pub_crop = self.create_publisher(String, '/artifacts/crop', 5)
        self.last_crop = {}

    def on_info(self, msg):
        if self.K is None:
            self.K = msg.k  # [fx 0 cx; 0 fy cy; 0 0 1]
            self.get_logger().info(f'Camera intrinsics: fx={msg.k[0]:.1f} fy={msg.k[4]:.1f}')

    def on_images(self, rgb_msg, depth_msg):
        if self.K is None:
            return
        fx, cx, fy, cy = self.K[0], self.K[2], self.K[4], self.K[5]
        img = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg)  # 32FC1, metres

        result = self.model(img, conf=self.conf, device=self.device, verbose=False)[0]

        out = Detection3DArray()
        out.header = depth_msg.header  # frame: oak_rgb_camera_optical_frame
        annotated = img.copy()

        for xyxy, cls, score in zip(result.boxes.xyxy.cpu().numpy(),
                                    result.boxes.cls.cpu().numpy(),
                                    result.boxes.conf.cpu().numpy()):
            name = self.model.names[int(cls)]
            if self.keep and name not in self.keep:
                continue
            p = box_to_point(depth, xyxy, fx, fy, cx, cy, self.roi_scale)
            self.draw(annotated, xyxy, name, score, p)
            if p is None:
                continue
            self.publish_crop(img, xyxy, name, float(score))

            det = Detection3D()
            det.header = out.header
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = name
            hyp.hypothesis.score = float(score)
            hyp.pose.pose.position.x, hyp.pose.pose.position.y, hyp.pose.pose.position.z = p
            det.results.append(hyp)
            det.bbox.center.position = hyp.pose.pose.position
            # physical size of the box at that distance
            det.bbox.size.x = float((xyxy[2] - xyxy[0]) * p[2] / fx)
            det.bbox.size.y = float((xyxy[3] - xyxy[1]) * p[2] / fy)
            det.bbox.size.z = det.bbox.size.x
            out.detections.append(det)

        self.pub_det.publish(out)
        img_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        img_msg.header = rgb_msg.header
        self.pub_img.publish(img_msg)

    def publish_crop(self, img, xyxy, name, score):
        """Portrait = the upper 45 % of the box (the head), a little wider, cut from the full image."""
        import cv2
        now = time.monotonic()
        if now - self.last_crop.get(name, 0.0) < 0.5:
            return
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        if y2 - y1 < 30:
            return                                # too small to be a useful portrait
        self.last_crop[name] = now
        bw, bh = x2 - x1, y2 - y1
        h, w = img.shape[:2]
        cx = 0.5 * (x1 + x2)
        side = max(bw * 1.25, bh * 0.45)
        px1, px2 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
        py1, py2 = int(max(0, y1 - 0.05 * bh)), int(min(h, y1 - 0.05 * bh + side))
        crop = img[py1:py2, px1:px2]
        if crop.size == 0:
            return
        crop = cv2.resize(crop, (200, int(200 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_CUBIC)
        ok, jpg = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if ok:
            self.pub_crop.publish(String(data=json.dumps(dict(
                name=name, score=score, w=crop.shape[1], h=crop.shape[0],
                size=int(bh), jpeg=base64.b64encode(jpg.tobytes()).decode()))))

    @staticmethod
    def draw(img, xyxy, name, score, p):
        import cv2
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        color = (0, 200, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        dist = f' {math.sqrt(sum(c * c for c in p)):.1f}m' if p else ''
        label = f'{name} {score:.2f}{dist}'
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = ArtifactDetector()
    except SystemExit:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
