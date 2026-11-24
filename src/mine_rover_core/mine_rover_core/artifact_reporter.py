"""
artifact_reporter: turns camera detections into a list of artifacts on the map.

Works the same in simulation and on the real robot: it only needs
/oak/nn/spatial_detections (vision_msgs/Detection3DArray, in the camera frame),
which comes from artifact_detector in Gazebo or from the OAK-D Lite itself.

For every detection it
  1. transforms the 3D point from the camera frame to the map frame (TF),
  2. merges it with an existing artifact of the same class closer than
     'merge_radius' (averaging the position), or starts a new candidate,
  3. reports the artifact once it has been seen 'min_observations' times
     (a single wrong YOLO box is not enough), like a DARPA SubT report.

Outputs:
  /artifacts/markers   visualization_msgs/MarkerArray  (spheres + labels in RViz)
  /artifacts/report    std_msgs/String  one line per newly confirmed artifact
  log line             "ARTIFACT FOUND #1: backpack at (x, y, z)"

Parameters:
  pose_topic        '' = use TF fixed_frame <- camera (wheel odometry). Or a nav_msgs/Odometry topic
                    with the robot pose (the simulator's /ground_truth/odom, or the wheel+IMU fused
                    odometry of the real robot): the point is then placed with that pose. Reason: on
                    tracks the wheel yaw slips, which moves far-away finds by tens of centimetres.
  fixed_frame       frame the artifacts are reported in ('odom' for now;
                    'map' once SLAM runs)
  merge_radius      m, detections of the same class closer than this are one artifact
  min_observations  detections needed before an artifact is reported
  min_score         ignore detections below this confidence
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from std_msgs.msg import String
from tf2_geometry_msgs import do_transform_point
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray

# Marker colour per class (r, g, b); anything else is white
COLORS = {
    'person': (1.0, 0.2, 0.2),
    'survivor': (1.0, 0.2, 0.2),
    'backpack': (1.0, 0.6, 0.0),
    'fire extinguisher': (0.9, 0.0, 0.0),
    'helmet': (1.0, 1.0, 0.0),
}


def quat_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Artifact:
    def __init__(self, name, x, y, z, score):
        self.name, self.x, self.y, self.z = name, x, y, z
        self.count, self.best_score = 1, score
        self.reported = False
        self.id = None

    def add(self, x, y, z, score):
        n = self.count
        self.x = (self.x * n + x) / (n + 1)
        self.y = (self.y * n + y) / (n + 1)
        self.z = (self.z * n + z) / (n + 1)
        self.count += 1
        self.best_score = max(self.best_score, score)

    def dist2(self, x, y, z):
        return (self.x - x) ** 2 + (self.y - y) ** 2 + (self.z - z) ** 2


class ArtifactReporter(Node):

    def __init__(self):
        super().__init__('artifact_reporter')
        self.declare_parameter('fixed_frame', 'odom')
        self.declare_parameter('pose_topic', '')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('merge_radius', 1.0)
        self.declare_parameter('min_observations', 5)
        self.declare_parameter('min_score', 0.5)
        self.frame = self.get_parameter('fixed_frame').value
        self.radius2 = self.get_parameter('merge_radius').value ** 2
        self.min_obs = self.get_parameter('min_observations').value
        self.min_score = self.get_parameter('min_score').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.artifacts = []
        self.next_id = 1
        self.pose = None
        self.base = self.get_parameter('base_frame').value
        pose_topic = self.get_parameter('pose_topic').value
        if pose_topic:
            self.create_subscription(Odometry, pose_topic, self.on_pose, 10)
            self.frame = 'world'

        self.create_subscription(Detection3DArray, '/oak/nn/spatial_detections', self.on_detections, 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/artifacts/markers', 10)
        self.pub_report = self.create_publisher(String, '/artifacts/report', 10)
        self.create_timer(1.0, self.publish_markers)
        self.get_logger().info(f'Reporting artifacts in frame "{self.frame}"')

    def on_pose(self, msg):
        self.pose = msg.pose.pose

    def point_in_world(self, msg, p):
        """Camera point -> world with the robot pose from pose_topic (+ static camera mount TF)."""
        if self.pose is None:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(self.base, msg.header.frame_id, Time(),
                                                 timeout=Duration(seconds=0.2))
        except TransformException as e:
            self.get_logger().warn(f'No transform {msg.header.frame_id} -> {self.base}: {e}',
                                   throttle_duration_sec=5.0)
            return None
        q = tf.transform.rotation
        Rc = quat_matrix(q.x, q.y, q.z, q.w)
        t = tf.transform.translation
        pb = Rc @ np.array([p.x, p.y, p.z]) + np.array([t.x, t.y, t.z])      # in base_footprint
        o = self.pose.orientation
        Rb = quat_matrix(o.x, o.y, o.z, o.w)
        w = Rb @ pb + np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z])
        return float(w[0]), float(w[1]), float(w[2])

    def on_detections(self, msg):
        for det in msg.detections:
            if not det.results:
                continue
            hyp = det.results[0]
            if hyp.hypothesis.score < self.min_score:
                continue
            p = PointStamped()
            p.header = msg.header
            p.point = hyp.pose.pose.position
            if self.get_parameter('pose_topic').value:
                w = self.point_in_world(msg, p.point)
                if w is not None:
                    self.add_observation(hyp.hypothesis.class_id, *w, hyp.hypothesis.score)
                continue
            try:
                tf = self.tf_buffer.lookup_transform(
                    self.frame, msg.header.frame_id, msg.header.stamp,
                    timeout=Duration(seconds=0.2))
            except TransformException as e:
                self.get_logger().warn(f'No transform {msg.header.frame_id} -> {self.frame}: {e}',
                                       throttle_duration_sec=5.0)
                return
            q = do_transform_point(p, tf).point
            self.add_observation(hyp.hypothesis.class_id, q.x, q.y, q.z, hyp.hypothesis.score)

    def add_observation(self, name, x, y, z, score):
        for a in self.artifacts:
            if a.name == name and a.dist2(x, y, z) < self.radius2:
                a.add(x, y, z, score)
                break
        else:
            a = Artifact(name, x, y, z, score)
            self.artifacts.append(a)

        if not a.reported and a.count >= self.min_obs:
            a.reported = True
            a.id = self.next_id
            self.next_id += 1
            text = (f'ARTIFACT FOUND #{a.id}: {a.name} at '
                    f'({a.x:.2f}, {a.y:.2f}, {a.z:.2f}) m in "{self.frame}"')
            self.get_logger().info(text)
            self.pub_report.publish(String(data=text))

    def publish_markers(self):
        arr = MarkerArray()
        now = self.get_clock().now().to_msg()
        for i, a in enumerate(a for a in self.artifacts if a.reported):
            r, g, b = COLORS.get(a.name, (1.0, 1.0, 1.0))
            sphere = Marker()
            sphere.header.frame_id = self.frame
            sphere.header.stamp = now
            sphere.ns, sphere.id = 'artifacts', a.id
            sphere.type, sphere.action = Marker.SPHERE, Marker.ADD
            sphere.pose.position.x, sphere.pose.position.y, sphere.pose.position.z = a.x, a.y, a.z
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.35
            sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = r, g, b, 0.9
            arr.markers.append(sphere)

            label = Marker()
            label.header = sphere.header
            label.ns, label.id = 'artifact_labels', a.id
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = a.x, a.y
            label.pose.position.z = a.z + 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.25
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f'#{a.id} {a.name}\n({a.x:.1f}, {a.y:.1f})'
            arr.markers.append(label)
        self.pub_markers.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = ArtifactReporter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
