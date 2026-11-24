"""
oak_points_relay: fixes the frame of the simulated OAK-D Lite point cloud.

Gazebo's rgbd_camera publishes the colour/depth images in the camera *optical*
frame (z forward, x right, y down) but the point cloud with x forward, y left,
z up. Both get the same frame name, so RViz drew the cloud turned by 90 deg.

This node only changes the label:
    /oak/points_raw  (frame oak_rgb_camera_optical_frame, wrong)
 -> /oak/points      (frame oak_rgb_camera_frame, x forward: correct)

Simulation only. The real depthai-ros driver publishes a correct cloud.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class OakPointsRelay(Node):

    def __init__(self):
        super().__init__('oak_points_relay')
        self.declare_parameter('frame_id', 'oak_rgb_camera_frame')
        self.frame_id = self.get_parameter('frame_id').value
        # reliable (default QoS) so RViz, which subscribes as reliable, receives it
        self.pub = self.create_publisher(PointCloud2, '/oak/points', 5)
        self.create_subscription(PointCloud2, '/oak/points_raw', self.relay, qos_profile_sensor_data)

    def relay(self, msg):
        msg.header.frame_id = self.frame_id
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = OakPointsRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
