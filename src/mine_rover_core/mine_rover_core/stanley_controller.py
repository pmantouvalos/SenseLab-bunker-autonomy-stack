"""
stanley_controller: follows the lane centre with the Stanley path-tracking law.

Stanley (Stanford, DARPA Grand Challenge) steers the *front axle* with two terms:

    delta = psi_e  +  atan( k * e / (k_soft + v) )

    psi_e  heading error: angle between the robot heading and the path tangent
    e      cross-track error: lateral distance from the front axle to the path
    v      forward speed;  k_soft keeps the law stable at very low speed

The first term turns the robot parallel to the path, the second pulls it back onto
it (strongly when slow, gently when fast). The Bunker is a tracked, differential-drive
robot, so the steering angle is turned into a yaw rate with the bicycle-model relation

    omega = v * tan(delta) / L        (L = virtual wheelbase)

and the speed is reduced when the steering demand is large.

The lane detector already publishes the path in the robot frame (base_footprint), so
here the robot is at the origin looking along +x and no TF is needed.

    in  : /lane/center_path   nav_msgs/Path
    out : /auto/cmd_vel       geometry_msgs/Twist  (the autonomy_manager decides if it is used)
          /auto/stanley_debug std_msgs/Float32MultiArray [e, psi_e, delta, v, omega, tracking]
"""
import math

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def stanley(path_xy, v, front_x, k, k_soft):
    """
    path_xy : list of (x, y) in the robot frame, ordered from the robot outwards
    Returns (delta, e, psi_e) or None.
    """
    if len(path_xy) < 2:
        return None
    # nearest path point to the front axle
    i = min(range(len(path_xy)),
            key=lambda j: (path_xy[j][0] - front_x) ** 2 + path_xy[j][1] ** 2)
    j = min(i + 1, len(path_xy) - 1)
    if j == i:
        i = j - 1
    dx, dy = path_xy[j][0] - path_xy[i][0], path_xy[j][1] - path_xy[i][1]
    psi_e = math.atan2(dy, dx)          # robot heading is 0 in its own frame
    e = path_xy[i][1]                   # + = path is on the left of the front axle
    delta = psi_e + math.atan2(k * e, k_soft + abs(v))
    return delta, e, psi_e


class StanleyController(Node):

    def __init__(self):
        super().__init__('stanley_controller')
        # The same controller is started twice: one follows the lane (defaults), the other
        # follows the slalom path (path_topic:=/avoid/path cmd_topic:=/avoid/cmd_vel).
        self.declare_parameter('path_topic', '/lane/center_path')
        self.declare_parameter('cmd_topic', '/auto/cmd_vel')
        self.declare_parameter('debug_topic', '/auto/stanley_debug')
        for name, default in (('k', 1.2), ('k_soft', 0.3), ('wheelbase', 0.5),
                              ('front_axle_x', 0.3), ('max_steer', 0.6),
                              ('v_max', 0.5), ('v_min', 0.2), ('max_omega', 1.0),
                              ('path_timeout', 0.5)):
            self.declare_parameter(name, default)
        self.p = {n: self.get_parameter(n).value for n in (
            'k', 'k_soft', 'wheelbase', 'front_axle_x', 'max_steer', 'v_max', 'v_min',
            'max_omega', 'path_timeout')}
        self.path = []
        self.path_time = None
        self.v_cmd = self.p['v_max']
        path_topic = self.get_parameter('path_topic').value
        self.create_subscription(Path, path_topic, self.on_path, 5)
        self.pub = self.create_publisher(Twist, self.get_parameter('cmd_topic').value, 5)
        self.pub_dbg = self.create_publisher(
            Float32MultiArray, self.get_parameter('debug_topic').value, 5)
        self.create_timer(0.05, self.step)   # 20 Hz
        self.get_logger().info(f'Stanley controller ready: waiting for {path_topic}')

    def on_path(self, msg):
        self.path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.path_time = self.get_clock().now()

    def step(self):
        p = self.p
        cmd = Twist()
        dbg = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        fresh = (self.path_time is not None and
                 (self.get_clock().now() - self.path_time).nanoseconds / 1e9 < p['path_timeout'])
        res = stanley(self.path, self.v_cmd, p['front_axle_x'], p['k'], p['k_soft']) if fresh else None
        if res is not None:
            delta, e, psi_e = res
            delta = max(-p['max_steer'], min(p['max_steer'], delta))
            # slow down when the steering demand is large
            v = p['v_max'] * (1.0 - 0.6 * abs(delta) / p['max_steer'])
            v = max(p['v_min'], v)
            omega = v * math.tan(delta) / p['wheelbase']
            omega = max(-p['max_omega'], min(p['max_omega'], omega))
            self.v_cmd = v
            cmd.linear.x, cmd.angular.z = v, omega
            dbg = [e, psi_e, delta, v, omega, 1.0]
        self.pub.publish(cmd)
        self.pub_dbg.publish(Float32MultiArray(data=[float(x) for x in dbg]))


def main(args=None):
    rclpy.init(args=args)
    node = StanleyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
