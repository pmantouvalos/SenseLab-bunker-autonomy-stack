"""
slalom_planner: remembers the big obstacles and draws a smooth path around them.

The track is a straight corridor, so everything is planned in "track coordinates":
    s = x (along the track)   d = y (sideways, 0 = lane centre, + = left)
which are simply the odom axes, because the rover starts at the track entrance
looking along +x (odom is the "fake SLAM": it tells us how far we have gone).

    in  : /obstacles            vision_msgs/Detection3DArray  (obstacle_detector, robot frame)
          TF odom -> base_footprint
    out : /avoid/path           nav_msgs/Path  in base_footprint (a Stanley controller follows it)
          /avoid/status         std_msgs/String  JSON for the autonomy_manager
          /avoid/debug          visualization_msgs/MarkerArray (obstacles + path, for RViz)

Idea, step by step
  1. MEMORY. A blocking blob is stored (in odom) only when it was measured from
     gate_min..gate_max metres: closer than ~0.6 m the camera cannot see it any more
     and further than ~2.6 m the stereo depth is too noisy. The camera only sees the
     front face of a rock, so its depth is assumed to be `obstacle_depth`.
  2. FREE CORRIDOR. For each obstacle the rover centre may be on the left or on the right of
     it. The allowed interval on one side is between the obstacle edge pushed away by
     (half robot width + margin) and the lane line pulled in by half a robot width.
     The apex of the pass is the middle of that interval. No interval wide enough -> BLOCKED.
  3. CURVE. d(s) is a flat line at each apex while the robot body is beside the rock
     ([x_front - body_half, x_back + body_half]) and a smooth S-curve (smoothstep: zero slope
     at both ends) between two apexes. After the last rock it returns to the centre line.
  4. The curve is re-computed 10 times per second from the memory, so it follows every new
     measurement. The side chosen for a rock never flips once it is chosen (hysteresis).
"""
import json
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener
from vision_msgs.msg import Detection3DArray
from visualization_msgs.msg import Marker, MarkerArray


def smoothstep(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


class SlalomPlanner(Node):

    def __init__(self):
        super().__init__('slalom_planner')
        p = self.declare_parameter
        p('base_frame', 'base_footprint')
        p('odom_frame', 'odom')
        # '' = use the TF odom->base (wheel odometry). On tracks the wheels slip when turning,
        # so the wheel yaw drifts away from the real heading (seen in the simulator: 11 deg
        # after one S-curve) and every rock would be stored 40 cm sideways. In the simulator we
        # therefore use the ground-truth pose; on the real robot use wheel odometry + IMU
        # (robot_localization) published on a topic and put its name here.
        p('pose_topic', '')
        p('half_width', 0.285)       # robot half width (m)
        p('margin', 0.05)            # extra clearance to a rock (m)
        p('lane_half', 0.90)         # lane line centre, the robot centre may go to lane_half-half_width
        p('lane_extra', 0.10)        # allowed to go this much further if the gap is otherwise too small
        p('body_half', 0.40)         # robot half length + a little (m)
        p('obstacle_depth', 1.1)     # assumed depth (x) of a rock, the camera only sees its front
        p('trigger_dist', 2.2)       # m, gap between the robot nose and the rock to start the manoeuvre
        p('nose', 0.345)             # robot centre -> front (m)
        p('gate_min', 0.6)           # only use measurements from this range ...
        p('gate_max', 2.6)           # ... to this range
        p('min_obs', 3)              # frames needed before an obstacle is trusted
        p('ramp_default', 1.2)       # m, length of the first sideways move
        p('ramp_out', 1.2)           # m, return to the centre after the last rock
        p('horizon', 3.5)            # m of path published ahead
        p('merge_dist', 0.6)         # blobs closer than this belong to the same rock
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.base, self.odom = g('base_frame'), g('odom_frame')
        self.hw, self.margin = g('half_width'), g('margin')
        self.lane_half, self.lane_extra = g('lane_half'), g('lane_extra')
        self.body_half, self.depth = g('body_half'), g('obstacle_depth')
        self.trigger, self.nose = g('trigger_dist'), g('nose')
        self.gate_min, self.gate_max = g('gate_min'), g('gate_max')
        self.min_obs = int(g('min_obs'))
        self.ramp_default, self.ramp_out = g('ramp_default'), g('ramp_out')
        self.horizon, self.merge = g('horizon'), g('merge_dist')

        self.obs = []                # remembered rocks (odom frame)
        self.last_pose = None
        pose_topic = g('pose_topic')
        if pose_topic:
            self.create_subscription(Odometry, pose_topic, self.on_pose, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Detection3DArray, '/obstacles', self.on_obstacles, 5)
        self.pub_path = self.create_publisher(Path, '/avoid/path', 5)
        self.pub_status = self.create_publisher(String, '/avoid/status', 5)
        self.pub_dbg = self.create_publisher(MarkerArray, '/avoid/debug', 2)
        self.create_timer(0.1, self.plan)
        self.get_logger().info('slalom_planner ready')

    # ------------------------------------------------------------------ pose
    def on_pose(self, msg):
        q, p = msg.pose.pose.orientation, msg.pose.pose.position
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.last_pose = (p.x, p.y, yaw)

    def pose(self):
        if self.last_pose is not None:
            return self.last_pose
        if self.get_parameter('pose_topic').value:
            return None
        try:
            tf = self.tf_buffer.lookup_transform(self.odom, self.base, Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        return t.x, t.y, yaw

    # ------------------------------------------------------------------ memory
    def on_obstacles(self, msg):
        pose = self.pose()
        if pose is None:
            return
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        for d in msg.detections:
            if not d.results or d.results[0].hypothesis.class_id != 'blocking':
                continue
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            sx, sy = d.bbox.size.x, d.bbox.size.y
            near = cx - sx / 2
            if not (self.gate_min <= near <= self.gate_max):
                continue
            xs, ys = [], []
            for ax in (cx - sx / 2, cx + sx / 2):
                for ay in (cy - sy / 2, cy + sy / 2):
                    xs.append(x + c * ax - s * ay)
                    ys.append(y + s * ax + c * ay)
            box = dict(x0=min(xs), x1=max(xs), y0=min(ys), y1=max(ys))
            if box['y0'] > 1.3 or box['y1'] < -1.3:
                continue                      # not in the lane
            self.merge_box(box, near)

    def merge_box(self, b, near):
        """
        Update the memory with one measurement taken from `near` metres.
        Stereo depth is much more accurate when close, so a closer measurement replaces
        the stored one, a similar one is averaged and a clearly farther one is ignored.
        """
        for o in self.obs:
            if (abs(o['x0'] - b['x0']) < self.merge and b['y0'] < o['y1'] + self.merge
                    and b['y1'] > o['y0'] - self.merge):
                o['n'] += 1
                if near < o['rng'] - 0.05:
                    o.update(b, rng=near)
                elif near <= o['rng'] + 0.4:
                    for k in ('x0', 'x1', 'y0', 'y1'):
                        o[k] = 0.5 * o[k] + 0.5 * b[k]
                return
        self.obs.append(dict(b, n=1, side=None, ramp0=None, rng=near))

    # ------------------------------------------------------------------ planning
    def corridor(self, o, side, extra=0.0):
        """Allowed interval for the robot centre on one side of the rock, or None."""
        lim = self.lane_half + extra - self.hw
        if side > 0:
            lo, hi = o['y1'] + self.hw + self.margin, lim
        else:
            lo, hi = -lim, o['y0'] - self.hw - self.margin
        return (lo, hi) if hi - lo >= 0.0 else None

    def choose(self, o):
        """Side (+1 left / -1 right) and interval (lo, hi) for one rock, or None."""
        for extra in (0.0, self.lane_extra):
            opts = {sd: self.corridor(o, sd, extra) for sd in (+1, -1)}
            opts = {sd: iv for sd, iv in opts.items() if iv is not None}
            if o['side'] in opts:
                return o['side'], opts[o['side']]
            if opts:
                sd = max(opts, key=lambda k: opts[k][1] - opts[k][0])
                return sd, opts[sd]
        return None

    def build_holds(self, xr):
        """List of holds (x_start, x_end, apex, ramp_start) or None if blocked."""
        trusted = sorted((o for o in self.obs if o['n'] >= self.min_obs), key=lambda o: o['x0'])
        holds = []
        for o in trusted:
            sel = self.choose(o)
            if sel is None:
                return None
            side, (lo, hi) = sel
            o['side'] = side
            x_start = o['x0'] - self.body_half
            x_end = max(o['x1'], o['x0'] + self.depth) + self.body_half
            if holds and x_start <= holds[-1]['end']:         # overlapping holds
                h = holds[-1]
                if h['side'] != side:
                    return None
                h['lo'], h['hi'] = max(h['lo'], lo), min(h['hi'], hi)
                if h['hi'] < h['lo']:
                    return None
                h['end'] = max(h['end'], x_end)
                continue
            holds.append(dict(start=x_start, end=x_end, side=side, lo=lo, hi=hi, o=o))
        for i, h in enumerate(holds):
            h['apex'] = 0.5 * (h['lo'] + h['hi'])
            if i > 0:
                h['ramp0'] = holds[i - 1]['end']
            else:
                o = h['o']
                if o['ramp0'] is None:
                    o['ramp0'] = max(h['start'] - self.ramp_default, xr)   # frozen once planned
                h['ramp0'] = min(o['ramp0'], h['start'] - 0.2)
        return holds

    def lateral(self, holds, x):
        """d(x): the sideways target of the rover centre."""
        d_prev, x_prev_end = 0.0, -1e9
        for h in holds:
            if x < h['ramp0']:
                return d_prev
            if x < h['start']:
                return d_prev + (h['apex'] - d_prev) * smoothstep(
                    (x - h['ramp0']) / max(0.05, h['start'] - h['ramp0']))
            if x <= h['end']:
                return h['apex']
            d_prev, x_prev_end = h['apex'], h['end']
        if holds:
            return d_prev * (1.0 - smoothstep((x - x_prev_end) / self.ramp_out))
        return 0.0

    def plan(self):
        pose = self.pose()
        if pose is None:
            return
        x, y, yaw = pose
        holds = self.build_holds(x)
        blocked = holds is None
        status = dict(blocked=blocked, n_known=len([o for o in self.obs if o['n'] >= self.min_obs]),
                      n_seen=len(self.obs), unpassed=False, ready=False, dist=None, clear=True)
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.base
        if not blocked:
            unpassed = [h for h in holds if h['end'] > x]
            status['unpassed'] = bool(unpassed)
            status['ready'] = bool(unpassed)
            if unpassed:
                status['dist'] = round(unpassed[0]['o']['x0'] - (x + self.nose), 2)
            d_now = self.lateral(holds, x)
            status['clear'] = (not unpassed) and abs(y) < 0.2
            status['d_target'] = round(d_now, 2)
            c, s = math.cos(yaw), math.sin(yaw)
            for xs in np.arange(x - 0.3, x + self.horizon, 0.05):
                ds = self.lateral(holds, xs)
                dx, dy = xs - x, ds - y
                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = c * dx + s * dy
                ps.pose.position.y = -s * dx + c * dy
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
        self.pub_status.publish(String(data=json.dumps(status)))
        if not blocked:
            self.pub_path.publish(path)
        self.publish_debug(holds)

    def publish_debug(self, holds):
        arr = MarkerArray()
        for i, o in enumerate(self.obs):
            m = Marker()
            m.header.frame_id, m.ns, m.id = self.odom, 'rocks', i
            m.type, m.action = Marker.CUBE, Marker.ADD
            m.pose.position.x = 0.5 * (o['x0'] + max(o['x1'], o['x0'] + self.depth))
            m.pose.position.y = 0.5 * (o['y0'] + o['y1'])
            m.pose.position.z = 0.3
            m.pose.orientation.w = 1.0
            m.scale.x = max(o['x1'], o['x0'] + self.depth) - o['x0']
            m.scale.y = o['y1'] - o['y0']
            m.scale.z = 0.6
            m.color.r, m.color.g, m.color.b, m.color.a = 0.8, 0.2, 0.2, 0.6
            arr.markers.append(m)
        self.pub_dbg.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = SlalomPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
