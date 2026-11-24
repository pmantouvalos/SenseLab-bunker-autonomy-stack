"""
autonomy_manager: the state machine of the rover's autonomy, and the ONLY node that
writes /cmd_vel.

Everything that wants to move the robot (the operator's dashboard, the Stanley lane
follower, future behaviours) sends its command to its own topic. This node decides,
state by state, which one is allowed through. If it is not allowed, zero is sent.

                          MANUAL ──────────────┐
                            ▲  │ AUTO          │
                    MANUAL  │  ▼               │
    STANDBY ───AUTO──► AUTO_ACQUIRE ─lane ok─► AUTO_FOLLOW ◄─────────┐
       ▲                  │ timeout              │   │ artifact      │ hold over
       │                  ▼                      │   ▼ (optional)    │
       ├───────────── (back to STANDBY)          │  AUTO_ARTIFACT_HOLD
       │                                         │ lane lost
       │   timeout                               ▼
       └─────────────────────────────────── AUTO_LANE_LOST ──lane back──► AUTO_FOLLOW

    Big rock ahead (slalom_planner):  AUTO_FOLLOW ──► AUTO_OBSTACLE_STOP ──► AUTO_AVOID ──► AUTO_FOLLOW
                                      (stop at a safe distance, plan)   (drive the curve)
                                      no gap wide enough: AUTO_BLOCKED (waits for the operator)

    ESTOP: reachable from EVERY state, latched, zero velocity at 20 Hz. The only way out
    is the RESET command (which goes to STANDBY, never straight back to motion).

Where to add new autonomy: add a state name to STATES, a handler `_tick_<name>` that
returns the Twist to send (or None for stop) and calls `self.enter(...)` to change
state, and, if the operator can choose it, a command in `on_command`.

Topics
    in  /autonomy/command   std_msgs/String   MANUAL | AUTO | STANDBY | ESTOP | RESET
        /manual/cmd_vel     geometry_msgs/Twist  from the dashboard (dead-man: stale = stop)
        /auto/cmd_vel       geometry_msgs/Twist  from the Stanley controller (lane)
        /avoid/cmd_vel      geometry_msgs/Twist  from the second Stanley (slalom path)
        /avoid/status       std_msgs/String      JSON from slalom_planner
        /lane/status        std_msgs/String      JSON from lane_detector
        /artifacts/report   std_msgs/String      from artifact_reporter
    out /cmd_vel            geometry_msgs/Twist  to the robot, 20 Hz, always
        /autonomy/state     std_msgs/String      JSON for the dashboard, 5 Hz
"""
import json

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

STANDBY = 'STANDBY'
MANUAL = 'MANUAL'
AUTO_ACQUIRE = 'AUTO_ACQUIRE'
AUTO_FOLLOW = 'AUTO_FOLLOW'
AUTO_LANE_LOST = 'AUTO_LANE_LOST'
AUTO_ARTIFACT_HOLD = 'AUTO_ARTIFACT_HOLD'
AUTO_OBSTACLE_STOP = 'AUTO_OBSTACLE_STOP'   # big rock ahead: stop at a safe distance
AUTO_AVOID = 'AUTO_AVOID'                   # drive the slalom path of slalom_planner
AUTO_BLOCKED = 'AUTO_BLOCKED'               # no gap wide enough: wait for the operator
ESTOP = 'ESTOP'
STATES = (STANDBY, MANUAL, AUTO_ACQUIRE, AUTO_FOLLOW, AUTO_LANE_LOST, AUTO_ARTIFACT_HOLD,
          AUTO_OBSTACLE_STOP, AUTO_AVOID, AUTO_BLOCKED, ESTOP)

MODE_OF = {STANDBY: 'STANDBY', MANUAL: 'MANUAL', ESTOP: 'ESTOP',
           AUTO_ACQUIRE: 'AUTO', AUTO_FOLLOW: 'AUTO', AUTO_LANE_LOST: 'AUTO',
           AUTO_ARTIFACT_HOLD: 'AUTO', AUTO_OBSTACLE_STOP: 'AUTO', AUTO_AVOID: 'AUTO',
           AUTO_BLOCKED: 'AUTO'}


class AutonomyManager(Node):

    def __init__(self):
        super().__init__('autonomy_manager')
        self.declare_parameter('manual_timeout', 0.5)      # s without a manual command -> stop
        self.declare_parameter('auto_timeout', 0.5)        # s without a Stanley command -> stop
        self.declare_parameter('acquire_time', 0.5)        # s of valid lane to start following
        self.declare_parameter('acquire_timeout', 10.0)    # s waiting for a lane before giving up
        self.declare_parameter('lane_lost_time', 1.0)      # s without lane -> AUTO_LANE_LOST
        self.declare_parameter('lane_lost_timeout', 6.0)   # s lost -> give up (STANDBY)
        self.declare_parameter('artifact_hold_s', 0.0)     # >0: stop this long at every new artifact
        self.declare_parameter('avoid_speed', 0.25)        # m/s limit while driving the slalom
        self.declare_parameter('stop_settle_s', 1.0)       # s standing still before planning is used
        self.declare_parameter('trigger_dist', 2.2)        # m between robot nose and rock to start
        self.declare_parameter('avoid_timeout', 90.0)      # s in AVOID before giving up
        self.declare_parameter('max_manual_lin', 0.8)
        self.declare_parameter('max_manual_ang', 1.5)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.manual_timeout, self.auto_timeout = g('manual_timeout'), g('auto_timeout')
        self.acquire_time, self.acquire_timeout = g('acquire_time'), g('acquire_timeout')
        self.lane_lost_time, self.lane_lost_timeout = g('lane_lost_time'), g('lane_lost_timeout')
        self.artifact_hold_s = g('artifact_hold_s')
        self.max_lin, self.max_ang = g('max_manual_lin'), g('max_manual_ang')
        self.avoid_speed, self.stop_settle = g('avoid_speed'), g('stop_settle_s')
        self.avoid_timeout = g('avoid_timeout')
        self.trigger_dist = g('trigger_dist')

        self.state = STANDBY
        self.reason = 'Ready. Choose MANUAL or AUTO.'
        self.state_since = None
        self.manual = (Twist(), None)       # (message, time received)
        self.auto = (Twist(), None)
        self.lane = {'valid': False}
        self.lane_time = None
        self.lane_ok_since = None           # start of the current run of valid lane frames
        self.lane_bad_since = None
        self.artifacts = 0
        self.pending_artifact = False
        self.last_out = Twist()
        self.avoid = (Twist(), None)         # slalom Stanley command
        self.plan = {}                       # last /avoid/status from slalom_planner
        self.plan_time = None
        self.clear_since = None

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_state = self.create_publisher(String, '/autonomy/state', 5)
        self.create_subscription(String, '/autonomy/command', self.on_command, 10)
        self.create_subscription(Twist, '/manual/cmd_vel', self.on_manual, 10)
        self.create_subscription(Twist, '/auto/cmd_vel', self.on_auto, 10)
        self.create_subscription(Twist, '/avoid/cmd_vel', self.on_avoid, 10)
        self.create_subscription(String, '/avoid/status', self.on_plan, 10)
        self.create_subscription(String, '/lane/status', self.on_lane, 10)
        self.create_subscription(String, '/artifacts/report', self.on_artifact, 10)
        self.create_timer(0.05, self.tick)        # 20 Hz
        self.create_timer(0.2, self.publish_state)  # 5 Hz
        self.get_logger().info('Autonomy manager ready: STANDBY')

    # ------------------------------------------------------------- helpers
    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def age(self, t):
        return float('inf') if t is None else self.now() - t

    def in_state_for(self):
        return 0.0 if self.state_since is None else self.now() - self.state_since

    def enter(self, state, reason):
        if state == self.state:
            self.reason = reason
            return
        self.get_logger().info(f'{self.state} -> {state}: {reason}')
        self.state, self.reason, self.state_since = state, reason, self.now()
        self.lane_ok_since = self.lane_bad_since = None
        self.pending_artifact = False
        self.clear_since = None

    # ------------------------------------------------------------- inputs
    def on_command(self, msg):
        cmd = msg.data.strip().upper()
        if cmd == 'ESTOP':
            self.enter(ESTOP, 'EMERGENCY STOP pressed. Press RESET to release.')
        elif self.state == ESTOP:
            if cmd == 'RESET':
                self.enter(STANDBY, 'E-stop released. Choose MANUAL or AUTO.')
            else:
                self.get_logger().warn(f'"{cmd}" ignored: E-stop is active, press RESET first')
        elif cmd == 'MANUAL':
            self.enter(MANUAL, 'Operator is driving.')
        elif cmd == 'AUTO':
            self.enter(AUTO_ACQUIRE, 'Looking for the lane...')
        elif cmd in ('STANDBY', 'STOP'):
            self.enter(STANDBY, 'Stopped by the operator.')
        else:
            self.get_logger().warn(f'Unknown command "{msg.data}"')

    def on_manual(self, msg):
        self.manual = (msg, self.now())

    def on_auto(self, msg):
        self.auto = (msg, self.now())

    def on_avoid(self, msg):
        self.avoid = (msg, self.now())

    def on_plan(self, msg):
        try:
            self.plan = json.loads(msg.data)
        except ValueError:
            return
        self.plan_time = self.now()

    def plan_fresh(self):
        return self.plan_time is not None and self.age(self.plan_time) < 1.0

    def on_lane(self, msg):
        try:
            self.lane = json.loads(msg.data)
        except ValueError:
            return
        self.lane_time = self.now()
        if self.lane.get('valid'):
            self.lane_bad_since = None
            if self.lane_ok_since is None:
                self.lane_ok_since = self.now()
        else:
            self.lane_ok_since = None
            if self.lane_bad_since is None:
                self.lane_bad_since = self.now()

    def on_artifact(self, msg):
        self.artifacts += 1
        self.pending_artifact = True

    def lane_valid_for(self):
        """Seconds the lane has been continuously valid (0 if not valid / stale)."""
        if self.lane_time is None or self.age(self.lane_time) > 1.0 or self.lane_ok_since is None:
            return 0.0
        return self.now() - self.lane_ok_since

    def lane_missing_for(self):
        """Seconds without a valid lane (stale status counts as missing)."""
        if self.lane_time is None or self.age(self.lane_time) > 1.0:
            return self.in_state_for()
        if self.lane_bad_since is None:
            return 0.0
        return self.now() - self.lane_bad_since

    # ------------------------------------------------------------- state handlers
    # Each returns the Twist to send, or None to send zero.
    def _tick_standby(self):
        return None

    def _tick_estop(self):
        return None

    def _tick_manual(self):
        msg, t = self.manual
        if self.age(t) > self.manual_timeout:
            return None                       # dead-man: no recent command -> stop
        out = Twist()
        out.linear.x = max(-self.max_lin, min(self.max_lin, msg.linear.x))
        out.angular.z = max(-self.max_ang, min(self.max_ang, msg.angular.z))
        return out

    def _tick_auto_acquire(self):
        if self.lane_valid_for() >= self.acquire_time:
            self.enter(AUTO_FOLLOW, 'Lane found. Following it (Stanley).')
        elif self.in_state_for() > self.acquire_timeout:
            self.enter(STANDBY, 'No lane found. Autonomy stopped.')
        return None

    def _tick_auto_follow(self):
        if self.plan_fresh():
            if self.plan.get('blocked'):
                self.enter(AUTO_BLOCKED, 'No gap wide enough around the rock. Waiting for the operator.')
                return None
            dist = self.plan.get('dist')
            if self.plan.get('ready') and dist is not None and dist <= self.trigger_dist:
                self.enter(AUTO_OBSTACLE_STOP, f'Big rock {dist:.1f} m ahead. Stopping to plan the slalom.')
                return None
        if self.lane_missing_for() > self.lane_lost_time:
            self.enter(AUTO_LANE_LOST, 'Lane lost. Robot stopped.')
            return None
        if self.pending_artifact and self.artifact_hold_s > 0.0:
            self.enter(AUTO_ARTIFACT_HOLD, 'Artifact found. Holding position.')
            return None
        self.pending_artifact = False
        msg, t = self.auto
        if self.age(t) > self.auto_timeout:
            return None                       # no fresh Stanley command -> stop
        return msg

    def _tick_auto_obstacle_stop(self):
        if not self.plan_fresh() or self.plan.get('blocked'):
            if self.plan_fresh():
                self.enter(AUTO_BLOCKED, 'No gap wide enough around the rock. Waiting for the operator.')
            return None
        if self.in_state_for() >= self.stop_settle and self.plan.get('ready'):
            self.enter(AUTO_AVOID, 'Slalom path ready. Driving around the rock.')
        return None

    def _tick_auto_avoid(self):
        if self.plan_fresh() and self.plan.get('blocked'):
            self.enter(AUTO_BLOCKED, 'No gap wide enough around the rock. Waiting for the operator.')
            return None
        if self.in_state_for() > self.avoid_timeout:
            self.enter(STANDBY, 'Slalom took too long. Autonomy stopped.')
            return None
        if self.plan_fresh() and self.plan.get('clear'):
            if self.clear_since is None:
                self.clear_since = self.now()
            elif self.now() - self.clear_since > 0.5:
                self.enter(AUTO_FOLLOW, 'Rocks passed. Following the lane.')
                return None
        else:
            self.clear_since = None
        msg, t = self.avoid
        if self.age(t) > self.auto_timeout:
            return None
        out = Twist()
        # keep the same curvature but never faster than avoid_speed
        k = min(1.0, self.avoid_speed / msg.linear.x) if msg.linear.x > 1e-3 else 1.0
        out.linear.x, out.angular.z = msg.linear.x * k, msg.angular.z * k
        return out

    def _tick_auto_blocked(self):
        if self.plan_fresh() and not self.plan.get('blocked') and self.plan.get('ready'):
            self.enter(AUTO_OBSTACLE_STOP, 'A gap is available now. Planning again.')
        return None

    def _tick_auto_lane_lost(self):
        if self.lane_valid_for() >= self.acquire_time:
            self.enter(AUTO_FOLLOW, 'Lane found again. Following it (Stanley).')
        elif self.in_state_for() > self.lane_lost_timeout:
            self.enter(STANDBY, 'Lane lost for too long. Autonomy stopped.')
        return None

    def _tick_auto_artifact_hold(self):
        if self.in_state_for() > self.artifact_hold_s:
            self.enter(AUTO_FOLLOW, 'Hold over. Following the lane.')
        return None

    # ------------------------------------------------------------- main loop
    def tick(self):
        if self.state_since is None:
            if self.now() == 0.0:
                return                        # waiting for the simulation clock
            self.state_since = self.now()
        handler = getattr(self, '_tick_' + self.state.lower())
        out = handler()
        if out is None or self.state == ESTOP:
            out = Twist()
        self.last_out = out
        self.pub_cmd.publish(out)

    def publish_state(self):
        d = dict(state=self.state, mode=MODE_OF[self.state], reason=self.reason,
                 in_state_s=round(self.in_state_for(), 1), lane=self.lane,
                 lane_fresh=self.age(self.lane_time) < 1.0, artifacts=self.artifacts,
                 cmd=dict(lin=round(self.last_out.linear.x, 3), ang=round(self.last_out.angular.z, 3)))
        self.pub_state.publish(String(data=json.dumps(d)))


def main(args=None):
    rclpy.init(args=args)
    node = AutonomyManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_cmd.publish(Twist())          # leave the robot stopped
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
