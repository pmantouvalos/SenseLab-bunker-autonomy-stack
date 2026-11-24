"""
dashboard_server: web dashboard of the rover (open http://localhost:8080 in a browser).

A small HTTP server inside a ROS 2 node (Python standard library only, no extra
packages, works offline). The page lets the operator

  * choose MANUAL / AUTO / STANDBY and press the EMERGENCY STOP (and RESET),
  * drive by hand (buttons, WASD/arrow keys, speed slider) in MANUAL mode,
  * watch the YOLO camera (artifact boxes) and the lane view (mask + lane centre),
  * see the autonomy state, the lane, the Stanley numbers and the speed,
  * get a flashing WARNING + sound whenever an artifact (the figures to find) is found, with the
    portrait of the figure, and see it on a map of the track,

The page only sends *requests*; the autonomy_manager decides what the robot does.

    in  : /autonomy/state, /artifacts/report, /artifacts/crop, /avoid/debug, /auto/stanley_debug, /odom,
          /oak/nn/detections_image, /lane/debug_image
    out : /autonomy/command (String), /manual/cmd_vel (Twist, 10 Hz, dead-man),
          /cmd_vel (zero Twists, only as an extra layer when E-STOP is pressed)

Parameters: pose_topic ('/odom'; the simulator uses '/ground_truth/odom'), port (8080), host ('127.0.0.1' = this computer only. Use '0.0.0.0' to open
the dashboard from another device, but then ANYONE on the network can drive the robot).
"""
import base64
import json
import math
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from visualization_msgs.msg import MarkerArray

COMMANDS = {'MANUAL', 'AUTO', 'STANDBY', 'ESTOP', 'RESET'}
ARTIFACT_RE = re.compile(r'ARTIFACT FOUND #(\d+): (.+?) at \(([-\d.]+), ([-\d.]+), ([-\d.]+)\)')


class Stream:
    """Latest JPEG of an image topic, shared with the HTTP threads."""

    def __init__(self):
        self.jpeg = None
        self.stamp = 0
        self.cond = threading.Condition()

    def put(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.stamp += 1
            self.cond.notify_all()


class Dashboard(Node):

    def __init__(self):
        super().__init__('dashboard')
        self.declare_parameter('port', 8080)
        self.declare_parameter('host', '127.0.0.1')
        self.declare_parameter('jpeg_quality', 70)
        self.declare_parameter('pose_topic', '/odom')
        self.quality = self.get_parameter('jpeg_quality').value
        self.bridge = CvBridge()

        self.lock = threading.Lock()
        self.state = {'state': 'NO_MANAGER', 'mode': 'NONE', 'reason': 'autonomy_manager is not running'}
        self.state_time = 0.0
        self.alerts = []
        self.stanley = [0.0] * 6
        self.speed = 0.0
        self.robot = dict(x=0.0, y=0.0, yaw=0.0)
        self.rocks = []
        self.crops = {}                 # figure name -> (quality, jpeg bytes, version)
        self.drive = (0.0, 0.0, 0.0)    # lin, ang, wall time of the last request
        self.streams = {'camera': Stream(), 'lane': Stream()}

        self.pub_cmd = self.create_publisher(String, '/autonomy/command', 10)
        self.pub_manual = self.create_publisher(Twist, '/manual/cmd_vel', 10)
        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(String, '/autonomy/state', self.on_state, 5)
        self.create_subscription(String, '/artifacts/report', self.on_report, 20)
        self.create_subscription(Float32MultiArray, '/auto/stanley_debug', self.on_stanley, 5)
        self.create_subscription(Odometry, '/odom', self.on_odom, 5)
        self.create_subscription(Odometry, self.get_parameter('pose_topic').value, self.on_pose, 5)
        self.create_subscription(String, '/artifacts/crop', self.on_crop, 20)
        self.create_subscription(MarkerArray, '/avoid/debug', self.on_rocks, 2)
        self.create_subscription(Image, '/oak/nn/detections_image',
                                 lambda m: self.on_image('camera', m), 2)
        self.create_subscription(Image, '/lane/debug_image',
                                 lambda m: self.on_image('lane', m), 2)
        self.create_timer(0.1, self.send_manual)   # 10 Hz manual stream

        share = get_package_share_directory('mine_rover_core')
        with open(os.path.join(share, 'dashboard', 'index.html'), 'rb') as f:
            self.html = f.read()
        host, port = self.get_parameter('host').value, self.get_parameter('port').value
        self.httpd = ThreadingHTTPServer((host, port), self.make_handler())
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.get_logger().info(f'Dashboard: http://{"localhost" if host == "127.0.0.1" else host}:{port}')

    # ------------------------------------------------------------- ROS inputs
    def on_state(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.state, self.state_time = d, time.monotonic()

    def on_report(self, msg):
        m = ARTIFACT_RE.search(msg.data)
        alert = dict(text=msg.data, time=time.strftime('%H:%M:%S'))
        if m:
            alert.update(id=int(m[1]), name=m[2], x=float(m[3]), y=float(m[4]), z=float(m[5]))
        else:
            alert.update(id=len(self.alerts) + 1, name='unknown')
        with self.lock:
            self.alerts.append(alert)
        self.get_logger().warn(f'ALERT shown on dashboard: {msg.data}')

    def on_crop(self, msg):
        try:
            d = json.loads(msg.data)
            jpg = base64.b64decode(d['jpeg'])
        except (ValueError, KeyError):
            return
        quality = d.get('score', 0.0) * d.get('size', 0)       # confident AND big = a good portrait
        with self.lock:
            old = self.crops.get(d['name'])
            if old is None or quality > old[0]:
                self.crops[d['name']] = (quality, jpg, (old[2] + 1) if old else 1)

    def on_pose(self, msg):
        q, p = msg.pose.pose.orientation, msg.pose.pose.position
        with self.lock:
            self.robot = dict(x=p.x, y=p.y, yaw=math.atan2(2 * (q.w * q.z + q.x * q.y),
                                                            1 - 2 * (q.y * q.y + q.z * q.z)))

    def on_rocks(self, msg):
        rocks = [dict(x=m.pose.position.x, y=m.pose.position.y, sx=m.scale.x, sy=m.scale.y)
                 for m in msg.markers]
        with self.lock:
            self.rocks = rocks

    def on_stanley(self, msg):
        with self.lock:
            self.stanley = list(msg.data)

    def on_odom(self, msg):
        with self.lock:
            self.speed = msg.twist.twist.linear.x

    def on_image(self, name, msg):
        s = self.streams[name]
        if time.monotonic() - getattr(s, 'last_encode', 0.0) < 0.08:   # max ~12 fps
            return
        s.last_encode = time.monotonic()
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if ok:
            s.put(jpg.tobytes())

    # ------------------------------------------------------------- outputs
    def send_manual(self):
        lin, ang, t = self.drive
        if time.monotonic() - t > 0.4:       # dead-man: the page stopped asking
            lin = ang = 0.0
        msg = Twist()
        msg.linear.x, msg.angular.z = float(lin), float(ang)
        self.pub_manual.publish(msg)

    def command(self, cmd):
        self.pub_cmd.publish(String(data=cmd))
        if cmd == 'ESTOP':
            self.drive = (0.0, 0.0, 0.0)
            for _ in range(5):               # extra layer: zero straight to the robot too
                self.pub_vel.publish(Twist())

    def snapshot(self):
        with self.lock:
            stale = time.monotonic() - self.state_time > 1.5
            state = dict(self.state)
            if stale:
                state.update(state='NO_MANAGER', mode='NONE',
                             reason='autonomy_manager is not responding')
            alerts = [dict(a, crop=self.crops[a['name']][2] if a.get('name') in self.crops else 0)
                      for a in self.alerts]
            return dict(state=state, alerts=alerts, stanley=self.stanley, speed=round(self.speed, 3),
                        robot=self.robot, rocks=self.rocks)

    # ------------------------------------------------------------- HTTP
    def make_handler(self):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send_json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(node.html)))
                    self.end_headers()
                    self.wfile.write(node.html)
                elif self.path == '/api/state':
                    self.send_json(node.snapshot())
                elif self.path.startswith('/crop/'):
                    name = self.path.split('/')[-1].split('?')[0].replace('.jpg', '')
                    with node.lock:
                        c = node.crops.get(name)
                    if c is None:
                        return self.send_error(404)
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-store')
                    self.send_header('Content-Length', str(len(c[1])))
                    self.end_headers()
                    self.wfile.write(c[1])
                elif self.path.startswith('/stream/'):
                    self.stream(self.path.split('/')[-1])
                else:
                    self.send_error(404)

            def do_POST(self):
                try:
                    n = int(self.headers.get('Content-Length', 0))
                    data = json.loads(self.rfile.read(n) or b'{}')
                except ValueError:
                    return self.send_json({'ok': False}, 400)
                if self.path == '/api/command':
                    cmd = str(data.get('cmd', '')).upper()
                    if cmd not in COMMANDS:
                        return self.send_json({'ok': False, 'error': 'unknown command'}, 400)
                    node.command(cmd)
                    return self.send_json({'ok': True})
                if self.path == '/api/drive':
                    lin = max(-1.0, min(1.0, float(data.get('lin', 0.0))))
                    ang = max(-2.0, min(2.0, float(data.get('ang', 0.0))))
                    node.drive = (lin, ang, time.monotonic())
                    return self.send_json({'ok': True})
                self.send_error(404)

            def stream(self, name):
                s = node.streams.get(name)
                if s is None:
                    return self.send_error(404)
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                last = -1
                try:
                    while rclpy.ok():
                        with s.cond:
                            s.cond.wait_for(lambda: s.stamp != last, timeout=2.0)
                            jpg, last = s.jpeg, s.stamp
                        if jpg is None:
                            continue
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                         + str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler


def main(args=None):
    rclpy.init(args=args)
    node = Dashboard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.httpd.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
