import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import math
from  irobot_create_msgs.action import Undock

class BallDetector:
    def __init__(self):
        self.lower1 = np.array([0,120,70])
        self.upper1 = np.array([10,255,255])
        self.lower2 = np.array([170,120,70])
        self.upper2 = np.array([180,255,255])
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.lower1, self.upper1)
        m2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 5:
            return None, None
        return (int(x), int(y)), r

class PotentialFieldController:
    def __init__(self, img_w, f_x, rep_range=1.0):
        self.cx = img_w / 2
        self.f_x = f_x
        self.ball_R = 0.05
        self.K_att = 1.0
        self.K_rep = 0.5
        self.rep_range = rep_range
        self.min_lin = 0.0
        self.max_lin = 0.3
        self.max_ang = 1.0
        self.eps_lin = 1e-2
        self.eps_ang = 1e-2
    def compute(self, center, r, scan):
        if center is None:
            att_x = att_y = 0.0
        else:
            theta = math.atan((center[0] - self.cx) / self.f_x)
            dist = (self.f_x * self.ball_R) / r
            att_x = self.K_att * dist * math.cos(theta)
            att_y = self.K_att * dist * math.sin(theta)
        ranges = np.array(scan.ranges)
        valid = (ranges > 0) & (ranges < self.rep_range) & np.isfinite(ranges)
        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        mags = self.K_rep * (1.0 / ranges[valid] - 1.0 / self.rep_range)
        rep_x = -np.sum(mags * np.cos(angles[valid]))
        rep_y = -np.sum(mags * np.sin(angles[valid]))
        vx = att_x + rep_x
        vy = att_y + rep_y
        desired = math.atan2(vy, vx)
        lin = math.hypot(vx, vy)
        lin = max(self.min_lin, min(lin, self.max_lin))
        ang_vel = max(-self.max_ang, min(self.K_att * desired, self.max_ang))
        if center is None:
            lin = self.max_lin * 0.5
            ang_vel = max(-self.max_ang, min(self.K_att * desired, self.max_ang))
        if abs(lin) < self.eps_lin and abs(ang_vel) < self.eps_ang:
            lin = 0.0
            ang_vel = np.random.uniform(-1.0, 1.0)
        return lin, ang_vel

class AutonomousNavigator(Node):
    def __init__(self):
        super().__init__('autonomous_navigator')
        self.bridge = CvBridge()
        self.detector = BallDetector()
        self.img_w = 640
        self.img_h = 480
        hfov = math.radians(69.0)
        fx = self.img_w / (2 * math.tan(hfov / 2))
        self.ctrl = PotentialFieldController(self.img_w, fx)
        self.last_image = None
        self.last_scan = None
        self.undocked = False
        self.undock_client = ActionClient(self, Undock, 'undock')
        self.undock_client.wait_for_server()
        self.undock_client.send_goal_async(Undock.Goal()).add_done_callback(self._undock_response)
        self.create_subscription(Image, 'oakd/rgb/preview/image_raw', self.image_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.1, self.timer_cb)
    def _undock_response(self, future):
        handle = future.result()
        if not handle.accepted:
            return
        handle.get_result_async().add_done_callback(self._undock_done)
    def _undock_done(self, future):
        result = future.result().result
        if result.success:
            self.undocked = True
    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.last_image = cv2.resize(img, (self.img_w, self.img_h))
    def scan_cb(self, msg):
        self.last_scan = msg
    def timer_cb(self):
        if not self.undocked or self.last_image is None or self.last_scan is None:
            return
        center, r = self.detector.detect(self.last_image)
        lin, ang = self.ctrl.compute(center, r, self.last_scan)
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        cmd.linear.y = 0.0
        cmd.linear.z = 0.0
        cmd.angular.x = 0.0
        cmd.angular.y = 0.0
        self.pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
