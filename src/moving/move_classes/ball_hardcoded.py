import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import LaserScan, Image, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
import math

# Undock action (TurtleBot4 Lite)
from  irobot_create_msgs.action import Undock


# --- Détection de la balle en 2D (HSV) ---
class BallDetector:
    def __init__(self):
        self.lower1 = np.array([0,120,70])
        self.upper1 = np.array([10,255,255])
        self.lower2 = np.array([170,120,70])
        self.upper2 = np.array([180,255,255])

    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, self.lower1, self.upper1)
        m2 = cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        c = max(contours, key=cv2.contourArea)
        (x, y), r = cv2.minEnclosingCircle(c)
        if r < 5:
            return None, None
        return (int(x), int(y)), r

# --- Contrôleur par champs de potentiel ---
class PotentialFieldController:
    def __init__(self, img_w, f_x, repulsive_range=1.0):
        self.cx = img_w / 2
        self.f_x = f_x
        self.ball_R = 0.15
        self.K_att = 1.0
        self.K_rep = 0.5
        self.rep_range = repulsive_range
        self.min_forward = 0.0
        self.max_lin = 0.3
        self.max_ang = 1.0
        self.eps_lin = 1e-2
        self.eps_ang = 1e-2

    def compute(self, center, radius, scan):
        # Attraction
        if center is None:
            att_x, att_y = 0.0, 0.0
        else:
            theta = math.atan((center[0] - self.cx) / self.f_x)
            dist = (self.f_x * self.ball_R) / radius
            att_x = self.K_att * dist * math.cos(theta)
            att_y = self.K_att * dist * math.sin(theta)
        # Répulsion
        rep_x, rep_y = 0.0, 0.0
        angle = scan.angle_min
        for r in scan.ranges:
            if np.isfinite(r) and 0 < r < self.rep_range:
                mag = self.K_rep * (1.0 / r - 1.0 / self.rep_range)
                rep_x -= mag * math.cos(angle)
                rep_y -= mag * math.sin(angle)
            angle += scan.angle_increment
        vx = att_x + rep_x
        vy = att_y + rep_y
        desired_theta = math.atan2(vy, vx)
        lin = np.hypot(vx, vy)
        lin = max(self.min_forward, min(lin, self.max_lin))
        ang = max(-self.max_ang, min(self.K_att * desired_theta, self.max_ang))
        if center is None:
            lin = 0.0
            ang = math.copysign(0.3, desired_theta)
        if abs(lin) < self.eps_lin and abs(ang) < self.eps_ang:
            ang = np.random.uniform(-1.0, 1.0)
            lin = 0.0
        return lin, ang

# --- Node ROS2 combinant undock puis navigation ---
class AutonomousNavigator(Node):
    def __init__(self):
        super().__init__('autonomous_navigator')
        self.bridge = CvBridge()
        self.detector = BallDetector()
        self.img_w = 640
        self.img_h = 480
        HFOV = math.radians(69.0)
        fx = self.img_w / (2 * math.tan(HFOV / 2))
        self.ctrl = PotentialFieldController(img_w=self.img_w, f_x=fx)

        # Undock action client
        self.undock_client = ActionClient(self, Undock, 'undock')
        self.undocked = False
        self._send_undock_goal()

        # Topic subs/pubs
        self.create_subscription(Image, '/oakd/rgb/preview/image_raw', self.image_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.create_subscription(Imu, '/imu', self.imu_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.last_image = None
        self.last_scan = None

        # Timer 10Hz
        self.create_timer(0.1, self.timer_cb)

    def _send_undock_goal(self):
        self.undock_client.wait_for_server()
        goal_msg = Undock.Goal()
        # Usage standard : pas de champ dock_action pour Undock
        send_goal_future = self.undock_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._undock_response)

    def _undock_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Undock goal rejeté')
            return
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._undock_done)

    def _undock_done(self, future):
        result = future.result().result
        if result.success:
            self.undocked = True
            self.get_logger().info('Undock terminé, démarrage navigation')
        else:
            self.get_logger().error('Undock échoué')

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.last_image = cv2.resize(img, (self.img_w, self.img_h))

    def scan_cb(self, msg):
        self.last_scan = msg

    def imu_cb(self, msg):
        q = msg.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def timer_cb(self):
        if not self.undocked:
            return
        if self.last_image is None or self.last_scan is None:
            return
        center, radius = self.detector.detect(self.last_image)
        lin, ang = self.ctrl.compute(center, radius, self.last_scan)
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = AutonomousNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
