# computation_controller.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import cv2
from stable_baselines3 import PPO

class ComputationController(Node):
    """
    ROS2 node that computes actions using a PPO model and publishes
    high-level commands on 'robot_action'.
    """
    def __init__(self):
        super().__init__('computation_controller')
        self.model = PPO.load('/path/to/ball_nav/models/ball_finder.zip')
        self.bridge = CvBridge()
        self.rgb = None
        self.scan = None
        self.success_dist = 0.3
        self.action_in_progress = False
        # Publishers and subscribers
        self.command_pub = self.create_publisher(String, 'robot_action', 10)
        self.create_subscription(String, 'action_done', self.done_cb, 10)
        self.create_subscription(Image, '/camera/color/image_raw', self.image_cb, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info('ComputationController started')

    def image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        img = cv2.resize(img, (128, 128))
        self.rgb = img.astype(np.float32) / 255.0

    def scan_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges[np.isinf(ranges)] = np.nan
        self.scan = ranges

    def done_cb(self, msg: String):
        self.get_logger().info(f"Action done: {msg.data}")
        self.action_in_progress = False

    def get_ball_bearing_index(self):
        if self.rgb is None or self.scan is None:
            return None
        img = (self.rgb * 255).astype(np.uint8)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower1, upper1 = np.array([0,100,100]), np.array([10,255,255])
        lower2, upper2 = np.array([160,100,100]), np.array([179,255,255])
        mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cx = x + w/2
        # convert pixel to lidar index around center
        mid = len(self.scan) // 2
        return mid

    def control_loop(self):
        if self.action_in_progress or self.rgb is None or self.scan is None:
            return
        idx = self.get_ball_bearing_index()
        if idx is not None:
            dist = self.scan[idx]
            if not np.isnan(dist) and dist < self.success_dist:
                self.get_logger().info('Ball reached')
                self.timer.cancel()
                # send stop command
                msg = String(); msg.data='stop'
                self.command_pub.publish(msg)
                return
        obs = {'rgb': self.rgb, 'scan': self.scan[:360]}
        action, _ = self.model.predict(obs, deterministic=True)
        cmd = ['drive_short','drive_long','rotate_left','rotate_right'][action]
        self.command_pub.publish(String(data=cmd))
        self.action_in_progress = True

    def destroy_node(self):
        self.timer.cancel()
        super().destroy_node()



def main(args=None):
    rclpy.init(args=args)
    node = ComputationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()