#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import numpy as np
import cv2
from stable_baselines3 import PPO
from object_detector import PeopleAndBallDetector
from turtlebot_detector_node import DetectorNode

MODEL_PATH      = "/home/ezy21kl/Documents/Anti-Intruder-Bot/src/moving/move_classes/ball_finder_trainer_edgemere.zip"
DEVICE          = "cuda"                          # ou "cuda" si vous avez un GPU

class Computation2Controller(Node):
    """
    ROS2 node that runs a trained PPO policy to find a red ball.
    Publishes high-level commands ('undock', 'move_forward', 'move_backward',
    'rotate_left', 'rotate_right') to 'robot_action', listens to 'action_done'.
    Uses camera, lidar, and odometry to build observations.
    """
    def __init__(self):
        super().__init__('computation2_controller')
        # Load trained model (ensure the path matches your setup)
        #self.model = PPO.load('/home/ezy21kl/Documents/Anti-Intruder-Bot/src/moving/move_classes/ball_finder_trainer_edgemere.zip')
        self.model = PPO.load(
            MODEL_PATH,
            device=DEVICE,
            custom_objects={
                "n_steps": 1,   # réduit buffer_size à 1
                "n_envs": 1     # réduit nombre d’envs à 1
            }
        )
        # ROS interfaces
        self.pub_action = self.create_publisher(String, 'robot_action', 10)
        self.sub_done   = self.create_subscription(
            String, 'action_done', self.action_done_callback, 10
        )
        self.sub_image  = self.create_subscription(
            Image, '/camera/color/image_raw', self.image_cb, 10
        )
        self.sub_scan   = self.create_subscription(
            LaserScan, '/scan', self.scan_cb, 10
        )
        self.sub_odom   = self.create_subscription(
            Odometry, '/odom', self.odom_cb, 10
        )
        # State
        self.bridge = CvBridge()
        self.rgb = None
        self.lidar = None
        self.odom = None
        self.undocked = False
        self.action_in_progress = False

        self.detector_node = DetectorNode()

        # Send undock at startup
        self.create_timer(1.0, self.send_undock)
        self.get_logger().info('Computation2Controller started')

    def send_undock(self):
        if not self.undocked:
            msg = String(); msg.data = 'undock'
            self.pub_action.publish(msg)
            self.get_logger().info('Sent: undock')
            self.undocked = True

    def image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # Resize to match training resolution: 720x1280 (height x width)
        self.rgb = cv2.resize(img, (1280, 720))

    def scan_cb(self, msg: LaserScan):
        arr = np.array(msg.ranges, dtype=np.float32)
        arr[np.isinf(arr)] = np.nan
        # keep first 360 beams as trained
        self.lidar = arr[:360]

    def odom_cb(self, msg: Odometry):
        pos = msg.pose.pose.position
        self.odom = np.array([pos.x, pos.y, pos.z], dtype=np.float32)

    def action_done_callback(self, msg: String):
        self.get_logger().info(f'Action done: {msg.data}')
        self.action_in_progress = False
        if msg.data == 'undock_done':
            return
        self.send_rl_action()

    def send_rl_action(self):

        if self.action_in_progress or self.rgb is None or self.lidar is None or self.odom is None:
            return
        obs = {
            'rgb': self.rgb,
            'lidar': self.lidar,
            'imu': self.odom,
        }

        results = self.detector_node.detector.detect(self.rgb)
        balls = results.get('balls', [])

        if balls:
            # 2) Au moins une balle détectée : on prend la plus confiante
            best_ball = max(balls, key=lambda b: b['confidence'])
            pos = best_ball['position']  # 'left', 'center' ou 'right'
            self.get_logger().info(
                f"Balle détectée à {pos} (conf {best_ball['confidence']:.2f})"
            )

            # 3) Policy hard-codée :
            if pos == 'left':
                cmd = 'rotate_left'
            elif pos == 'right':
                cmd = 'rotate_right'
            else:  # 'center'
                cmd = 'move_forward'

            self.get_logger().info(f"Policy balle → commande : {cmd}")
        else:
            action, _ = self.model.predict(obs, deterministic=True)
            cmd = ['move_forward', 'move_backward', 'rotate_left', 'rotate_right'][int(action)]
            msg = String(); msg.data = cmd
            self.pub_action.publish(msg)
            self.get_logger().info(f'Published RL action: {cmd}')
            self.action_in_progress = True

    def destroy_node(self):
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Computation2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
