#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image, Imu, LaserScan
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf_transformations import quaternion_matrix
from stable_baselines3 import PPO
from turtlebot_detector_node import DetectorNode

# ------------------------------------------------------------
# Hyper-paramètres
MODEL_PATH      = "/home/ezy21kl/Documents/Anti-Intruder-Bot/src/moving/move_classes/ball_finder_trainer_edgemere.zip"
DEVICE          = "cuda"
IMAGE_TOPIC     = "/oakd/rgb/preview/image_raw"
IMU_TOPIC       = "/oakd/imu/data"
LIDAR_TOPIC     = "/scan"
TARGET_IMG_SIZE = (1280, 720)
GRAVITY         = np.array([0.0, 0.0, 9.80665], dtype=np.float32)
# ------------------------------------------------------------

class Computation2Controller(Node):
    def __init__(self):
        super().__init__('computation2_controller')

        # Chargement du modèle PPO
        self.get_logger().info(f"Chargement du modèle : {MODEL_PATH}")
        self.model = PPO.load(
            MODEL_PATH,
            device=DEVICE,
            custom_objects={"n_steps": 1, "n_envs": 1}
        )
        self.get_logger().info("Modèle chargé !")

        # Publisher / Subscriber
        self.pub_action = self.create_publisher(String, 'robot_action', 10)
        self.sub_done   = self.create_subscription(String, 'action_done', self.cb_action_done, 10)

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_img   = self.create_subscription(Image, IMAGE_TOPIC, self.cb_image, qos)
        self.sub_imu   = self.create_subscription(Imu,   IMU_TOPIC,   self.cb_imu,   qos)
        self.sub_lidar = self.create_subscription(LaserScan, LIDAR_TOPIC, self.cb_lidar, qos)

        # Bridge OpenCV 
        self.bridge = CvBridge()

        # État interne
        self.last_image     = None
        self.last_lidar     = None
        self.last_position  = np.zeros(3, dtype=np.float32)
        self.last_velocity  = np.zeros(3, dtype=np.float32)
        self.last_imu_time  = None
        self.last_obs       = None
        self.action_pending = False

        self.detector_node = DetectorNode()

        # On envoie un "undock" au démarrage
        self.send_command("undock")

    # ----------------- Callbacks capteurs -----------------

    def cb_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # même prétraitement que PolicyController
        img = cv2.resize(frame, TARGET_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        self.last_image = img
        self.update_last_obs()
        self.try_decide()

    def cb_imu(self, msg: Imu):
        # calcul du dt
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = stamp
            return
        dt = stamp - self.last_imu_time
        self.last_imu_time = stamp

        # rotation et soustraction de la gravité
        acc_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float32)
        q = msg.orientation
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        acc_world = rot.dot(acc_body) - GRAVITY

        # intégration pour obtenir vit. et pos.
        self.last_velocity += acc_world * dt
        self.last_position += self.last_velocity * dt

        self.update_last_obs()
        self.try_decide()

    def cb_lidar(self, msg: LaserScan):
        # 1) Conversion en float32
        ranges = np.array(msg.ranges, dtype=np.float32)
        # 2) Filtrage des infinis / NaN
        ranges = np.nan_to_num(
            ranges,
            nan=msg.range_max,
            posinf=msg.range_max,
            neginf=msg.range_min
        )
        # 3) Échantillonnage uniforme à 360 points
        if ranges.size > 360:
            idx = np.linspace(0, ranges.size - 1, 360).astype(int)
            ranges = ranges[idx]
        # 4) Stockage et mise à jour de l’observation
        self.last_lidar = ranges
        self.update_last_obs()
        self.try_decide()


    # ----------------- Logique de décision -----------------

    def update_last_obs(self):
        if self.last_image is not None and self.last_lidar is not None:
            self.last_obs = {
                "rgb":   self.last_image,
                "lidar": self.last_lidar,
                "imu":   self.last_position
            }

    def try_decide(self):
        if self.action_pending:
            return
        if self.last_obs is None:
            return

        # détection ballon
        results = self.detector_node.detector.detect(self.last_image)
        balls = results.get("balls", [])
        if balls:
            best = max(balls, key=lambda b: b["confidence"])
            pos = best["position"]  # 'left', 'center', 'right'
            self.get_logger().info(f"Balle détectée à {pos} (conf {best['confidence']:.2f})")
            if pos == "left":
                cmd = "rotate_left"
            elif pos == "right":
                cmd = "rotate_right"
            else:
                cmd = "move_forward"
            self.send_command(cmd)
        else:
            # politique PPO
            action_id, _ = self.model.predict(self.last_obs, deterministic=True)
            cmd = {0: "move_forward", 1: "move_backward", 2: "rotate_left", 3: "rotate_right"}\
                  .get(int(action_id), "move_forward")
            self.send_command(cmd)

    def cb_action_done(self, _msg: String):
        self.get_logger().info(f"Action done: {_msg.data}")
        self.action_pending = False
        # dès qu’on reçoit le done, on redécide
        self.try_decide()

    # ----------------- Envoi de commandes -----------------

    def send_command(self, cmd_str: str):
        msg = String()
        msg.data = cmd_str
        self.pub_action.publish(msg)
        self.get_logger().info(f"→ envoi : {cmd_str}")
        self.action_pending = True

    def destroy_node(self):
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Computation2Controller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt demandé (Ctrl-C).")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
