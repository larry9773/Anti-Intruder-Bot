#!/usr/bin/env python3
"""
ROS 2 node : décideur de haut niveau contrôlé par un modèle PPO
Entraîné dans BallFinderEnv.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, Imu, LaserScan
from std_msgs.msg import String            # ordres haut niveau
from cv_bridge import CvBridge
import numpy as np
from stable_baselines3 import PPO
import cv2
from tf_transformations import quaternion_matrix
from turtlebot_detector_node import DetectorNode

# ------------------------------------------------------------
# Quelques hyper-paramètres qu’il faudra peut-être ajuster :
MODEL_PATH      = "/home/ezy21kl/Documents/Anti-Intruder-Bot/src/moving/move_classes/ball_finder_trainer_edgemere.zip"
IMAGE_TOPIC     = "/oakd/rgb/preview/image_raw"      # topic caméra du Create 3
TARGET_IMG_SIZE = (1280, 720)                       # tailles utilisées au training
DEVICE          = "cuda"                          # ou "cuda" si vous avez un GPU
IMU_TOPIC       = "/oakd/imu/data"
LIDAR_TOPIC     = "/scan"
GRAVITY         = np.array([0.0, 0.0, 9.80665], dtype=np.float32)
# ------------------------------------------------------------

class PolicyController(Node):
    def __init__(self):
        super().__init__("policy_controller")

        # 1) Chargement du modèle
        self.get_logger().info(f"Chargement du modèle : {MODEL_PATH}")
        #self.model = PPO.load(MODEL_PATH, device=DEVICE)
        self.model = PPO.load(
            MODEL_PATH,
            device=DEVICE,
            custom_objects={
                "n_steps": 1,   # réduit buffer_size à 1
                "n_envs": 1     # réduit nombre d’envs à 1
            }
        )
        self.get_logger().info("Model loaded !")

        # 2) ROS I/O
        self.pub_cmd  = self.create_publisher(String, "robot_action", 10)
        self.sub_done = self.create_subscription(
            String, "action_done", self.cb_action_done, 10
        )

        qos = QoSProfile(
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1)

        self.sub_img  = self.create_subscription(
            Image, IMAGE_TOPIC, self.cb_image, qos_profile=qos
        )

        self.sub_imu   = self.create_subscription(Imu,   IMU_TOPIC,   self.cb_imu,   qos_profile=qos)
        self.sub_lidar = self.create_subscription(LaserScan, LIDAR_TOPIC, self.cb_lidar, qos_profile=qos)

        # 3) État interne
        self.bridge         = CvBridge()
        self.last_obs       = None          # observation la plus récente
        self.action_pending = False         # en attente d’un retour action_done

        self.last_lidar = None
        self.last_position = None
        self.last_image = None

        self.last_position     = np.zeros(3, dtype=np.float32)
        self.last_velocity = np.zeros(3, dtype=np.float32)
        self.last_imu_time = None

        self.detector_node = DetectorNode()

        # 4) Send an undock first so the robot is free to move
        self.send_command("undock")

    # =================================================================
    # callback : nouvelle image
    def cb_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        img = self.preprocess_image(frame)
        self.last_image = img
        self.update_last_obs()
        #self.try_decide()

    def cb_imu(self, msg: Imu):
        # Calculate dt
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = stamp
            return
        dt = stamp - self.last_imu_time
        self.last_imu_time = stamp

        # Rotate acceleration to world frame and remove gravity
        acc_body = np.array([msg.linear_acceleration.x,
                              msg.linear_acceleration.y,
                              msg.linear_acceleration.z], dtype=np.float32)
        # quaternion -> rotation matrix
        q = msg.orientation
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        acc_world = rot.dot(acc_body) - GRAVITY

        # Update velocity and position via integration
        self.last_velocity += acc_world * dt
        self.last_position     += self.last_velocity * dt
        self.update_last_obs()
        #self.try_decide()

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

    def try_decide(self):
        # Décider seulement quand tous les capteurs ont fourni des données
        if not self.action_pending and \
           (self.last_image is not None) and \
           (self.last_position   is not None) and \
           (self.last_lidar is not None):
            results = self.detector_node.detector.detect(self.last_image)
            balls = results.get("balls", [])
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
                self.send_command(cmd)
            else:
                self.decide_and_send()

    def update_last_obs(self):
        self.last_obs = {
            "rgb": self.last_image,
            "lidar": self.last_lidar,
            "imu": self.last_position
        }

    # callback : action terminée
    def cb_action_done(self, _msg: String):
        self.action_pending = False
        # Dès qu’on reçoit le DONE, on décide à nouveau
        '''if self.last_obs is not None:
            self.decide_and_send()'''
        self.try_decide()

    # =================================================================
    # fonctions utilitaires
    def preprocess_image(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, TARGET_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        #img = np.transpose(img, (2, 0, 1))  # C,H,W
        return img

    def decide_and_send(self):
        """Appelle le modèle et publie la commande """
        self.get_logger().info("last obs = ", self.last_obs)
        action_id, _ = self.model.predict(self.last_obs, deterministic=True)
        cmd_str = self.id_to_command(action_id)
        self.send_command(cmd_str)

    
    def send_command(self, cmd_str: str):
        self.get_logger().info(f"→ envoi : {cmd_str}")
        msg = String();  msg.data = cmd_str
        self.pub_cmd.publish(msg)
        self.action_pending = True                    # on attend le DONE

    # =================================================================
    # mapping action_id → chaîne comprise par move2_controller.py
    @staticmethod
    def id_to_command(a: int) -> str:
        """
        0 : avancer court (0.50 m)  
        1 : avancer long  (1.00 m)  
        2 : tourner gauche (+90°)  
        3 : tourner droite(-90°)  
        """
        lookup = {0: "move_forward",
                  1: "move_backward",
                  2: "rotate_left",
                  3: "rotate_right"}
        return lookup.get(a, "move_forward")  # par défaut on avance

# =======================================================================
def main(args=None):
    rclpy.init(args=args)
    node = PolicyController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt demandé (Ctrl-C).")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
