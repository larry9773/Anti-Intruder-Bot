#!/usr/bin/env python3
"""
ROS 2 node : décideur de haut niveau contrôlé par un modèle PPO
Entraîné dans BallFinderEnv.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image          # flux caméra
from std_msgs.msg import String            # ordres haut niveau
from cv_bridge import CvBridge
import numpy as np
from stable_baselines3 import PPO

# ------------------------------------------------------------
# Quelques hyper-paramètres qu’il faudra peut-être ajuster :
MODEL_PATH      = "/absolute/path/to/models/ppo_ball_finder.zip"
IMAGE_TOPIC     = "/oakd/rgb/preview/image_raw"      # topic caméra du Create 3
TARGET_IMG_SIZE = (720, 1280)                       # tailles utilisées au training
DEVICE          = "cuda"                          # ou "cuda" si vous avez un GPU
# ------------------------------------------------------------

class PolicyController(Node):
    def __init__(self):
        super().__init__("policy_controller")

        # 1) Chargement du modèle
        self.get_logger().info(f"Chargement du modèle : {MODEL_PATH}")
        self.model = PPO.load(MODEL_PATH, device=DEVICE)

        # 2) ROS I/O
        self.pub_cmd  = self.create_publisher(String, "robot_action", 10)
        self.sub_done = self.create_subscription(
            String, "action_done", self.cb_action_done, 10
        )
        self.sub_img  = self.create_subscription(
            Image, IMAGE_TOPIC, self.cb_image, 10
        )

        # 3) État interne
        self.bridge         = CvBridge()
        self.last_obs       = None          # observation la plus récente
        self.action_pending = False         # en attente d’un retour action_done

        # 4) Send an undock first so the robot is free to move
        self.send_command("undock")

    # =================================================================
    # callback : nouvelle image
    def cb_image(self, msg: Image):
        # Convertit ROS Image → np.ndarray (BGR)
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Pré-processing identique au training
        obs = self.preprocess(frame)
        self.last_obs = {"image": obs}  # BallFinderEnv expose probablement un dict

        # Si aucune action n’est en cours, on peut décider la suivante
        if not self.action_pending:
            self.decide_and_send()

    # callback : action terminée
    def cb_action_done(self, _msg: String):
        self.action_pending = False
        # Dès qu’on reçoit le DONE, on décide à nouveau
        if self.last_obs is not None:
            self.decide_and_send()

    # =================================================================
    # fonctions utilitaires
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize + normalise exactement comme dans BallFinderEnv"""
        import cv2
        img = cv2.resize(frame, TARGET_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0          # [0, 1]
        img = np.transpose(img, (2, 0, 1))            # C × H × W
        return img

    def decide_and_send(self):
        """Appelle le modèle et publie la commande """
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
