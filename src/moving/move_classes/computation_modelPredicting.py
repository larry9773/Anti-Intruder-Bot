#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import numpy as np
import cv2

class RobotComputationController(Node):
    def __init__(self):
        super().__init__('robot_computation_controller')

        # Publisher pour les commandes de haut niveau
        self.action_pub = self.create_publisher(String, 'robot_action', 10)
        # Publisher pour les commandes de mouvement (Twist)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriptions pour action_done et images caméra
        self.create_subscription(String, 'action_done', self.action_done_callback, 10)
        self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)

        # Timer pour undock
        self.undock_sent = False
        self.undock_timer = self.create_timer(2.0, self.send_undock)

        # CV Bridge
        self.bridge = CvBridge()

        # Chargement du modèle TensorFlow (format zip)
        model_path = '/home/ptruong/COM-304_ws/com-304-robotics-project/RL_Habitat_Homework/Anti-Intruder-Bot/training/PPO/ball_nav/models/ppo_ball_finder.zip'  
        self.model = tf.keras.models.load_model(model_path)
        self.get_logger().info(f"Modèle chargé depuis: {model_path}")

    def send_undock(self):
        if not self.undock_sent:
            msg = String()
            msg.data = "undock"
            self.action_pub.publish(msg)
            self.get_logger().info("Commande envoyée: undock")
            self.undock_sent = True
            self.undock_timer.cancel()

    def action_done_callback(self, msg):
        self.get_logger().info(f"Action terminée: {msg.data}")
        if msg.data == "undock_done":
            self.sequence = ["drive", "rotate_left", "drive", "rotate_left", "drive", "rotate_left"]
            self.seq_index = 0

        if self.seq_index < len(self.sequence):
            self.send_next_command()
        else:
            self.get_logger().info("Parcours du triangle terminé.")

    def send_next_command(self):
        command = self.sequence[self.seq_index]
        self.seq_index += 1
        msg = String()
        msg.data = command
        self.action_pub.publish(msg)
        self.get_logger().info(f"Commande envoyée: {command}")

    def preprocess(self, cv_image):
        """
        - Convertit BGR en RGB
        - Redimensionne à 128x128
        - Normalise et ajoute batch dimension
        """
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Keras attend shape (batch, height, width, channels)
        return np.expand_dims(img_normalized, axis=0)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Erreur de conversion d'image: {e}")
            return

        # Prétraitement
        input_tensor = self.preprocess(cv_image)

        # Prédiction via model.predict()
        preds = self.model.predict(input_tensor)
        action = int(np.argmax(preds[0]))

        # Création du Twist selon l'action
        cmd = Twist()
        if action == 0:  # Avancer
            cmd.linear.x = 0.2
            cmd.angular.z = 0.0
            self.get_logger().info("Action navigation: avancer")
        elif action == 1:  # Tourner à gauche
            cmd.linear.x = 0.0
            cmd.angular.z = 0.3
            self.get_logger().info("Action navigation: tourner à gauche")
        elif action == 2:  # Tourner à droite
            cmd.linear.x = 0.0
            cmd.angular.z = -0.3
            self.get_logger().info("Action navigation: tourner à droite")
        elif action == 3:  # Arrêter
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("Action navigation: arrêter")
        else:
            self.get_logger().warn("Action inconnue calculée")

        # Publication de la commande de mouvement
        self.cmd_pub.publish(cmd)


def compute_reward(distance_to_goal, collision):
    reward = -distance_to_goal
    if collision:
        reward -= 1.0
    if distance_to_goal < 0.2:
        reward += 10.0
    return reward


def main(args=None):
    rclpy.init(args=args)
    node = RobotComputationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruption clavier, arrêt du noeud.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
