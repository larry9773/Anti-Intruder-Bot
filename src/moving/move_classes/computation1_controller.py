#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class SimplePolicy(nn.Module):
    """
    Politique neuronale simple qui traite une image et retourne 4 logits correspondant aux actions :
    0: Avancer, 1: Tourner à gauche, 2: Tourner à droite, 3: Arrêter.
    """
    def __init__(self):
        super(SimplePolicy, self).__init__()
        # Ces dimensions sont données à titre indicatif.
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        # Calcul de la dimension après convolutions (ici pour une image de 128x128)
        self.fc1 = nn.Linear(32 * 29 * 29, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RobotComputationController(Node):
    def __init__(self):
        super().__init__('robot_computation_controller')
        
        # Publisher pour les commandes de haut niveau (ex: "undock", "drive", "rotate_left", etc.)
        self.action_pub = self.create_publisher(String, 'robot_action', 10)
        # Publisher pour les commandes de mouvement (Twist) à destination du robot
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscription pour les messages signalant la complétion d'une action
        self.create_subscription(String, 'action_done', self.action_done_callback, 10)
        # Subscription pour les images de la caméra
        self.create_subscription(Image, '/camera/rgb/image_raw', self.image_callback, 10)
        
        # Timer pour envoyer la commande "undock" après 2 secondes
        self.undock_sent = False
        self.undock_timer = self.create_timer(2.0, self.send_undock)
        
        # Variables pour gérer la séquence de commandes (ex: dessin d'un triangle)
        self.sequence = []
        self.seq_index = 0
        
        # CV Bridge pour convertir les messages Image ROS en images OpenCV
        self.bridge = CvBridge()
        
        # Initialisation d'une politique simple (à remplacer par votre modèle entraîné)
        self.policy = SimplePolicy()
        self.policy.eval()  # mode évaluation

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
        # Après la commande undock, initialisation de la séquence pour dessiner un triangle
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
        Prétraitement de l'image OpenCV :
         - Conversion BGR vers RGB
         - Redimensionnement à 128x128
         - Normalisation et conversion en tenseur PyTorch
        """
        img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (128, 128))
        img_normalized = img_resized.astype(np.float32) / 255.0
        tensor_img = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor_img

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Erreur de conversion d'image: {str(e)}")
            return

        # Prétraitement de l'image
        tensor_img = self.preprocess(cv_image)
        
        # Calcul de l'action via la politique (ici, on effectue une passe avant)
        with torch.no_grad():
            action_logits = self.policy(tensor_img)
        action = torch.argmax(action_logits, dim=1).item()

        # Création du message Twist selon l'action déterminée
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
