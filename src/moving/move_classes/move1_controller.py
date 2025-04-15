#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.action import ActionClient

# Import des actions pour contrôler le robot iRobot Create
from irobot_create_msgs.action import Undock, RotateAngle, DriveDistance

class MoveController(Node):
    def __init__(self):
        super().__init__('move_controller')
        # Souscription aux commandes venant du module de computation (topic robot_action)
        self.subscription = self.create_subscription(
            String,
            'robot_action',
            self.robot_action_callback,
            10)
        # Publication d'un message indiquant la fin de l'action (topic action_done)
        self.publisher = self.create_publisher(String, 'action_done', 10)
        self.action_in_progress = False

    def robot_action_callback(self, msg: String):
        if self.action_in_progress:
            self.get_logger().info("Une action est déjà en cours. Nouvelle commande ignorée.")
            return

        command = msg.data.strip().lower()
        self.get_logger().info(f"Commande reçue: {command}")
        self.action_in_progress = True

        if command == "undock":
            self.call_undock_action()
        elif command == "drive":
            self.drive_action_call()
        elif command == "rotate_left":
            # Pour tourner à gauche, on utilise 120° (2.0944 radians)
            self.call_rotate_action(2.0944)
        elif command == "rotate_right":
            # Pour tourner à droite, on utilise -120° (-2.0944 radians)
            self.call_rotate_action(-2.0944)
        else:
            self.get_logger().warn(f"Commande inconnue: {command}")
            self.action_in_progress = False

    def call_undock_action(self):
        self.undock_client = ActionClient(self, Undock, 'undock')
        self.get_logger().info("Attente du serveur d'action Undock...")
        if not self.undock_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Serveur Undock non disponible!")
            self.action_in_progress = False
            return

        goal_msg = Undock.Goal()
        self.get_logger().info("Envoi de l'objectif undock...")
        future = self.undock_client.send_goal_async(goal_msg)
        future.add_done_callback(self.undock_response_callback)

    def undock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Objectif undock rejeté.")
            self.action_in_progress = False
            return
        self.get_logger().info("Objectif undock accepté, attente du résultat...")
        goal_handle.get_result_async().add_done_callback(self.undock_result_callback)

    def undock_result_callback(self, future):
        # Vous pouvez traiter ici le résultat si nécessaire
        self.get_logger().info("Action undock terminée.")
        self.action_in_progress = False
        msg = String()
        msg.data = "undock_done"
        self.publisher.publish(msg)
    
    def drive_action_call(self):
        self.drive_client = ActionClient(self, DriveDistance, 'drive_distance')
        self.get_logger().info("Attente du serveur d'action DriveDistance...")
        if not self.drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Serveur DriveDistance non disponible!")
            self.action_in_progress = False
            return

        goal_msg = DriveDistance.Goal()
        goal_msg.distance = 0.5  # Avancer de 0.5 m (exemple pour un côté de triangle)
        self.get_logger().info("Envoi de l'objectif drive (0.5 m)...")
        future = self.drive_client.send_goal_async(goal_msg)
        future.add_done_callback(self.drive_goal_response_callback)

    def drive_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Objectif drive rejeté.")
            self.action_in_progress = False
            return
        self.get_logger().info("Objectif drive accepté, attente du résultat...")
        goal_handle.get_result_async().add_done_callback(self.drive_result_callback)

    def drive_result_callback(self, future):
        # Vous pouvez traiter ici le résultat si nécessaire
        self.get_logger().info("Action drive terminée.")
        self.action_in_progress = False
        msg = String()
        msg.data = "drive_done"
        self.publisher.publish(msg)

    def call_rotate_action(self, angle: float):
        self.rotate_client = ActionClient(self, RotateAngle, 'rotate_angle')
        self.get_logger().info("Attente du serveur d'action RotateAngle...")
        if not self.rotate_client.wait_for_server(timeout_sec=4.0):
            self.get_logger().error("Serveur RotateAngle non disponible!")
            self.action_in_progress = False
            return

        goal_msg = RotateAngle.Goal()
        goal_msg.angle = angle
        self.get_logger().info(f"Envoi de l'objectif rotate ({angle} rad)...")
        future = self.rotate_client.send_goal_async(goal_msg)
        future.add_done_callback(self.rotate_goal_response_callback)

    def rotate_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Objectif rotate rejeté.")
            self.action_in_progress = False
            return
        self.get_logger().info("Objectif rotate accepté, attente du résultat...")
        goal_handle.get_result_async().add_done_callback(self.rotate_result_callback)

    def rotate_result_callback(self, future):
        # Vous pouvez traiter ici le résultat si nécessaire
        self.get_logger().info("Action rotate terminée.")
        self.action_in_progress = False
        msg = String()
        msg.data = "rotate_done"
        self.publisher.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MoveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruption clavier, arrêt du noeud.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
