#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from rclpy.action import ActionClient
from irobot_create_msgs.action import Undock, RotateAngle, DriveDistance

class MoveController(Node):
    def __init__(self):
        super().__init__('move_controller')
        self.subscription = self.create_subscription(
            String,
            'robot_action',
            self.robot_action_callback,
            10)
        self.publisher = self.create_publisher(String, 'action_done', 10)
        self.action_in_progress = False

    def robot_action_callback(self, msg):
        if self.action_in_progress:
            self.get_logger().info("Action en cours. Commande ignorée.")
            return

        command = msg.data.strip().lower()
        self.get_logger().info(f"Commande reçue : {command}")
        self.action_in_progress = True

        if command == "undock":
            self.call_undock_action()
        elif command == "move_forward":
            # Choix de la distance selon la commande
            distance = 0.25
            self.drive_actionCall(distance)
        elif command == "move_backward":
            # Choix de la distance selon la commande
            distance = 0.25
            self.drive_actionCall(-distance)
        elif command == "rotate_left":
            self.call_rotate_action(0.523598)   # 30° en radians
        elif command == "rotate_right":
            self.call_rotate_action(-0.523598)  # -30° en radians
        else:
            self.get_logger().warn(f"Commande inconnue : {command}")
            self.action_in_progress = False

    def call_undock_action(self):
        self.undock_client = ActionClient(self, Undock, 'undock')
        self.get_logger().info("Attente du serveur d'action Undock...")
        if not self.undock_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Undock indisponible !")
            self.action_in_progress = False
            return

        goal_msg = Undock.Goal()
        self.get_logger().info("Envoi de la commande undock...")
        future = self.undock_client.send_goal_async(goal_msg)
        future.add_done_callback(self.undock_response_callback)

    def undock_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Undock rejeté.")
            self.action_in_progress = False
            return
        self.get_logger().info("Undock accepté, attente du résultat...")
        goal_handle.get_result_async().add_done_callback(self.undock_result_callback)

    def undock_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("Action Undock terminée.")
        self.action_in_progress = False
        msg = String()
        msg.data = "undock_done"
        self.publisher.publish(msg)
    
    def drive_actionCall(self, distance):
        self.drive_client = ActionClient(self, DriveDistance, 'drive_distance')
        self.get_logger().info("Attente du serveur DriveDistance...")
        if not self.drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Serveur DriveDistance indisponible!")
            self.action_in_progress = False
            return

        goal_msg = DriveDistance.Goal()
        goal_msg.distance = distance  # Distance en mètres
        self.get_logger().info(f"Envoi de DriveDistance ({distance} m)...")
        future = self.drive_client.send_goal_async(goal_msg)
        future.add_done_callback(self.drive_goal_response_callback)

    def drive_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("DriveDistance rejeté.")
            self.action_in_progress = False
            return
        self.get_logger().info("DriveDistance accepté, attente du résultat...")
        goal_handle.get_result_async().add_done_callback(self.drive_result_callback)

    def drive_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("DriveDistance terminé.")
        self.action_in_progress = False
        msg = String()
        msg.data = "drive_done"
        self.publisher.publish(msg)

    def call_rotate_action(self, angle):
        self.rotate_client = ActionClient(self, RotateAngle, 'rotate_angle')
        self.get_logger().info("Attente du serveur RotateAngle...")
        if not self.rotate_client.wait_for_server(timeout_sec=4.0):
            self.get_logger().error("Serveur RotateAngle indisponible!")
            self.action_in_progress = False
            return

        goal_msg = RotateAngle.Goal()
        goal_msg.angle = angle
        self.get_logger().info(f"Envoi de RotateAngle ({angle} rad)...")
        future = self.rotate_client.send_goal_async(goal_msg)
        future.add_done_callback(self.rotate_goal_response_callback)

    def rotate_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("RotateAngle rejeté.")
            self.action_in_progress = False
            return
        self.get_logger().info("RotateAngle accepté, attente du résultat...")
        goal_handle.get_result_async().add_done_callback(self.rotate_result_callback)

    def rotate_result_callback(self, future):
        result = future.result().result
        self.get_logger().info("RotateAngle terminé.")
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
        node.get_logger().info("Interruption clavier, arrêt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

