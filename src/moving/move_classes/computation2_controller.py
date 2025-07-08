#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ComputationController(Node):
    def __init__(self):
        super().__init__('computation_controller')
        self.publisher = self.create_publisher(String, 'robot_action', 10)
        self.subscription = self.create_subscription(
            String,
            'action_done',
            self.action_done_callback,
            10)
        self.undock_sent = False
        self.sequence = []  # Séquence des commandes pour dessiner un rectangle
        self.seq_index = 0
        self.current_lap = 0
        self.total_laps = 3  # Nombre de tours complets
        self.undock_timer = self.create_timer(2.0, self.send_undock)

    def send_undock(self):
        if not self.undock_sent:
            msg = String()
            msg.data = "undock"
            self.publisher.publish(msg)
            self.get_logger().info("Envoyé : undock")
            self.undock_sent = True
            self.undock_timer.cancel()

    def action_done_callback(self, msg):
        self.get_logger().info(f"Action terminée : {msg.data}")
        # Après l'undock, on initialise la séquence du rectangle
        if msg.data == "undock_done" and not self.sequence:
            self.init_rectangle_sequence()

        if self.seq_index < len(self.sequence):
            self.send_next_command()
        else:
            self.current_lap += 1
            if self.current_lap < self.total_laps:
                self.get_logger().info(f"Tour {self.current_lap} terminé. Démarrage du tour suivant...")
                self.init_rectangle_sequence()
                self.send_next_command()
            else:
                self.get_logger().info("Parcours rectangulaire terminé après 3 tours.")

    def init_rectangle_sequence(self):
        # Rectangle à angle droit avec deux côtés longs et deux côtés courts
        self.sequence = [
            "drive_long", "rotate_left",
            "drive_short", "rotate_left",
            "drive_long", "rotate_left",
            "drive_short", "rotate_left"
        ]
        self.seq_index = 0

    def send_next_command(self):
        command = self.sequence[self.seq_index]
        self.seq_index += 1
        msg = String()
        msg.data = command
        self.publisher.publish(msg)
        self.get_logger().info(f"Commande envoyée : {command}")

def main(args=None):
    rclpy.init(args=args)
    node = ComputationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interruption clavier, arrêt.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

