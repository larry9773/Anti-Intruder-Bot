#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from rclpy.action import ActionClient
from irobot_create_msgs.action import Undock, RotateAngle, DriveDistance

class MoveController(Node):
    def __init__(self):
        super().__init__('move_controller')

        # Souscription aux cmd_vel publiés par Computation2Controller
        self.sub_cmdvel = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmdvel_callback,
            10
        )

        self.subscription = self.create_subscription(
            String,
            'robot_action',
            self.cmd_callback,
            10)

        self.sub_cmd = self.create_subscription(
            String, 'cmd', self.cmd_callback, 10
        )

        self.undock_client = ActionClient(self, Undock, 'undock')

        # Publisher pour notifier la fin d'action
        self.pub_done = self.create_publisher(String, 'action_done', 10)
        self.action_in_progress = False

        # Paramètres par défaut
        self.default_distance = 0.25    # m
        self.default_angle = 0.523598   # rad (30°)

    def cmd_callback(self, msg: String):
        if msg.data == 'undock' and not self.action_in_progress:
            self.action_in_progress = True
            self._undock()

    def _undock(self):
        self.get_logger().info("Commande de undock reçue !")
        if not self.undock_client.wait_for_server(timeout_sec=100.0):
            self.get_logger().error("Serveur Undock indisponible !")
            self.action_in_progress = False
            return
        goal = Undock.Goal()
        self.get_logger().info("Envoi Undock → libération du robot")
        fut = self.undock_client.send_goal_async(goal)
        fut.add_done_callback(self._on_undock_response)

    def _on_undock_response(self, future):
        result = future.result().result
        status = 'undock_failed'
        if result.success:
            self.get_logger().info("Undock terminé avec succès.")
            status = 'undock_done'
        else:
            self.get_logger().error("Échec de l’undock.")
        self._publish_done(status)

    def cmdvel_callback(self, msg: Twist):
        if self.action_in_progress:
            self.get_logger().debug("Action en cours, commande cmd_vel ignorée.")
            return

        lin = msg.linear.x
        ang = msg.angular.z
        self.get_logger().info(f"cmd_vel reçu → linear: {lin:.2f}, angular: {ang:.2f}")
        self.action_in_progress = True

        # Priorité au mouvement linéaire
        if abs(lin) > 1e-3:
            distance = self.default_distance if lin > 0 else -self.default_distance
            self._drive_distance(distance)
        elif abs(ang) > 1e-3:
            angle = self.default_angle if ang > 0 else -self.default_angle
            self._rotate_angle(angle)
        else:
            # Pas de commande => réinitialiser l'état
            self.action_in_progress = False

    def _drive_distance(self, distance: float):
        self.drive_client = ActionClient(self, DriveDistance, 'drive_distance')
        if not self.drive_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Serveur DriveDistance indisponible!")
            self.action_in_progress = False
            return

        goal = DriveDistance.Goal()
        goal.distance = distance
        self.get_logger().info(f"Envoi DriveDistance: {distance:.2f} m")
        future = self.drive_client.send_goal_async(goal)
        future.add_done_callback(self._on_drive_response)

    def _on_drive_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("DriveDistance rejeté.")
            self.action_in_progress = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_drive_done)

    def _on_drive_done(self, future):
        self.get_logger().info("DriveDistance terminé.")
        self._publish_done('drive_done')

    def _rotate_angle(self, angle: float):
        self.rotate_client = ActionClient(self, RotateAngle, 'rotate_angle')
        if not self.rotate_client.wait_for_server(timeout_sec=4.0):
            self.get_logger().error("Serveur RotateAngle indisponible!")
            self.action_in_progress = False
            return

        goal = RotateAngle.Goal()
        goal.angle = angle
        self.get_logger().info(f"Envoi RotateAngle: {angle:.2f} rad")
        future = self.rotate_client.send_goal_async(goal)
        future.add_done_callback(self._on_rotate_response)

    def _on_rotate_response(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("RotateAngle rejeté.")
            self.action_in_progress = False
            return
        goal_handle.get_result_async().add_done_callback(self._on_rotate_done)

    def _on_rotate_done(self, future):
        self.get_logger().info("RotateAngle terminé.")
        self._publish_done('rotate_done')

    def _publish_done(self, status: str):
        self.action_in_progress = False
        msg = String()
        msg.data = status
        self.pub_done.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MoveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Arrêt demandé (Ctrl-C).")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
