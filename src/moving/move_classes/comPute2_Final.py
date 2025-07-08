#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image, Imu, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf_transformations import quaternion_matrix
from stable_baselines3 import PPO
from irobot_create_msgs.action import Undock
from rclpy.action import ActionClient
from turtlebot_detector_node import DetectorNode
from std_msgs.msg import String
import time
from cv_bridge import CvBridge, CvBridgeError

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

        self.pub_action = self.create_publisher(String, 'robot_action', 10)
        self.undocking = False

        # QoS pour capteurs
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriptions capteurs
        self.sub_img   = self.create_subscription(Image, IMAGE_TOPIC, self.cb_image, qos)
        self.sub_imu   = self.create_subscription(Imu,   IMU_TOPIC,   self.cb_imu,   qos)
        self.sub_lidar = self.create_subscription(LaserScan, LIDAR_TOPIC, self.cb_lidar, qos)
        self.pub_cmdvel = self.create_publisher(Twist,  'cmd_vel', 10)

        # Bridge OpenCV
        self.bridge = CvBridge()

        # État interne
        self.last_image    = None
        self.last_lidar    = None
        self.last_position = np.zeros(3, dtype=np.float32)
        self.last_velocity = np.zeros(3, dtype=np.float32)
        self.last_imu_time = None
        self.last_obs      = None

        self.detector_node = DetectorNode()
        self.get_logger().info("Demande d'undock du robot au démarrage.")
        self.undock_client = ActionClient(self, Undock, 'undock')
        
        # Undock initial pour libérer le robot
        '''self.pub_cmd = self.create_publisher(String, 'cmd', 10)
        self.get_logger().info("Publication de 'undock' au démarrage.")
        undock_msg = String()
        undock_msg.data = 'undock'
        self.pub_cmd.publish(undock_msg)'''
        self.send_command("undock")
        self.frame = None
    
    def undock_robot(self):
    # Envoie la requête d'undock via action
        if not self.undock_client.wait_for_server(timeout_sec=100.0):
            self.get_logger().error('Serveur action Undock non disponible.')
            return
        goal_msg = Undock.Goal()
        self.get_logger().info('Envoi du goal Undock → libération du robot')
        self.undocking = True
        send_goal_future = self.undock_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.undock_response)

    def undock_response(self, future):
        result = future.result().result
        if result.success:
            self.get_logger().info('Robot undocké avec succès !')
        else:
            self.get_logger().error('Échec de l undock.')
        self.undocking = False


    # ----------------- Callbacks capteurs -----------------

    def cb_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.frame = frame
        img = cv2.resize(frame, TARGET_IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        self.last_image = img
        self.update_last_obs()
        self.try_decide()

    def cb_imu(self, msg: Imu):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = stamp
            return
        dt = stamp - self.last_imu_time
        self.last_imu_time = stamp

        acc_body = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ], dtype=np.float32)
        q = msg.orientation
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        acc_world = rot.dot(acc_body) - GRAVITY

        self.last_velocity += acc_world * dt
        self.last_position += self.last_velocity * dt

        self.update_last_obs()
        self.try_decide()

    def cb_lidar(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(
            ranges,
            nan=msg.range_max,
            posinf=msg.range_max,
            neginf=msg.range_min
        )
        if ranges.size > 360:
            idx = np.linspace(0, ranges.size - 1, 360).astype(int)
            ranges = ranges[idx]
        self.last_lidar = ranges
        self.update_last_obs()
        self.try_decide()

    # ----------------- Logic de décision -----------------

    def update_last_obs(self):
        if self.last_image is not None and self.last_lidar is not None:
            self.last_obs = {
                "rgb":   self.last_image,
                "lidar": self.last_lidar,
                "imu":   self.last_position
            }

    def try_decide(self):
        if self.last_obs is None or self.undocking or self.last_image is None or self.last_lidar is None or self.last_position is None:
            return
        '''try:
            frame = self.bridge.imgmsg_to_cv2(self.last_image, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return'''
        results = self.detector_node.detector.detect(self.frame)
        balls = results.get("balls", [])
        humans = results.get("humans", [])
        report_lines = []


        # ---------- humans ----------
        if not humans:
            report_lines.append("❌  No human")
        else:
            best = max(humans, key=lambda d: d["confidence"])
            report_lines.append(
                f"✅  Human {best['position']} / "
                f"{best['distance']} / "
                f"{best['confidence']:.2f}"
            )

        # ---------- balls -----------
        time.sleep(2)
        if not balls:
            report_lines.append("⭕  No red ball")
            action_id, _ = self.model.predict(self.last_obs, deterministic=True)
            cmd = {0: "move_forward", 1: "move_backward", 2: "rotate_left", 3: "rotate_right"} \
                  .get(int(action_id), "move_forward")
            self.send_cmd_vel(cmd)
        else:
            best = max(balls, key=lambda d: d["confidence"])
            report_lines.append(
                f"🔴  Ball {best['position']} / "
                f"{best['distance']} / "
                f"{best['confidence']:.2f}"
            )
            best = max(balls, key=lambda b: b["confidence"])
            pos = best["position"]
            self.get_logger().info(f"Balle détectée à {pos} (conf {best['confidence']:.2f})")
            if pos == "left":
                cmd = "rotate_left"
            elif pos == "right":
                cmd = "rotate_right"
            else:
                cmd = "move_forward"

            self.send_vel1(cmd)
            

        self.get_logger().info(" | ".join(report_lines))
        '''if balls:
            best = max(balls, key=lambda b: b["confidence"])
            pos = best["position"]
            self.get_logger().info(f"Balle détectée à {pos} (conf {best['confidence']:.2f})")
            if pos == "left":
                cmd = "rotate_left"
            elif pos == "right":
                cmd = "rotate_right"
            else:
                cmd = "move_forward"
        else:
            action_id, _ = self.model.predict(self.last_obs, deterministic=True)
            cmd = {0: "move_forward", 1: "move_backward", 2: "rotate_left", 3: "rotate_right"} \
                  .get(int(action_id), "move_forward")'''
        #self.send_cmd_vel(cmd)

    # ----------------- Envoi de cmd_vel -----------------

    def send_cmd_vel(self, cmd_str: str):
        self.get_logger().info("envoi commande !")
        twist = Twist()
        # Ajustez les vitesses linéaire/angulaire selon vos besoins
        if cmd_str == "move_forward":
            twist.linear.x = 0.2
        elif cmd_str == "move_backward":
            twist.linear.x = -0.2
        elif cmd_str == "rotate_left":
            twist.angular.z = 0.5
        elif cmd_str == "rotate_right":
            twist.angular.z = -0.5
        # Sinon, twist reste à zéro (arrêt)

        self.pub_cmdvel.publish(twist)
        self.get_logger().info(f"Published cmd_vel → linear: {twist.linear.x:.2f}, angular: {twist.angular.z:.2f}")

    
    def send_vel1(self, cmd_str: str):
        self.get_logger().info("envoi commande !")
        twist = Twist()
        # Ajustez les vitesses linéaire/angulaire selon vos besoins
        if cmd_str == "move_forward":
            twist.linear.x = 0.2
        elif cmd_str == "move_backward":
            twist.linear.x = -0.2
        elif cmd_str == "rotate_left":
            twist.angular.z = 0.125
        elif cmd_str == "rotate_right":
            twist.angular.z = -0.125
        # Sinon, twist reste à zéro (arrêt)

        self.pub_cmdvel.publish(twist)
        self.get_logger().info(f"Published cmd_vel → linear: {twist.linear.x:.2f}, angular: {twist.angular.z:.2f}")

    def destroy_node(self):
        super().destroy_node()

    def send_command(self, cmd_str: str):
        msg = String()
        msg.data = cmd_str
        self.pub_action.publish(msg)
        self.get_logger().info(f"→ envoi : {cmd_str}")
        self.action_pending = True


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
