#!/usr/bin/env python3
"""
Subscribes to TurtleBot 4 Lite's RGB camera topic, runs the combined
People-and-Red-Ball detector, and prints a short report for every frame.

Requires:
  • rclpy                 (ROS 2 client library in Python)
  • sensor_msgs.msg.Image (standard ROS image message)
  • cv_bridge             (convert ROS images ↔︎ OpenCV BGR)
  • opencv-python         (already needed by your detector)
  • the local module      'human_detector.py' containing PeopleAndBallDetector
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from object_detector import PeopleAndBallDetector


class DetectorNode(Node):
    """
    Minimal subscriber node: receives raw camera frames,
    feeds them into PeopleAndBallDetector, and logs a summary.
    """

    def __init__(self):
        super().__init__("people_and_ball_detector")

        # Parameters you might want to expose on the ROS param server
        self.declare_parameter("camera_topic",
                               "/camera/color/image_raw")  # TB4 default RGB topic
        self.declare_parameter("human_conf_thresh", 0.50)
        self.declare_parameter("ball_conf_thresh", 0.50)
        self.declare_parameter("ball_min_radius", 10)
        self.declare_parameter("debug_windows", False)

        topic          = self.get_parameter("camera_topic").get_parameter_value().string_value
        human_thresh   = self.get_parameter("human_conf_thresh").get_parameter_value().double_value
        ball_thresh    = self.get_parameter("ball_conf_thresh").get_parameter_value().double_value
        ball_min_rad   = self.get_parameter("ball_min_radius").get_parameter_value().integer_value
        debug_flag     = self.get_parameter("debug_windows").get_parameter_value().bool_value

        # Detector instance
        self.detector = PeopleAndBallDetector(
            conf_thresh=human_thresh,
            ball_min_radius=ball_min_rad,
            ball_conf_thresh=ball_thresh,
            debug=debug_flag
        )

        # OpenCV/ROS bridge
        self.bridge = CvBridge()

        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            topic,
            self.image_callback,
            10,      # QoS depth
        )
        self.subscription  # prevent unused-variable warning

        self.get_logger().info(f"Listening to camera topic: {topic}")

    # ------------------------------------------------------------------
    def image_callback(self, msg: Image):
        """Convert ROS image → OpenCV BGR, run detector, print results."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        results        = self.detector.detect(frame)
        humans         = results.get("humans", [])
        balls          = results.get("balls",  [])

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
        if not balls:
            report_lines.append("⭕  No red ball")
        else:
            best = max(balls, key=lambda d: d["confidence"])
            report_lines.append(
                f"🔴  Ball {best['position']} / "
                f"{best['distance']} / "
                f"{best['confidence']:.2f}"
            )

        self.get_logger().info(" | ".join(report_lines))

        # If debug windows were requested we have to keep them responsive.
        if self.detector.debug:
            # A very small waitKey keeps the HighGUI event loop alive.
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Ensure graceful shutdown
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()