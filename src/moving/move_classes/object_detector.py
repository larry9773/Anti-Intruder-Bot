import cv2
import numpy as np


class HumanDetector:
    def __init__(self, prototxt='MobileNetSSD_deploy.prototxt',
                 model='MobileNetSSD_deploy.caffemodel',
                 confidence_threshold=0.5,
                 debug=False):
        self.debug = debug
        self.confidence_threshold = confidence_threshold
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

    def detect(self, image):
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        detections = self.net.forward()

        results = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                if self.CLASSES[idx] != "person":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                center_x = (startX + endX) / 2
                position = get_relative_position(center_x, w)

                # Estimate distance based on bounding box height
                distance = estimate_human_distance(startY, endY, h)

                results.append({
                    "box": (startX, startY, endX, endY),
                    "confidence": float(confidence),
                    "position": position,
                    "distance": distance
                })

                if self.debug:
                    label = f"{self.CLASSES[idx]}: {confidence:.2f} ({position})"
                    cv2.rectangle(image, (startX, startY), (endX, endY),
                                  (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(image, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        if self.debug:
            cv2.imshow("Detections", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results


class RedBallDetector:
    def __init__(self, min_radius=10, min_confidence=0.8, debug=False):
        self.min_radius = min_radius
        self.min_confidence = min_confidence
        self.debug = debug

    def detect(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # red wraps around the hue circle → two ranges
        lower1, upper1 = np.array([0, 120, 70]),  np.array([10, 255, 255])
        lower2, upper2 = np.array([170, 120, 70]), np.array([180, 255, 255])

        mask  = cv2.inRange(hsv, lower1, upper1)
        mask |= cv2.inRange(hsv, lower2, upper2)

        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, 1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results = []

        h, w = image.shape[:2]
        for c in cnts:
            ((x, y), r) = cv2.minEnclosingCircle(c)
            if r < self.min_radius:
                continue

            contour_area = cv2.contourArea(c)
            circle_area = np.pi * r * r  # area of the enclosing circle
            confidence = min(1.0, contour_area / circle_area)

            if confidence < self.min_confidence:
                continue

            pos = get_relative_position(x, w)
            dist = estimate_ball_distance(r, h)

            results.append({
                "center":   (int(x), int(y)),
                "radius":   int(r),
                "position": pos,
                "distance": dist,
                "confidence": confidence
            })

            if self.debug:
                cv2.circle(image, (int(x), int(y)), int(r), (255, 0, 0), 2)
                cv2.putText(image, f"{pos} {confidence:.2f}", (int(x)-20, int(y)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if self.debug:
            cv2.imshow("Red-ball mask", mask)
            cv2.imshow("Ball detections", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return results


class PeopleAndBallDetector:
    def __init__(self,
                 prototxt='MobileNetSSD_deploy.prototxt',
                 model='MobileNetSSD_deploy.caffemodel',
                 conf_thresh=0.5,
                 ball_min_radius=10,
                 ball_conf_thresh=0.8,
                 debug=False):

        self.human = HumanDetector(prototxt, model, conf_thresh, debug)
        self.ball  = RedBallDetector(ball_min_radius, ball_conf_thresh, debug)
        self.debug = debug

    def detect(self, image):
        """Returns one dict containing both result lists."""
        return {
            "humans": self.human.detect(image),
            "balls":  self.ball.detect(image)
        }


def get_relative_position(x_center, image_width):
    if x_center < image_width / 2.2:
        return "left"
    elif x_center > image_width / 1.8:
        return "right"
    else:
        return "center"


def estimate_human_distance(startY, endY, h):
    box_height = endY - startY
    relative_size = box_height / h
    if relative_size > 0.8:
        return "very close"
    elif relative_size > 0.6:
        return "close"
    elif relative_size > 0.4:
        return "medium"
    else:
        return "far"


def estimate_ball_distance(radius_px, img_height):
    """
    Very rough distance proxy identical in spirit to the one you
    already use for the person bounding-box height.
    Tune the thresholds to taste.
    """
    relative = (2 * radius_px) / img_height      # diameter / image height
    if relative > 0.8:
        return "very close"
    elif relative > 0.6:
        return "close"
    elif relative > 0.4:
        return "medium"
    else:
        return "far"
