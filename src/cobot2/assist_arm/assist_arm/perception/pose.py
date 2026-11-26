"""웹캠 영상 → MediaPipe Hands(21점) + Pose(33점) → /user/landmarks (assist_arm_msgs/UserLandmarks)."""
import sys

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from assist_arm_msgs.msg import UserLandmarks
from assist_arm.common.heartbeat import attach_heartbeat
from assist_arm.common.mp_loader import HandTracker, MediaPipeMissing, PoseTracker


class PoseNode(Node):
    def __init__(self):
        super().__init__('pose_node')
        p = self.declare_parameter
        p('source', 'webcam')              # webcam (아무 USB 캠: webcam_publisher device) / realsense
        p('image_topic', '')               # 비우면 source 에 맞게: /webcam/image_raw 또는 /camera/color/image_raw
        p('landmarks_topic', '/user/landmarks')
        p('enable_hands', True)
        p('enable_pose', True)
        p('model_complexity', 0)        # 0 = CPU 친화
        p('num_hands', 2)
        p('min_hand_detection_confidence', 0.5)
        p('min_hand_presence_confidence', 0.5)
        p('min_hand_tracking_confidence', 0.5)
        p('max_rate_hz', 15.0)
        p('hand_task_model', '~/data/assist_arm/models/hand_landmarker.task')   # mediapipe 1.x (Tasks API)
        p('pose_task_model', '~/data/assist_arm/models/pose_landmarker_lite.task')
        p('show_debug', False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.hands = HandTracker(max_hands=int(g('num_hands')),
                                 min_det=float(g('min_hand_detection_confidence')),
                                 min_presence=float(g('min_hand_presence_confidence')),
                                 min_track=float(g('min_hand_tracking_confidence')),
                                 model_complexity=g('model_complexity'),
                                 task_model_path=g('hand_task_model')) if g('enable_hands') else None
        self.pose = PoseTracker(model_complexity=g('model_complexity'),
                                task_model_path=g('pose_task_model')) if g('enable_pose') else None
        self.show = g('show_debug')
        self.min_dt = 1.0 / max(1e-3, float(g('max_rate_hz')))
        self.last_t = 0.0
        self.bridge = CvBridge()
        topic = g('image_topic') or ('/camera/color/image_raw' if g('source') == 'realsense' else '/webcam/image_raw')
        self.pub = self.create_publisher(UserLandmarks, g('landmarks_topic'), 10)
        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        attach_heartbeat(self)
        self.get_logger().info(f"pose_node 준비 (source={g('source')}, image={topic})")

    def on_image(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_t < self.min_dt:
            return
        self.last_t = now
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        out = UserLandmarks()
        out.header = msg.header
        out.image_height, out.image_width = bgr.shape[:2]
        if self.hands:
            hands = self.hands.process_all(rgb)
            if hands:
                h = hands[0]
                out.hand_valid, out.handedness, out.hand_score = True, h['handedness'], h['score']
                out.hand = h['pts'].reshape(-1).tolist()
                if h['world'] is not None:
                    out.hand_world_valid, out.hand_world = True, h['world'].reshape(-1).tolist()
            if len(hands) > 1:
                h = hands[1]
                out.hand2_valid, out.handedness2, out.hand2_score = True, h['handedness'], h['score']
                out.hand2 = h['pts'].reshape(-1).tolist()
                if h['world'] is not None:
                    out.hand2_world = h['world'].reshape(-1).tolist()
        if self.pose:
            lm = self.pose.process(rgb)
            if lm is not None:
                out.pose_valid = True
                out.pose = lm.reshape(-1).tolist()
        self.pub.publish(out)
        if self.show:
            self._draw(bgr, out)

    def _draw(self, bgr, out):
        h, w = bgr.shape[:2]
        for valid, pts in ((out.hand_valid, out.hand), (out.hand2_valid, out.hand2)):
            if valid:
                for i in range(21):
                    cv2.circle(bgr, (int(pts[3 * i] * w), int(pts[3 * i + 1] * h)), 3, (0, 255, 0), -1)
        if out.pose_valid:
            for i in (0, 11, 12, 23, 24):
                cv2.circle(bgr, (int(out.pose[4 * i] * w), int(out.pose[4 * i + 1] * h)), 5,
                           (255, 0, 0), -1)
        cv2.imshow('pose_node', bgr)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = PoseNode()
    except MediaPipeMissing as e:
        print(f'[pose_node] {e}', file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(1)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001  (종료 신호 중 publish 실패)
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
