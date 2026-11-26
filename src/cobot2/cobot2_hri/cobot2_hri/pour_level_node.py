"""pour_level_node (호연) — 컵 수위 → /cobot2/pour/level (CupLevel ratio 0~1, confidence), 10 Hz 이상.  [v1.1]

카메라: 옆에서 컵을 보는 고정 카메라 (source=webcam, device) 기본 / source=topic 이면 ROS 이미지 토픽 (RealSense 등)
설정: config/pour_level.yaml (roi, y_bottom, y_full) — tools/pour_level_roi.py 로 클릭해서 만들기
서비스: /cobot2/pour/capture_empty (std_srvs/Trigger) — 빈 컵일 때 기준 사진 저장 (따르기 전에 부르면 투명 컵도 더 정확)
판단이 어려우면 ratio 는 직전 값을 유지하고 confidence 를 낮춤 → pour 모듈이 시간·각도 기준으로 대체
"""
import os

import cv2
import rclpy
import yaml
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger

from cobot2_interfaces.msg import CupLevel
from cobot2_hri.common.level import LevelEstimator


class PourLevelNode(Node):
    def __init__(self):
        super().__init__('pour_level_node')
        self.declare_parameter('config_file', '')
        self.declare_parameter('rate_hz', 15.0)
        self.declare_parameter('show_debug', False)
        self.declare_parameter('source', '')         # 비우면 config 값
        self.declare_parameter('device', -1)
        path = self.get_parameter('config_file').value
        if not path:
            from ament_index_python.packages import get_package_share_directory
            path = os.path.join(get_package_share_directory('cobot2_hri'), 'config', 'pour_level.yaml')
        c = yaml.safe_load(open(os.path.expanduser(path)))
        src = self.get_parameter('source').value or c.get('source', 'webcam')
        dev = self.get_parameter('device').value
        dev = int(c.get('device', 0)) if dev < 0 else int(dev)
        self.est = LevelEstimator(c['roi'], c['y_bottom'], c['y_full'])
        self.frame, self.cap = None, None
        if src == 'webcam':
            self.cap = cv2.VideoCapture(dev)
            if not self.cap.isOpened():
                self.get_logger().error(f'카메라 {dev} 를 열 수 없어요 (v4l2-ctl --list-devices)')
        else:
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image
            br = CvBridge()
            self.create_subscription(Image, c.get('image_topic', '/camera/color/image_raw'),
                                     lambda m: setattr(self, 'frame', br.imgmsg_to_cv2(m, 'bgr8')), 5)
        self.pub = self.create_publisher(CupLevel, '/cobot2/pour/level', 10)
        self.create_service(Trigger, '/cobot2/pour/capture_empty', self.on_empty)
        self.create_timer(1.0 / float(self.get_parameter('rate_hz').value), self.tick)
        self.get_logger().info(f'pour_level_node: source={src}, roi={c["roi"]}, config={path}')

    def _grab(self):
        if self.cap is not None:
            ok, f = self.cap.read()
            return f if ok else None
        return self.frame

    def on_empty(self, req, res):
        f = self._grab()
        if f is None:
            res.success, res.message = False, '영상 없음'
            return res
        self.est.set_empty(f)
        res.success, res.message = True, '빈 컵 기준 저장'
        return res

    def tick(self):
        f = self._grab()
        m = CupLevel()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'pour_camera'
        if f is None:
            m.ratio, m.confidence = float(self.est.ratio), 0.0
        else:
            m.ratio, m.confidence = self.est.update(f)
            if self.get_parameter('show_debug').value:
                cv2.imshow('pour_level', self.est.draw(f.copy()))
                cv2.waitKey(1)
        self.pub.publish(m)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PourLevelNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
