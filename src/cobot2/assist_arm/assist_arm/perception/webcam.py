"""웹캠 → /webcam/image_raw. pose_node 와 spill_node 가 같은 웹캠을 공유하기 위한 단일 열기 지점."""
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from assist_arm.common.heartbeat import attach_heartbeat


class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        self.declare_parameter('device', 0)          # /dev/videoN 의 N 또는 경로 문자열
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 480)
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('topic', '/webcam/image_raw')
        self.declare_parameter('flip', True)          # 거울 모드(사용자가 보기 편하게)
        dev = self.get_parameter('device').value
        self.cap = cv2.VideoCapture(dev)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.get_parameter('width').value)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.get_parameter('height').value)
        if not self.cap.isOpened():
            self.get_logger().error(
                f'웹캠 {dev} 를 열 수 없습니다. RealSense 도 /dev/video* 를 쓰니 '
                'v4l2-ctl --list-devices 로 웹캠 번호를 확인해 device 파라미터를 바꾸세요.')
        self.flip = self.get_parameter('flip').value
        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.get_parameter('topic').value, 5)
        self.create_timer(1.0 / float(self.get_parameter('fps').value), self.tick)
        attach_heartbeat(self)

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self._assist_hb_alive = False
            return
        self._assist_hb_alive = True
        if self.flip:
            frame = cv2.flip(frame, 1)
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'webcam'
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
