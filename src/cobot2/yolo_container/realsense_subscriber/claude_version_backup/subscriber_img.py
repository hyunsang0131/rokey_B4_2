"""RealSense RGB/Depth 토픽을 받아 OpenCV 창 두 개로 보여주는 노드 (q 키로 종료)."""
import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'


class RealsenseImgSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')
        self.bridge = CvBridge()
        self.color = None
        self.depth = None
        self.create_subscription(Image, COLOR_TOPIC, self.color_cb, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, 10)
        self.get_logger().info('Realsense Image Subscriber Started.')
        self.get_logger().info('Waiting for RGB and Depth images...')

    def color_cb(self, msg):
        self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_cb(self, msg):
        self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseImgSubscriber()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.03)
            if node.color is not None:
                cv2.imshow('RGB', node.color)
            if node.depth is not None:
                # mm 단위 depth(0~3m)를 보기 좋은 컬러맵으로 변환
                d = np.clip(node.depth, 0, 3000).astype(np.float32) / 3000.0 * 255
                cv2.imshow('Depth', cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_JET))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
