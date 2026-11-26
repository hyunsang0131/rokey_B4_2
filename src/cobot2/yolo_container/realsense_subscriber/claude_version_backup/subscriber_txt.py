"""RealSense RGB/Depth 토픽을 받아 정보를 글자로 출력하는 노드."""
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image

COLOR_TOPIC = '/camera/color/image_raw'
DEPTH_TOPIC = '/camera/aligned_depth_to_color/image_raw'


class RealsenseTxtSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')
        self.create_subscription(Image, COLOR_TOPIC, self.color_cb, 10)
        self.create_subscription(Image, DEPTH_TOPIC, self.depth_cb, 10)
        self.color_count = 0
        self.depth_count = 0
        self.get_logger().info('Realsense Text Subscriber Started.')
        self.get_logger().info('Waiting for RGB and Depth images...')

    def color_cb(self, msg):
        self.color_count += 1
        if self.color_count % 30 == 1:  # 30fps 기준 1초에 한 번만 출력
            self.get_logger().info(
                f'[RGB]   {msg.width}x{msg.height} encoding={msg.encoding} '
                f'frame_id={msg.header.frame_id} 받은 개수={self.color_count}')

    def depth_cb(self, msg):
        self.depth_count += 1
        if self.depth_count % 30 == 1:
            # 16UC1: 픽셀 하나가 2바이트(mm 단위). 화면 가운데 거리 계산
            cx, cy = msg.width // 2, msg.height // 2
            i = cy * msg.step + cx * 2
            center_mm = int.from_bytes(msg.data[i:i + 2], 'little')
            self.get_logger().info(
                f'[Depth] {msg.width}x{msg.height} encoding={msg.encoding} '
                f'가운데 거리={center_mm} mm 받은 개수={self.depth_count}')


def main(args=None):
    rclpy.init(args=args)
    node = RealsenseTxtSubscriber()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
