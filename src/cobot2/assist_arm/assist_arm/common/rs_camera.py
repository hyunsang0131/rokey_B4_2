"""손목 RealSense 공용 구독 + 픽셀→베이스 변환 (spill_node 가 사용).

토픽과 수식은 object_detection/realsense.py(ImgNode), detection.py 와 동일.
로봇 자세는 dsr_controller2 의 get_current_posx 서비스를 비동기로 받아 쓴다.
"""
import threading

from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image

from assist_arm.common.dsr_services import DsrServices
from assist_arm.common.geometry import (load_gripper2cam, median_depth, pixel_ray_to_plane, pixel_to_camera,
                                        transform_to_base)


class WristCamera:
    def __init__(self, node, callback_group=None, robot_id='dsr01', gripper2cam_path='',
                 color_topic='/camera/color/image_raw',
                 depth_topic='/camera/aligned_depth_to_color/image_raw',
                 info_topic='/camera/color/camera_info', min_depth=150.0, max_depth=1500.0):
        self.node = node
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.color = self.depth = self.intr = None
        self.min_depth, self.max_depth = min_depth, max_depth
        self.T_g2c = load_gripper2cam(gripper2cam_path)
        self.dsr = DsrServices(node, robot_id=robot_id)
        kw = {'callback_group': callback_group} if callback_group else {}
        node.create_subscription(Image, color_topic, self._on_color, 5, **kw)
        node.create_subscription(Image, depth_topic, self._on_depth, 5, **kw)
        node.create_subscription(CameraInfo, info_topic, self._on_info, 5, **kw)

    def _on_info(self, msg):
        self.intr = {"fx": msg.k[0], "fy": msg.k[4], "ppx": msg.k[2], "ppy": msg.k[5]}

    def _on_color(self, msg):
        with self.lock:
            self.color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def _on_depth(self, msg):
        with self.lock:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def snapshot(self):
        """(bgr, depth) 복사본 또는 (None, None)"""
        with self.lock:
            if self.color is None or self.depth is None or self.intr is None:
                return None, None
            return self.color.copy(), self.depth.copy()

    def pixel_to_cam(self, u, v, depth, win=5):
        z = median_depth(depth, int(u), int(v), win)
        if z is None or not (self.min_depth < z < self.max_depth):
            return None
        return pixel_to_camera(u, v, z, self.intr)

    def to_base_async(self, cam_points, callback):
        """cam_points: list[(x,y,z) | None] → callback(list[np.array | None] | None)"""
        def on_posx(posx):
            if posx is None:
                callback(None)
                return
            callback([None if c is None else transform_to_base(c, self.T_g2c, posx)
                      for c in cam_points])
        self.dsr.request_posx(on_posx)

    def pixels_to_base_async(self, uvs, depth, callback, plane_z=None, win=5, wide_win=15):
        """여러 픽셀 → 베이스 3D. 깊이 순서: (1) win 중앙값 (2) wide_win 중앙값 (3) plane_z 평면 교점.
        callback(list[(xyz | None, source)] | None)  source: depth / depth_wide / plane"""
        cams = []
        for u, v in uvs:
            c = self.pixel_to_cam(u, v, depth, win)
            src = 'depth'
            if c is None:
                c, src = self.pixel_to_cam(u, v, depth, wide_win), 'depth_wide'
            cams.append((c, src, u, v))

        def on_posx(posx):
            if posx is None:
                callback(None)
                return
            out = []
            for c, src, u, v in cams:
                if c is not None:
                    out.append((transform_to_base(c, self.T_g2c, posx), src))
                elif plane_z is not None:
                    p = pixel_ray_to_plane(u, v, self.intr, self.T_g2c, posx, plane_z)
                    out.append((p, 'plane') if p is not None else (None, 'none'))
                else:
                    out.append((None, 'none'))
            callback(out)
        self.dsr.request_posx(on_posx)
