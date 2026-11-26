"""yolo-detection 컨테이너 안에서 도는 3D 위치 서비스 (assist_arm 용).

기존 object_detection 은 공구 모델(yolov8n_tools_0122.pt)이 고정이라 cup/bowl 등이 없다.
기존 파일을 수정하지 않기 위해 같은 코드를 재사용하면서 모델 경로만 파라미터로 받는 노드를 따로 둔다.
  - 영상/깊이/내부파라미터: object_detection.realsense.ImgNode (그대로)
  - 탐지 집계: object_detection.yolo.YoloModel 의 get_frames/_aggregate_detections (그대로)
  - 3D 계산: object_detection.detection.ObjectDetectionNode 의 _get_depth/_pixel_to_camera_coords (그대로 상속)
  - 서비스 타입: od_msg/SrvDepthPosition (그대로), 이름 /assist/get_3d_position
    (컨테이너는 yolo_container 만 마운트하므로 assist_arm_msgs 를 쓸 수 없어 기존 타입에 규약을 얹음)

요청 규약 (request.target)
  "cup"                → 기존과 동일: 가장 확실한 1개 [x, y, z] (카메라 mm), 없으면 [0,0,0]
  "all:bowl,plate,cup" → 해당 클래스 전부, 1개당 6개 값을 이어 붙임:
                         [x, y, z, score, class_index(요청 목록 순서), fill]
                         fill = 그릇 안쪽이 채워진 정도 0~1 (임시 규칙), 계산 불가면 -1
별칭: aliases 파라미터 "요청이름:모델클래스" (예: "rice_bowl:bowl", "plate:bowl") — 커스텀 모델 전까지 COCO 로 대체
model_path 기본 yolov8n.pt (COCO: cup 41, fork 42, knife 43, spoon 44, bowl 45, bottle 39) — 첫 실행 시 다운로드.
  (이미 받아 둔 파일: /yolo_train/Fruits/yolov8n.pt — -p model_path:=/yolo_train/Fruits/yolov8n.pt 로 다운로드 없이)
식탁 커스텀 모델 (assist_arm/tools/colab_yolo_train.ipynb 로 학습): 호스트 ~/corecode/yolo_train/assist_table/best.pt
  → -p model_path:=/yolo_train/assist_table/best.pt  (클래스: rice_bowl side_dish cup water_bottle spoon chopsticks tissue cloth soup_bowl plate fork — 순서는 config/yolo_classes.yaml)
  모델에 그 이름이 있으면 별칭보다 먼저 씀 → aliases 는 그대로 둬도 됨.
"""
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ultralytics import YOLO

from od_msg.srv import SrvDepthPosition
from object_detection.detection import ObjectDetectionNode
from object_detection.yolo import YoloModel


class ConfigurableYolo(YoloModel):
    def __init__(self, model_path):  # YoloModel.__init__ (고정 경로) 대신
        self.model = YOLO(model_path)
        self.reversed_class_dict = {v: int(k) for k, v in self.model.names.items()}


def fill_ratio(bgr, depth, box):
    """임시 규칙: 그릇 안쪽(가운데 50%)이 테두리(바깥 고리)와 색이 다르거나 더 솟아 있으면 '음식 있음'.
    반환 0(빈 그릇)~1(가득). depth 는 mm(카메라 기준, 작을수록 가까움 = 위에서 볼 때 높음)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return -1.0
    bw, bh = x2 - x1, y2 - y1
    ix1, iy1, ix2, iy2 = x1 + bw // 4, y1 + bh // 4, x2 - bw // 4, y2 - bh // 4
    lab = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = np.ones(lab.shape[:2], bool)
    ring[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1] = False
    inner = lab[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1].reshape(-1, 3)
    color_score = np.clip((np.linalg.norm(inner.mean(0) - lab[ring].mean(0)) - 10) / 30, 0, 1)
    texture_score = np.clip((inner.std(0).mean() - 6) / 20, 0, 1)
    d = depth[y1:y2, x1:x2].astype(np.float32)
    di, dr = d[iy1 - y1:iy2 - y1, ix1 - x1:ix2 - x1], d[ring]
    depth_score = 0.0
    if (di > 0).sum() > 20 and (dr > 0).sum() > 20:
        # 빈 그릇은 바닥이 테두리보다 멀다(깊다). 음식이 차면 차이가 줄어듦
        gap = np.median(di[di > 0]) - np.median(dr[dr > 0])
        depth_score = float(np.clip(1 - gap / 40.0, 0, 1))
    return float(np.clip(0.4 * color_score + 0.3 * texture_score + 0.3 * depth_score, 0, 1))


class AssistDetectionNode(ObjectDetectionNode):
    def __init__(self):
        Node.__init__(self, 'assist_detection_node')
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('service_name', '/assist/get_3d_position')
        self.declare_parameter('aliases', ['rice_bowl:bowl', 'soup_bowl:bowl', 'side_dish:bowl', 'side_dish_bowl:bowl',
                                           'plate:bowl', 'water_bottle:bottle'])
        self.declare_parameter('min_score', 0.4)
        from object_detection.realsense import ImgNode
        self.img_node = ImgNode()
        self.model = ConfigurableYolo(self.get_parameter('model_path').value)
        self.alias = dict(a.split(':', 1) for a in self.get_parameter('aliases').value if ':' in a)
        self.get_logger().info(f'classes: {list(self.model.reversed_class_dict)} aliases: {self.alias}')
        self.intrinsics = self._wait_for_valid_data(self.img_node.get_camera_intrinsic, 'camera intrinsics')
        self.create_service(SrvDepthPosition, self.get_parameter('service_name').value, self.handle)
        self.get_logger().info('AssistDetectionNode initialized.')

    def _cls(self, name):
        name = name.strip()
        if name in self.model.reversed_class_dict:
            return self.model.reversed_class_dict[name]
        return self.model.reversed_class_dict.get(self.alias.get(name, ''), None)

    def handle(self, request, response):
        t = request.target
        if not t.startswith('all:'):
            original = t
            request.target = t if t in self.model.reversed_class_dict else self.alias.get(t, t)
            self.get_logger().info(f'single: {original} → {request.target}')
            return self.handle_get_depth(request, response)
        names = [n for n in t[4:].split(',') if n.strip()]
        ids = [self._cls(n) for n in names]
        frames = self.model.get_frames(self.img_node)
        if not frames:
            response.depth_position = []
            return response
        dets = self.model._aggregate_detections(self.model.model(frames, verbose=False),
                                                confidence_threshold=self.get_parameter('min_score').value)
        bgr = frames[-1]
        depth = self._wait_for_valid_data(self.img_node.get_depth_frame, 'depth frame')
        out = []
        for d in dets:
            for k, cid in enumerate(ids):
                if cid is None or d['label'] != cid:
                    continue
                b = d['box']
                cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
                cz = self._get_depth(cx, cy)
                if cz is None:
                    continue
                x, y, z = self._pixel_to_camera_coords(cx, cy, cz)
                out += [x, y, z, d['score'], float(k), fill_ratio(bgr, depth, b)]
        self.get_logger().info(f'all: {names} → {len(out) // 6}개')
        response.depth_position = [float(v) for v in out]
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AssistDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
