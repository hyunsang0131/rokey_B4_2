"""rs_hand_node: 손목 RealSense 컬러 영상에 MediaPipe Hands → 손 21관절 (주 인식, 관찰 자세에서).

관찰 자세(/assist/observing=True, 로봇 정지)일 때만 돈다 (require_observing). 손목 카메라라 로봇이 움직이면 좌표가 틀어짐.
출력
  /rs/landmarks  assist_arm_msgs/UserLandmarks  웹캠 pose_node 와 같은 형식 (21점 정규화 + world landmarks, 양손)
                                                → gesture_node 가 같은 학습 모델로 제스처 인식
  /rs/hand_table assist_arm_msgs/HandTable      손 점들의 로봇 베이스 3D (mm): fingertip(검지 끝), palm, points(양손 21점씩)
                                                → finger_pointer 콕 판정, safety_monitor 근접 감시
3D 변환: aligned depth (중앙값 5px → 15px) → 카메라 좌표 → T_gripper2camera + 현재 posx → 베이스.
         깊이가 없으면 (너무 가까움·구멍) 카메라 광선과 테이블 평면(table_z_mm) 교점으로 대체.
재사용: object_detection 과 같은 RealSense 토픽/수식, robot_control 의 T_gripper2camera, dsr get_current_posx.
"""
import sys

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool

from assist_arm_msgs.msg import HandTable, UserLandmarks
from assist_arm.common.features import INDEX_TIP, PALM_IDS, is_pointing
from assist_arm.common.heartbeat import attach_heartbeat
from assist_arm.common.mp_loader import HandTracker, MediaPipeMissing
from assist_arm.common.rs_camera import WristCamera


class RsHandNode(Node):
    def __init__(self):
        super().__init__('rs_hand_node')
        p = self.declare_parameter
        p('require_observing', True)
        p('rate_hz', 15.0)
        p('robot_id', 'dsr01')
        p('gripper2cam_path', '')
        p('table_z_mm', 0.0)
        p('min_depth_mm', 100.0)     # D435i 최소 깊이 ~10~20cm
        p('max_depth_mm', 1500.0)
        p('num_hands', 2)
        p('min_hand_detection_confidence', 0.5)
        p('min_hand_presence_confidence', 0.5)
        p('min_hand_tracking_confidence', 0.5)
        p('hand_task_model', '~/data/assist_arm/models/hand_landmarker.task')
        p('dry_posx', [0.0])          # dry_run 테스트용 고정 자세 (원소 6개일 때만 사용)
        p('show_debug', False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        cbg = ReentrantCallbackGroup()
        self.cam = WristCamera(self, cbg, g('robot_id'), g('gripper2cam_path'),
                               min_depth=g('min_depth_mm'), max_depth=g('max_depth_mm'))
        if len(g('dry_posx')) == 6:
            fixed = [float(v) for v in g('dry_posx')]
            self.cam.dsr.request_posx = lambda cb, ref=0: cb(fixed)
        self.hands = HandTracker(max_hands=int(g('num_hands')), min_det=float(g('min_hand_detection_confidence')),
                                 min_presence=float(g('min_hand_presence_confidence')),
                                 min_track=float(g('min_hand_tracking_confidence')),
                                 task_model_path=g('hand_task_model'))
        self.observing = False
        self.lm_pub = self.create_publisher(UserLandmarks, '/rs/landmarks', 10)
        self.table_pub = self.create_publisher(HandTable, '/rs/hand_table', 10)
        self.create_subscription(Bool, '/assist/observing', lambda m: setattr(self, 'observing', m.data), 10,
                                 callback_group=cbg)
        self.create_timer(1.0 / float(g('rate_hz')), self.tick, callback_group=cbg)
        attach_heartbeat(self)

    def tick(self):
        if self.get_parameter('require_observing').value and not self.observing:
            return
        bgr, depth = self.cam.snapshot()
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        hands = self.hands.process_all(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        stamp = self.get_clock().now().to_msg()
        lm = UserLandmarks()
        lm.header.stamp, lm.header.frame_id = stamp, 'realsense'
        lm.image_width, lm.image_height = w, h
        if hands:
            h0 = hands[0]
            lm.hand_valid, lm.handedness, lm.hand_score = True, h0['handedness'], h0['score']
            lm.hand = h0['pts'].reshape(-1).tolist()
            if h0['world'] is not None:
                lm.hand_world_valid, lm.hand_world = True, h0['world'].reshape(-1).tolist()
        if len(hands) > 1:
            h1 = hands[1]
            lm.hand2_valid, lm.handedness2, lm.hand2_score = True, h1['handedness'], h1['score']
            lm.hand2 = h1['pts'].reshape(-1).tolist()
            if h1['world'] is not None:
                lm.hand2_world = h1['world'].reshape(-1).tolist()
        self.lm_pub.publish(lm)
        if not hands:
            self.table_pub.publish(HandTable(valid=False))
            return
        uvs = [tuple(q) for hd in hands[:2] for q in hd['pts'][:, :2] * [w, h]]
        pointing = is_pointing(hands[0]['pts'])
        n0 = 21

        def done(res):
            if res is None:
                self.get_logger().warn('로봇 자세 없음 → 3D 변환 생략', throttle_duration_sec=3.0)
                return
            pts = [r[0] for r in res]
            ok = [p is not None for p in pts]
            if not ok[INDEX_TIP]:
                return
            m = HandTable()
            m.header.stamp, m.header.frame_id = stamp, 'base'
            m.valid = True
            m.fingertip = Point(**dict(zip('xyz', map(float, pts[INDEX_TIP]))))
            palm = [pts[i] for i in PALM_IDS if pts[i] is not None]
            if palm:
                m.palm = Point(**dict(zip('xyz', map(float, np.mean(palm, axis=0)))))
            m.points = [Point(**dict(zip('xyz', map(float, p)))) for p in pts if p is not None]
            srcs = {r[1] for r in res[:n0]}
            m.gesture_hint = ('pointing' if pointing else 'other') + ('|plane' if 'plane' in srcs else '')
            self.table_pub.publish(m)
        self.cam.pixels_to_base_async(uvs, depth, done, plane_z=float(self.get_parameter('table_z_mm').value))

        if self.get_parameter('show_debug').value:
            for hd in hands:
                for x, y, _ in hd['pts']:
                    cv2.circle(bgr, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
            cv2.imshow('rs_hand_node', bgr)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = RsHandNode()
    except MediaPipeMissing as e:
        print(f'[rs_hand_node] {e}', file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(1)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
