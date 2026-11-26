"""finger_pointer: 검지 끝 → 로봇 베이스 테이블 좌표, 1초 머물면 '콕(poke)'.

소스 (pointer_source: auto | realsense | webcam, 기본 auto)
  realsense : 관찰 자세에서 rs_hand_node 의 /rs/hand_table (RealSense 깊이 3D 검지 끝) — 주 인식
  webcam    : 웹캠 손 랜드마크 → 테이블 평면 호모그래피 (아래) — 보조
  auto      : 관찰 중(/assist/observing) 이고 RealSense 손이 rs_timeout_sec 안에 보이면 realsense, 아니면 webcam
/user/hand_table 은 항상 웹캠 기준으로 발행 (safety_monitor 상시 근접 감시용).

[웹캠 경로] 웹캠 손 랜드마크 → 테이블 평면 호모그래피 → 로봇 베이스 테이블 좌표.

입력  /user/landmarks (pose_node, 웹캠 MediaPipe — 여기서 MediaPipe 를 다시 돌리지 않음)
캘리브  calibrate_pointer 로 만든 yaml (H: 이미지 px → 테이블 x,y mm, table_z_mm)
출력
  /user/fingertip   geometry_msgs/PointStamped  검지 끝(landmark 8) 테이블 좌표 (frame 'base', mm)
  /user/hand_table  assist_arm_msgs/HandTable    손 21점 + 팔(Pose 13~16) 테이블 좌표 → safety_monitor 근접 일시정지
  /user/poke        geometry_msgs/PointStamped  검지를 편 채(point) 끝이 dwell_radius_mm 안에 dwell_sec 머물면 1회
                                                 (손가락이 rearm_mm 이상 움직여야 다음 poke)
주의: 호모그래피는 '테이블 평면 위' 점에만 정확. 손가락이 테이블에 닿거나 가까울 때 쓰는 것이 전제.
      팔꿈치처럼 테이블에서 떠 있는 점은 대략값 (근접 감시용으로만 사용, 여유 margin 을 크게).
"""
import os
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Point, PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool

from assist_arm_msgs.msg import HandTable, UserLandmarks
from assist_arm.common.features import INDEX_TIP, PALM_IDS, is_pointing
from assist_arm.common.heartbeat import attach_heartbeat

ARM_IDS = (13, 14, 15, 16)   # 왼/오른 팔꿈치, 왼/오른 손목 (MediaPipe Pose)


def load_calibration(path):
    with open(os.path.expanduser(path)) as f:
        c = yaml.safe_load(f)
    return np.array(c['H'], dtype=np.float64).reshape(3, 3), float(c.get('table_z_mm', 0.0)), c


def apply_h(H, uv):
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    q = np.hstack([uv, np.ones((len(uv), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


class FingerPointer(Node):
    def __init__(self):
        super().__init__('finger_pointer')
        p = self.declare_parameter
        p('calibration_file', '~/data/assist_arm/pointer_calibration.yaml')
        p('dwell_sec', 1.0)
        p('dwell_radius_mm', 15.0)
        p('rearm_mm', 40.0)
        p('require_pointing', True)        # 검지만 편 손 모양일 때만 poke
        p('min_arm_visibility', 0.5)
        p('pointer_source', 'auto')        # auto / realsense / webcam
        p('rs_timeout_sec', 0.5)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.H = None
        path = os.path.expanduser(g('calibration_file'))
        if os.path.exists(path):
            self.H, self.table_z, _ = load_calibration(path)
            self.get_logger().info(f'포인터 캘리브레이션 로드: {path}')
        else:
            self.get_logger().error(f'캘리브레이션 없음: {path} → python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/calibrate_pointer.py 먼저')
            self.table_z = 0.0
        self.hist = {'webcam': [], 'realsense': []}      # 소스별 [(t, 좌표)]
        self.armed = True
        self.last_poke = None
        self.observing = False
        self.rs_t = -1e9
        self.tip_pub = self.create_publisher(PointStamped, '/user/fingertip', 10)
        self.hand_pub = self.create_publisher(HandTable, '/user/hand_table', 10)
        self.poke_pub = self.create_publisher(PointStamped, '/user/poke', 10)
        self.create_subscription(UserLandmarks, '/user/landmarks', self.on_lm, 10)
        self.create_subscription(HandTable, '/rs/hand_table', self.on_rs, 10)
        self.create_subscription(Bool, '/assist/observing', lambda m: setattr(self, 'observing', m.data), 10)
        attach_heartbeat(self)

    def active_source(self):
        src = self.get_parameter('pointer_source').value
        if src != 'auto':
            return src
        rs_fresh = time.monotonic() - self.rs_t < self.get_parameter('rs_timeout_sec').value
        return 'realsense' if (self.observing and rs_fresh) else 'webcam'

    def on_rs(self, m):
        """RealSense 3D 검지 끝 (베이스 mm). 관찰 자세에서 주 콕 판정."""
        if not m.valid:
            self.hist['realsense'] = []
            return
        self.rs_t = time.monotonic()
        if self.active_source() != 'realsense':
            return
        tip = np.array([m.fingertip.x, m.fingertip.y, m.fingertip.z])
        tm = PointStamped()
        tm.header = m.header
        tm.point = m.fingertip
        self.tip_pub.publish(tm)
        self._dwell(tip, m.gesture_hint.startswith('pointing'), m.header, 'realsense')

    def _pt(self, xy):
        return Point(x=float(xy[0]), y=float(xy[1]), z=self.table_z)

    def on_lm(self, m):
        if self.H is None:
            return
        out = HandTable()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base'
        W, Hh = m.image_width, m.image_height
        pts_uv = []
        hand = None
        if m.hand_valid:
            hand = np.asarray(m.hand, np.float32).reshape(21, 3)
            pts_uv += (hand[:, :2] * [W, Hh]).tolist()
        if m.hand2_valid:      # 두 번째 손도 근접 감시 점에 포함 (poke 는 첫 번째 손만)
            pts_uv += (np.asarray(m.hand2, np.float32).reshape(21, 3)[:, :2] * [W, Hh]).tolist()
        if m.pose_valid:
            pose = np.asarray(m.pose, np.float32).reshape(33, 4)
            pts_uv += [(pose[i, :2] * [W, Hh]).tolist() for i in ARM_IDS
                       if pose[i, 3] >= self.get_parameter('min_arm_visibility').value]
        if not pts_uv:
            self.hand_pub.publish(out)
            self.hist['webcam'] = []
            return
        xy = apply_h(self.H, pts_uv)
        out.valid = True
        out.points = [self._pt(q) for q in xy]
        if hand is None:
            self.hand_pub.publish(out)
            return
        tip = xy[INDEX_TIP]
        out.fingertip = self._pt(tip)
        out.palm = self._pt(xy[PALM_IDS].mean(axis=0))
        pointing = is_pointing(hand)
        out.gesture_hint = 'pointing' if pointing else 'other'
        self.hand_pub.publish(out)          # safety 용: 항상 웹캠 기준
        if self.active_source() != 'webcam':
            self.hist['webcam'] = []
            return
        tm = PointStamped()
        tm.header = out.header
        tm.point = out.fingertip
        self.tip_pub.publish(tm)
        self._dwell(np.array([tip[0], tip[1], self.table_z]), pointing, out.header, 'webcam')

    def _dwell(self, tip, pointing, header, source):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        now = time.monotonic()
        if self.last_poke is not None and np.linalg.norm(tip[:2] - self.last_poke[:2]) > g('rearm_mm'):
            self.armed = True
        hist = self.hist[source]
        if g('require_pointing') and not pointing:
            hist.clear()
            return
        hist.append((now, tip))
        hist[:] = [(t, q) for t, q in hist if now - t <= g('dwell_sec') + 0.5]
        if not self.armed or now - hist[0][0] < g('dwell_sec'):
            return
        win = np.array([q for t, q in hist if now - t <= g('dwell_sec')])
        if np.linalg.norm(win - win.mean(axis=0), axis=1).max() <= g('dwell_radius_mm'):
            c = win.mean(axis=0)
            pm = PointStamped()
            pm.header = header
            pm.header.frame_id = 'base'
            pm.point = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            self.poke_pub.publish(pm)
            self.get_logger().info(f'poke ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) [{source}]')
            self.armed, self.last_poke = False, c
            hist.clear()


def main(args=None):
    rclpy.init(args=args)
    node = FingerPointer()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
