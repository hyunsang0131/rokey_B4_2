"""안전 감시 노드. 원칙: 로봇은 사람과 접촉하지 않는다.

1) 일시정지 (자동 재개)
   사용자 손/팔(/user/hand_table 웹캠 호모그래피 — 상시, /rs/hand_table RealSense 3D — 관찰 자세)이 로봇 TCP 또는 현재 작업 지점(/assist/work_point)
   pause_mm(200) 안으로 들어오면 → /safety/pause=True + 두산 MovePause
   pause_mm + resume_margin_mm 밖으로 resume_delay_sec 동안 벗어나 있으면
     auto_resume=true  → /safety/pause=False + MoveResume (자동 재개)
     auto_resume=false → 일시정지 유지 (reason='pause_hold'), 담당자가 /safety/reset 을 부르면 MoveResume  [v2.2 기본값]
     ※ 손 인식은 카메라라 가려짐을 '손이 빠짐'으로 볼 수 있음 → 실물에서 검증 전에는 auto_resume=false
2) 정지 (래치, /safety/reset 으로만 해제. 해제는 '허가'일 뿐 로봇은 움직이지 않음 — 재개는 TaskRequest task='resume')
   - /user/gesture label=='stop' && confirmed (손바닥 내밀기)
   - /voice/command 에 stop_words 포함 ("멈춰")
   - /assist/heartbeat 에서 required_nodes 가 heartbeat_timeout 초 이상 끊김
   - dance 모드에서 TCP 속도 > dance_max_speed_mms*1.3, 또는 손/팔이 dance_zone_mm 안
   - /safety/manual_stop (Bool True)
   → /safety/stop=True + 두산 MoveStop
모드 (/safety/set_mode): normal / dance
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from assist_arm_msgs.msg import Gesture, HandTable, SafetyStatus
from assist_arm_msgs.srv import SetSafetyMode
from assist_arm.common.dsr_services import DsrServices
from assist_arm.common.heartbeat import HEARTBEAT_TOPIC

MODES = ('normal', 'dance')


class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')
        p = self.declare_parameter
        p('robot_id', 'dsr01')
        p('stop_mode', 1)                          # 0 QSTOP_STO, 1 QSTOP, 2 SSTOP, 3 HOLD
        p('call_dsr_stop', True)                   # MoveStop / MovePause / MoveResume 호출 여부
        p('stop_words', ['멈춰', '정지', '스톱', 'stop'])
        p('required_nodes', ['webcam_publisher', 'pose_node', 'gesture_node', 'finger_pointer',
                             'task_manager'])
        p('heartbeat_timeout', 1.5)
        p('startup_grace', 15.0)
        # 손/팔 근접 일시정지
        p('pause_mm', 200.0)
        p('resume_margin_mm', 50.0)
        p('resume_delay_sec', 1.0)
        p('auto_resume', False)                    # v2.2: 시연·첫 실물 시험은 false (담당자 확인 후 /safety/reset)
        p('hand_height_mm', 40.0)                  # 테이블 평면 점을 이만큼 띄워서 3D 거리 계산
        p('hand_timeout_sec', 0.5)
        p('tcp_poll_hz', 10.0)
        # dance
        p('dance_zone_mm', 500.0)
        p('dance_max_speed_mms', 150.0)
        # 공유 한계값 (task_manager 가 /safety/limits 로 받아 씀)
        p('speed_free_mms', 250.0)
        p('speed_near_human_mms', 100.0)
        p('speed_final_approach_mms', 40.0)
        p('wipe_force_n', 4.0)
        p('max_contact_force_n', 15.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.dsr = DsrServices(self, robot_id=g('robot_id'))
        self.stopped, self.reason = False, ''
        self.paused, self.clear_since = False, None
        self.pause_hold = False                    # auto_resume=false 에서 손은 빠졌지만 담당자 재개를 기다리는 중
        self.mode = 'normal'
        self.t0 = time.monotonic()
        self.last_hb = {}
        self.hand_pts, self.hand_t = None, 0.0
        self.rs_pts, self.rs_t = None, 0.0        # RealSense 손 (관찰 자세에서, 실제 3D)
        self.work_pt = None
        self.tcp = self.tcp_prev = None
        self.dist = -1.0

        self.stop_pub = self.create_publisher(Bool, '/safety/stop', 10)
        self.pause_pub = self.create_publisher(Bool, '/safety/pause', 10)
        self.status_pub = self.create_publisher(SafetyStatus, '/safety/status', 10)
        self.limits_pub = self.create_publisher(String, '/safety/limits', 10)
        self.create_subscription(Gesture, '/user/gesture', self.on_gesture, 10)
        self.create_subscription(String, '/voice/command', self.on_voice, 10)
        self.create_subscription(String, HEARTBEAT_TOPIC, self.on_hb, 50)
        self.create_subscription(HandTable, '/user/hand_table', self.on_hand, 10)
        self.create_subscription(HandTable, '/rs/hand_table', self.on_rs_hand, 10)
        self.create_subscription(PointStamped, '/assist/work_point', self.on_work, 10)
        self.create_subscription(Bool, '/safety/manual_stop', lambda m: m.data and self.trigger('manual'), 10)
        self.create_service(Trigger, '/safety/reset', self.on_reset)
        self.create_service(SetSafetyMode, '/safety/set_mode', self.on_set_mode)
        self.create_timer(0.1, self.tick)
        self.create_timer(1.0 / float(g('tcp_poll_hz')), self.poll_tcp)
        self.get_logger().info('safety_monitor 시작 — 펜던트 비상정지 담당자는 항상 대기!')

    # ---- 입력 ----
    def on_gesture(self, m):
        if m.label == 'stop' and m.confirmed:
            self.trigger('gesture_stop')

    def on_voice(self, m):
        txt = m.data.lower()
        if any(w.lower() in txt for w in self.get_parameter('stop_words').value):
            self.trigger('voice_stop')

    def on_hb(self, m):
        self.last_hb[m.data] = time.monotonic()

    def on_hand(self, m):
        z = self.get_parameter('hand_height_mm').value
        self.hand_pts = np.array([[q.x, q.y, q.z + z] for q in m.points]) if m.valid and m.points else None
        self.hand_t = time.monotonic()

    def on_rs_hand(self, m):
        self.rs_pts = np.array([[q.x, q.y, q.z] for q in m.points]) if m.valid and m.points else None
        self.rs_t = time.monotonic()

    def on_work(self, m):
        self.work_pt = None if math.isnan(m.point.x) else np.array([m.point.x, m.point.y, m.point.z])

    def poll_tcp(self):
        def cb(posx):
            if posx is not None:
                self.tcp_prev = self.tcp
                self.tcp = (time.monotonic(), np.array(posx[:3]))
        self.dsr.request_posx(cb)

    def on_set_mode(self, req, res):
        if req.mode not in MODES:
            res.success, res.message = False, f'모드는 {MODES} 중 하나'
            return res
        self.mode = req.mode
        res.success, res.message = True, f'mode={req.mode}'
        self.get_logger().warn(f'안전 모드 → {req.mode}')
        return res

    # ---- 판단 ----
    def trigger(self, reason):
        if self.stopped:
            return
        self.stopped, self.reason = True, reason
        self.get_logger().error(f'!!! SAFETY STOP: {reason} (mode={self.mode})')
        if self.get_parameter('call_dsr_stop').value:
            self.dsr.move_stop(self.get_parameter('stop_mode').value)
        self.publish()

    def set_pause(self, on, why=''):
        if on == self.paused:
            return
        self.paused = on
        self.pause_hold = False
        if on:
            self.get_logger().warn(f'일시정지: {why}')
        else:
            self.get_logger().info('재개: 손/팔이 작업 지점에서 벗어남')
        if self.get_parameter('call_dsr_stop').value:
            (self.dsr.move_pause if on else self.dsr.move_resume)()

    def _speed(self):
        if self.tcp is None or self.tcp_prev is None:
            return None
        dt = self.tcp[0] - self.tcp_prev[0]
        if dt < 1e-3 or dt > 1.0:
            return None
        return float(np.linalg.norm(self.tcp[1] - self.tcp_prev[1]) / dt)

    def tick(self):
        now = time.monotonic()
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        if now - self.t0 > g('startup_grace'):
            for n in g('required_nodes'):
                t = self.last_hb.get(n)
                if t is None or now - t > g('heartbeat_timeout'):
                    self.trigger(f'heartbeat:{n}')
                    break
        # 손/팔 ↔ TCP·작업 지점 거리
        self.dist = -1.0
        srcs = [p for p, t in ((self.hand_pts, self.hand_t), (self.rs_pts, self.rs_t))
                if p is not None and len(p) and now - t < g('hand_timeout_sec')]
        hands = np.vstack(srcs) if srcs else None
        targets = [t for t in (None if self.tcp is None else self.tcp[1], self.work_pt) if t is not None]
        if hands is not None and targets:
            self.dist = float(min(np.linalg.norm(hands - t, axis=1).min() for t in targets))
        if self.dist >= 0 and self.dist < g('pause_mm'):
            self.clear_since = None
            self.set_pause(True, f'손/팔 {self.dist:.0f}mm')
        elif self.paused:
            if self.dist < 0 or self.dist > g('pause_mm') + g('resume_margin_mm'):
                self.clear_since = self.clear_since or now
                if now - self.clear_since >= g('resume_delay_sec'):
                    if g('auto_resume'):
                        self.set_pause(False)
                    elif not self.pause_hold:
                        self.pause_hold = True
                        self.get_logger().warn('손/팔이 벗어남 — 담당자 확인 후 /safety/reset 으로 재개 (auto_resume=false)')
            else:
                self.clear_since = None
                self.pause_hold = False
        if self.mode == 'dance':
            v = self._speed()
            if v is not None and v > g('dance_max_speed_mms') * 1.3:
                self.trigger(f'dance_speed({v:.0f}mm/s)')
            if 0 <= self.dist < g('dance_zone_mm'):
                self.trigger(f'dance_zone({self.dist:.0f}mm)')
        self.publish()

    def publish(self):
        self.stop_pub.publish(Bool(data=self.stopped))
        self.pause_pub.publish(Bool(data=self.paused))
        s = SafetyStatus()
        s.header.stamp = self.get_clock().now().to_msg()
        s.stop, s.reason, s.mode, s.paused = self.stopped, self.reason, self.mode, self.paused
        if not self.stopped and self.pause_hold:
            s.reason = 'pause_hold'                # 담당자 재개(/safety/reset) 대기
        s.hand_distance_mm = float(self.dist)
        self.status_pub.publish(s)
        g = lambda n: float(self.get_parameter(n).value)  # noqa: E731
        lim = {k: g(k) for k in ('speed_free_mms', 'speed_near_human_mms', 'speed_final_approach_mms',
                                 'wipe_force_n', 'max_contact_force_n', 'pause_mm',
                                 'dance_zone_mm', 'dance_max_speed_mms')}
        lim['mode'] = self.mode
        self.limits_pub.publish(String(data=json.dumps(lim)))

    def on_reset(self, req, res):
        """담당자 버튼: ① 정지 래치 해제(허가만, 로봇은 안 움직임 → 재개는 task='resume') ② pause_hold 이면 MoveResume"""
        if self.stopped:
            self.stopped, self.reason, self.mode = False, '', 'normal'
            self.t0 = time.monotonic()
            res.success, res.message = True, ('정지 해제 → normal 모드. 로봇은 움직이지 않음 — '
                                              '상태 확인 뒤 resume 요청으로 재개 (로봇 상태는 펜던트에서 확인)')
        elif self.paused and self.pause_hold:
            self.set_pause(False)
            res.success, res.message = True, '일시정지 해제 (담당자 확인) → MoveResume'
        elif self.paused:
            res.success, res.message = False, f'손/팔이 아직 작업 지점 가까이 있음 ({self.dist:.0f}mm) — 재개 안 함'
        else:
            self.mode = 'normal'
            res.success, res.message = True, '정지·일시정지 상태 아님 (normal 모드)'
        self.get_logger().warn(res.message)
        self.publish()
        return res


def main(args=None):
    rclpy.init(args=args)
    node = SafetyMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
