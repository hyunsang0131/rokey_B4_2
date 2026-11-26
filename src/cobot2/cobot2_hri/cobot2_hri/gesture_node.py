"""gesture_node (호연) — 고정 웹캠 MediaPipe → 손동작 명령, 자리 상태, 정지.  [팀 설계 v1.1]

출력
  /cobot2/user/command   UserCommand (source=gesture)   손동작 명령 (확정 + 쿨다운)
      push(손등 밀기)      → clear_table  [손이 있는 자리 / 가장 가까운 자리 / 없으면 전체]
      cup_grasp(빈손 C자)  → pour         [가장 가까운 자리], pour_ratio = default_pour_ratio
      point(검지 1초 머묾) → set_table    [가리킨 자리]
      stop(손바닥)         → stop  (+ /cobot2/safety/stop = true)
  /cobot2/hri/seat_status SeatStatus   자리별 EMPTY / EATING(최근 손 활동) / FINISHED(엄지척), 바뀔 때 + 1 Hz
  /cobot2/safety/stop     std_msgs/Bool  true 만 보냄 (false = 재개 허용은 task_manager 의 resume 몫)
      · 정지 손동작
      · /cobot2/robot/state 가 MOVING 인데 손·팔이 로봇 작업 영역(seats.yaml robot_zone) + proximity_m 안
입력
  카메라: source=webcam (device) 기본 / realsense (/camera/color/image_raw — 손목 카메라라 자리·근접 판단은 끔)
  /cobot2/robot/state (근접 정지), /cobot2/user/command (set_table/clear_table 끝난 자리 상태 초기화)
테스트: debug_input:=true 면 카메라 대신 /cobot2/hri/debug_hand (Float32MultiArray: [오른손?1:0, 21×3 정규화 좌표]) 사용
"""
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32MultiArray

from cobot2_interfaces.msg import RobotState, SeatStatus, UserCommand
from cobot2_hri.common.classifier import GestureClassifier
from cobot2_hri.common.features import INDEX_TIP, PALM_IDS
from cobot2_hri.common.mp_loader import HandTracker, MediaPipeMissing, PoseTracker
from cobot2_hri.common.table_map import Zones, load_calibration, to_table

STOP_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
ARM_IDS = (13, 14, 15, 16)          # MediaPipe Pose 팔꿈치·손목


def _share(*p):
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory('cobot2_hri'), *p)


class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        p = self.declare_parameter
        p('source', 'webcam')                  # webcam / realsense
        p('device', 0)
        p('image_topic', '/camera/color/image_raw')
        p('rate_hz', 15.0)
        p('calibration_file', '~/data/cobot2/webcam_table.yaml')
        p('seats_file', '')                    # 비우면 share/cobot2_hri/config/seats.yaml
        p('static_model', '~/data/assist_arm/models/gesture_static.joblib')
        p('seq_model', '~/data/assist_arm/models/gesture_seq.onnx')
        p('seq_labels', '~/data/assist_arm/models/gesture_seq_labels.json')
        p('hand_task_model', '~/data/assist_arm/models/hand_landmarker.task')
        p('pose_task_model', '~/data/assist_arm/models/pose_landmarker_lite.task')
        p('palm_sign', 1.0)                    # 손바닥(stop)/손등(push) 이 뒤바뀌면 -1.0
        p('confirm_frames', 10)                # 15fps 기준 약 0.7초 같은 판정이면 확정
        p('min_confidence', 0.6)
        p('cooldown_sec', 3.0)                 # 같은 명령 다시 보내기까지
        p('default_pour_ratio', 0.6)
        p('point_dwell_sec', 1.0)
        p('eating_window_sec', 30.0)           # 이 시간 안에 손이 자리에 있었으면 EATING
        p('empty_after_sec', 120.0)            # 이 시간 넘게 손이 없으면 EMPTY
        p('finished_hold_sec', 900.0)          # FINISHED 유지 (set_table/clear_table 이 오면 초기화)
        p('stop_repeat_sec', 0.5)
        p('debug_input', False)
        p('show_debug', False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g

        self.clf = GestureClassifier(g('static_model'), g('seq_model'), g('seq_labels'), float(g('palm_sign')),
                                     self.get_logger())
        self.debug = bool(g('debug_input'))
        self.hands = self.pose = self.cap = None
        if not self.debug:
            self.hands = HandTracker(max_hands=2, task_model_path=g('hand_task_model'))
            self.pose = PoseTracker(task_model_path=g('pose_task_model'))
        self.H = None
        if g('source') == 'webcam':
            self.H, _ = load_calibration(g('calibration_file'))
            if self.H is None:
                alt = os.path.expanduser('~/data/assist_arm/pointer_calibration.yaml')    # assist_arm 에서 만든 것
                self.H, _ = load_calibration(alt)
            if self.H is None:
                self.get_logger().warn('웹캠→테이블 캘리브레이션 없음 → 자리 상태·근접 정지 꺼짐 '
                                       '(tools/calibrate_webcam_table.py 로 만들기)')
        else:
            self.get_logger().info('source=realsense: 손목 카메라라 자리 상태·근접 정지는 끄고 손동작 명령만')
        self.zones = Zones(g('seats_file') or _share('config', 'seats.yaml'))

        self.hist = deque(maxlen=int(g('confirm_frames')))
        self.confirmed = None
        self.last_cmd = {}
        self.point = None                      # (seat, start_t)
        self.last_seen = {sid: -1e9 for sid, _, _ in self.zones.seats}
        self.finished = {}                     # seat → t
        self.status = {sid: SeatStatus.EMPTY for sid, _, _ in self.zones.seats}
        self.robot_moving = False
        self.last_stop_pub = -1e9
        self.debug_hand = None
        self.frame = None

        cbg = ReentrantCallbackGroup()
        self.cmd_pub = self.create_publisher(UserCommand, '/cobot2/user/command', 10)
        self.seat_pub = self.create_publisher(SeatStatus, '/cobot2/hri/seat_status', 10)
        self.stop_pub = self.create_publisher(Bool, '/cobot2/safety/stop', STOP_QOS)
        self.create_subscription(RobotState, '/cobot2/robot/state',
                                 lambda m: setattr(self, 'robot_moving', m.state == RobotState.MOVING), 10,
                                 callback_group=cbg)
        self.create_subscription(UserCommand, '/cobot2/user/command', self.on_command, 10, callback_group=cbg)
        if self.debug:
            self.create_subscription(Float32MultiArray, '/cobot2/hri/debug_hand',
                                     lambda m: setattr(self, 'debug_hand', (time.monotonic(), list(m.data))), 10,
                                     callback_group=cbg)
        elif g('source') == 'webcam':
            self.cap = cv2.VideoCapture(int(g('device')))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                self.get_logger().error(f"웹캠 {g('device')} 를 열 수 없어요")
        else:
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image
            self.bridge = CvBridge()
            self.create_subscription(Image, g('image_topic'),
                                     lambda m: setattr(self, 'frame', self.bridge.imgmsg_to_cv2(m, 'bgr8')), 5,
                                     callback_group=cbg)
        self.create_timer(1.0 / float(g('rate_hz')), self.tick, callback_group=cbg)
        self.create_timer(1.0, self.publish_seats, callback_group=cbg)
        self.get_logger().info(f"gesture_node 준비: source={g('source')}, 분류={self.clf.source()}, "
                               f"자리 {len(self.zones.seats)}개, 캘리브레이션={'있음' if self.H is not None else '없음'}")

    # ------------------------------------------------------------ 입력 한 프레임
    def _read(self):
        """→ (hands [(pts21x3, handedness, world)], arm_uv [(u,v)], (w,h)) 또는 None"""
        if self.debug:
            if self.debug_hand is None or time.monotonic() - self.debug_hand[0] > 0.5:
                return [], [], (640, 480)
            d = self.debug_hand[1]
            if len(d) < 64:
                return [], [], (640, 480)
            pts = np.array(d[1:64], np.float32).reshape(21, 3)
            return [(pts, 'Right' if d[0] > 0.5 else 'Left', None)], [], (640, 480)
        if self.cap is not None:
            ok, frame = self.cap.read()
            frame = frame if ok else None
        else:
            frame = self.frame
        if frame is None:
            return None
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = [(x['pts'], x['handedness'], x['world']) for x in self.hands.process_all(rgb)]
        arms = []
        if self.H is not None:
            lm = self.pose.process(rgb)
            if lm is not None:
                arms = [(lm[i, 0] * w, lm[i, 1] * h) for i in ARM_IDS if lm[i, 3] > 0.5]
        if self.get_parameter('show_debug').value:
            for pts, _, _ in hands:
                for x, y, _ in pts:
                    cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)
            cv2.putText(frame, f'{self.confirmed or ""}', (10, 30), 0, 1.0, (0, 255, 255), 2)
            cv2.imshow('gesture_node', frame)
            cv2.waitKey(1)
        return hands, arms, (w, h)

    def tick(self):
        r = self._read()
        if r is None:
            return
        hands, arms, (w, h) = r
        now = time.monotonic()
        pts, hd, world = hands[0] if hands else (None, 'Right', None)
        label, conf = self.clf.step(pts, hd, world, now)
        lab = label if conf >= self.get_parameter('min_confidence').value else 'idle'
        self.hist.append(lab)
        stable = lab if (len(self.hist) == self.hist.maxlen and len(set(self.hist)) == 1) else None

        # 테이블 좌표 (캘리브레이션 있을 때만)
        palm_xy = tip_xy = None
        table_pts = []
        if self.H is not None:
            for hp, _, _ in hands:
                uv = hp[:, :2] * [w, h]
                table_pts += list(to_table(self.H, uv))
            if arms:
                table_pts += list(to_table(self.H, arms))
            if pts is not None:
                xy = to_table(self.H, pts[:, :2] * [w, h])
                palm_xy, tip_xy = xy[PALM_IDS].mean(axis=0), xy[INDEX_TIP]
            for hp, _, _ in hands:
                sid = self.zones.seat_of(to_table(self.H, hp[PALM_IDS, :2].mean(axis=0, keepdims=True) * [w, h])[0])
                if sid is not None:
                    self.last_seen[sid] = now

        # 안전: 로봇이 움직이는 중 손·팔이 작업 영역 근처
        if self.robot_moving and table_pts and self.zones.near_robot(table_pts):
            self.send_stop('로봇 작업 영역에 손·팔 접근')

        # 손동작 확정 (바뀌는 순간 한 번)
        if stable != self.confirmed:
            self.confirmed = stable
            if stable:
                self.on_gesture(stable, palm_xy, now)
        if stable == 'point':
            self.track_point(tip_xy, now)
        else:
            self.point = None
        self.update_seats(now)

    # ------------------------------------------------------------ 손동작 → 명령
    def seat_for(self, xy):
        if xy is None:
            return None
        return self.zones.seat_of(xy) or self.zones.nearest_seat(xy)

    def on_gesture(self, label, palm_xy, now):
        if label == 'stop':
            self.send_stop('정지 손동작')
            self.send_cmd(UserCommand.CMD_STOP, [], force=True)
        elif label == 'push':
            sid = self.seat_for(palm_xy)
            self.send_cmd(UserCommand.CMD_CLEAR_TABLE, [sid] if sid else [])
        elif label == 'cup_grasp':
            sid = self.seat_for(palm_xy)
            self.send_cmd(UserCommand.CMD_POUR, [sid] if sid else [], self.get_parameter('default_pour_ratio').value)
        elif label == 'thumbs_up':
            sid = self.seat_for(palm_xy)
            targets = [sid] if sid else [s for s, _, _ in self.zones.seats if self.status.get(s) == SeatStatus.EATING]
            for s in targets:
                self.finished[s] = now
            self.get_logger().info(f'다 먹음 표시: 자리 {targets}')

    def track_point(self, tip_xy, now):
        sid = self.zones.seat_of(tip_xy) if tip_xy is not None else None
        if sid is None:
            self.point = None
            return
        if self.point is None or self.point[0] != sid:
            self.point = (sid, now)
        elif now - self.point[1] >= self.get_parameter('point_dwell_sec').value:
            self.send_cmd(UserCommand.CMD_SET_TABLE, [sid])
            self.point = (sid, float('inf'))          # 손을 떼기 전까지 다시 안 보냄

    def send_cmd(self, command, seats, ratio=0.0, force=False):
        now = time.monotonic()
        key = (command, tuple(seats))
        if not force and now - self.last_cmd.get(key, -1e9) < self.get_parameter('cooldown_sec').value:
            return
        self.last_cmd[key] = now
        m = UserCommand(source=UserCommand.SOURCE_GESTURE, command=command,
                        seat_ids=[int(s) for s in seats], pour_ratio=float(ratio))
        m.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(m)
        self.get_logger().info(f'명령: {command} 자리 {list(seats)}' + (f' 비율 {ratio:.2f}' if ratio else ''))

    def send_stop(self, why):
        now = time.monotonic()
        if now - self.last_stop_pub < self.get_parameter('stop_repeat_sec').value:
            return
        self.last_stop_pub = now
        self.stop_pub.publish(Bool(data=True))         # false 는 절대 보내지 않음
        self.get_logger().warn(f'safety/stop = true ({why})')

    def on_command(self, m):
        """set_table / clear_table 이 오면 해당 자리 '다 먹음' 초기화"""
        if m.command in (UserCommand.CMD_SET_TABLE, UserCommand.CMD_CLEAR_TABLE):
            for s in (m.seat_ids or [sid for sid, _, _ in self.zones.seats]):
                self.finished.pop(int(s), None)

    # ------------------------------------------------------------ 자리 상태
    def update_seats(self, now):
        g = self.get_parameter
        changed = False
        for sid, _, _ in self.zones.seats:
            old = self.status[sid]
            if sid in self.finished and now - self.finished[sid] < g('finished_hold_sec').value:
                new = SeatStatus.FINISHED
            elif now - self.last_seen[sid] < g('eating_window_sec').value:
                new = SeatStatus.EATING
            elif now - self.last_seen[sid] > g('empty_after_sec').value:
                new = SeatStatus.EMPTY
            else:
                new = old if old != SeatStatus.FINISHED else SeatStatus.EMPTY
            if new != old:
                self.status[sid] = new
                changed = True
        if changed:
            self.publish_seats()

    def publish_seats(self):
        for sid, st in self.status.items():
            m = SeatStatus(seat_id=int(sid), status=int(st))
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = 'base_link'
            self.seat_pub.publish(m)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = GestureNode()
    except MediaPipeMissing as e:
        print(f'[gesture_node] {e}', file=sys.stderr)
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
