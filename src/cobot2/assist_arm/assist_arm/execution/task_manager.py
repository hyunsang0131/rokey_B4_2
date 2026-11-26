"""task_manager: 가정용 손 상호작용 로봇의 상태 머신 (로봇에 명령을 내리는 유일한 노드).

요청: /assist/request (request_router 가 제스처·음성·콕을 합쳐 만듦) → 기능별 서브태스크(subtasks.py)
  bring_dish   콕 / "<음식> 줘"         → 그릇을 사용자 앞으로
  clear_dish   콕 + 치워                → 그 그릇 정리
  clear_empty  치워 (손등 밀기)          → 빈 그릇 스스로 찾아 정리
  clear_all    식사 끝 (엄지척 / "다 먹었어") → 그릇 → 컵 → 수저 순서로 전체 정리
  pour_water   "물 줘" + 컵 잡기 손동작   → 컵 찾기 → 따르기 → 손 옆 15cm 테이블에 내려놓기
  seat_setting "N명이야" + 자리 콕 N번    → 자리마다 수저·그릇·음식 배치
  music        노래/율동                 → 곡 추천·재생 + 안전한 율동 웨이포인트
  wipe_table   닦아줘                    → 걸레로 테이블 얼룩 닦기 (힘제어)
  cancel       그만/취소                 → 현재 작업 중단 후 홈
상태: /assist/state (String), 안내: /assist/announce (String)
안전: /safety/stop → SafetyAbort (자동 재개 없음), /safety/pause → 동작 대기 후 재개 (safety_monitor 가 MovePause/Resume)
"""
import json
import os
import threading
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger

from assist_arm_msgs.msg import HandTable, TaskRequest
from assist_arm_msgs.srv import SetSafetyMode
from assist_arm.common.geometry import load_gripper2cam
from assist_arm.common.heartbeat import attach_heartbeat
from assist_arm.execution.robot_io import RobotIO, SafetyAbort, TaskCancelled, TaskFailed
from assist_arm.execution.skills import NO_HOME, TASKS, validate_dance
from od_msg.srv import SrvDepthPosition


class TaskManager(Node, RobotIO):
    def __init__(self):
        super().__init__('task_manager')
        p = self.declare_parameter
        # 로봇/그리퍼 (robot_control.py 와 같은 기본값)
        p('robot_id', 'dsr01')
        p('robot_model', 'm0609')
        p('gripper_name', 'rg2')
        p('toolcharger_ip', '192.168.1.1')
        p('toolcharger_port', '502')
        p('gripper2cam_path', '')
        p('waypoints_file', '')
        p('table_layout_file', '')
        p('dry_run', False)
        p('dry_fake_detections', True)
        p('dry_move_sec', 0.0)              # dry_run 에서 동작 1개당 기다릴 시간 (흐름 테스트용)      # dry_run 에서 waypoints.yaml 의 dry_detections 사용
        # 기능 on/off
        for t in TASKS:
            p(f'enable_{t}', True)
        # 인지
        p('detection_service', '/assist/get_3d_position')
        p('cup_class', 'cup')
        p('detect_retries', 3)
        p('settle_sec', 0.5)
        p('poke_radius_mm', 120.0)
        p('empty_fill_threshold', 0.3)
        p('max_clear_items', 8)
        p('seats_file', '~/data/assist_arm/seats.yaml')
        p('cup_near_hand_mm', 250.0)        # 손 → 컵 내려놓는 거리. safety pause_mm(200) 보다 크게
        # 속도 (safety_monitor /safety/limits 가 오면 덮어씀)
        p('speed_free_mms', 250.0)
        p('speed_near_human_mms', 100.0)
        p('speed_final_approach_mms', 40.0)
        p('dance_max_speed_mms', 150.0)
        p('joint_vel_dps', 60.0)
        # 닦기
        p('wipe_force_n', 4.0)
        p('max_contact_force_n', 15.0)
        p('wipe_stiffness', [500.0, 500.0, 500.0, 100.0, 100.0, 100.0])
        p('wipe_margin_mm', 30.0)
        p('wipe_max_amp_mm', 80.0)
        p('wipe_step_mm', 15.0)
        p('dab_sec', 1.5)
        p('open_on_fail', False)
        p('return_to_observe', True)        # 작업이 끝나면 홈 대신 관찰 자세로 돌아가 스냅샷 후 손동작 대기
        p('start_observing', True)          # 시작하자마자 관찰 자세로            # 실패 시 그리퍼 열기 (물건을 떨어뜨릴 수 있어 기본 off)
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731

        share = get_package_share_directory('assist_arm')
        self.wp = self._yaml(self.g('waypoints_file') or os.path.join(share, 'config', 'waypoints.yaml'))
        self.layout = self._yaml(self.g('table_layout_file') or os.path.join(share, 'config', 'table_layout.yaml'))
        self.table_z = float(self.layout.get('table_z_mm', 0.0))
        validate_dance(self.wp, self.get_logger())
        self.T_g2c = load_gripper2cam(self.g('gripper2cam_path'))

        self.stop_flag = self.pause_flag = self.cancel_flag = False
        self.busy = False
        self.request = None
        self.palm = None
        self.limits = {}
        self.now_playing = None
        self.rack_used, self.clear_count, self.dry_removed = {}, 0, set()

        cbg = ReentrantCallbackGroup()
        self.create_subscription(TaskRequest, '/assist/request', self.on_request, 10, callback_group=cbg)
        self.create_subscription(Bool, '/safety/stop', self.on_stop, 10, callback_group=cbg)
        self.create_subscription(Bool, '/safety/pause', lambda m: setattr(self, 'pause_flag', bool(m.data)),
                                 10, callback_group=cbg)
        self.create_subscription(String, '/safety/limits', self.on_limits, 10, callback_group=cbg)
        self.create_subscription(HandTable, '/user/hand_table', self.on_hand, 10, callback_group=cbg)
        self.create_subscription(String, '/music/now_playing', self.on_music, 10, callback_group=cbg)
        self.state_pub = self.create_publisher(String, '/assist/state', 10)
        self.announce_pub = self.create_publisher(String, '/assist/announce', 10)
        self.scene_pub = self.create_publisher(String, '/assist/scene', 10)
        self.snapshot_pub = self.create_publisher(String, '/table/snapshot', 10)
        self.observing_pub = self.create_publisher(Bool, '/assist/observing', 10)
        self.snapshot, self.at_observe = None, False
        self.create_timer(0.1, self._pub_observing)
        self.done_pub = self.create_publisher(String, '/assist/task_done', 10)
        self.express_done = False
        self.music_play_pub = self.create_publisher(String, '/music/play', 10)
        self.music_stop_pub = self.create_publisher(Empty, '/music/stop', 10)
        self.det_cli = self.create_client(SrvDepthPosition, self.g('detection_service'), callback_group=cbg)
        self.spill_scan_cli = self.create_client(Trigger, '/spill/scan', callback_group=cbg)
        self.mode_cli = self.create_client(SetSafetyMode, '/safety/set_mode', callback_group=cbg)
        attach_heartbeat(self)

    @staticmethod
    def _yaml(path):
        with open(os.path.expanduser(path)) as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------ 콜백
    def _pub_observing(self):
        """관찰 자세에 멈춰 있고 작업이 없을 때만 True → rs_hand_node·gesture_node·finger_pointer 가 RealSense 사용"""
        obs = bool(self.at_observe and not self.busy and not self.stop_flag and not self.pause_flag)
        try:
            self.observing_pub.publish(Bool(data=obs))
        except Exception:  # noqa: BLE001 (종료 중)
            pass

    def go_rest(self):
        """작업 뒤 쉬는 자세: return_to_observe 면 관찰 자세 + 스냅샷, 아니면 홈"""
        if self.g('return_to_observe'):
            self.state_note('OBSERVE_TABLE')
            TASKS['observe_table'](self, {})
            self.state_note('OBSERVING', '— 손동작 기다리는 중 (RealSense)')
        else:
            self.state_note('HOME')
            self.home()
            self.state_note('IDLE')

    def on_request(self, m):
        if m.task == 'cancel':
            if self.busy:
                self.cancel_flag = True
            return
        if m.task not in TASKS:
            self.get_logger().warn(f'모르는 작업: {m.task}')
            return
        if not self.g(f'enable_{m.task}'):
            self.say(f'{m.task} 기능은 꺼져 있어요')
            return
        if self.busy or self.request is not None:
            self.say('지금 다른 일을 하고 있어요')
            return
        self.request = m

    def on_stop(self, m):
        if m.data and not self.stop_flag:
            self.get_logger().error('/safety/stop 수신 → 현재 작업 중단')
        self.stop_flag = bool(m.data)

    def on_limits(self, m):
        try:
            self.limits = json.loads(m.data)
        except ValueError:
            pass

    def on_hand(self, m):
        if m.valid:
            self.palm = (time.monotonic(), [m.palm.x, m.palm.y, m.palm.z])

    def on_music(self, m):
        try:
            self.now_playing = json.loads(m.data)
        except ValueError:
            pass

    # ------------------------------------------------------------------ 유틸
    def state_note(self, s, extra=''):
        try:
            self.state_pub.publish(String(data=f'{s} {extra}'.strip()))
        except Exception:  # noqa: BLE001 (종료 중)
            pass
        self.get_logger().info(f'[STATE] {s} {extra}')

    def say(self, text):
        self.announce_pub.publish(String(data=text))
        self.get_logger().info(f'[안내] {text}')

    def speed(self, key):
        return float(self.limits.get(key, self.g(key)))

    def call(self, client, req, timeout=10.0):
        if not client.wait_for_service(timeout_sec=2.0):
            raise TaskFailed(f'서비스 없음: {client.srv_name}')
        fut = client.call_async(req)
        t0 = time.monotonic()
        while not fut.done():
            self.check()
            if time.monotonic() - t0 > timeout:
                raise TaskFailed(f'서비스 타임아웃: {client.srv_name}')
            time.sleep(0.02)
        return fut.result()

    def set_safety_mode(self, mode):
        try:
            self.call(self.mode_cli, SetSafetyMode.Request(mode=mode), timeout=3.0)
        except TaskFailed as e:
            self.get_logger().warn(f'safety 모드 변경 실패: {e}')

    # ------------------------------------------------------------------ 닦기 (힘제어)
    def wipe_at(self, xy, kind, size_mm):
        """걸레를 쥔 채 얼룩 위로 → 툴 +Z(아래) 방향 약한 힘 → 지그재그(액체) / 한 방향 쓸기(가루)."""
        c = self.item_cfg('rag')
        z_contact = self.table_z + float(c.get('wipe_z', 30.0))
        above = [xy[0], xy[1], z_contact + 60.0] + self.orient('rag')
        pre = [xy[0], xy[1], z_contact + 15.0] + self.orient('rag')
        va = self.speed('speed_final_approach_mms')
        self.movel(above, self.speed('speed_free_mms'))
        self.movel(pre, va)
        amp = float(np.clip(size_mm / 2 + self.g('wipe_margin_mm'), 20.0, self.g('wipe_max_amp_mm')))
        step, rows = float(self.g('wipe_step_mm')), max(2, int(2 * amp / max(1.0, self.g('wipe_step_mm'))))
        if kind == 'liquid':     # 툴 X 왕복, 툴 Y 로 한 칸씩
            moves = [[0, -amp, 0, 0, 0, 0], [-amp, 0, 0, 0, 0, 0]]
            for r in range(rows):
                moves += [[2 * amp if r % 2 == 0 else -2 * amp, 0, 0, 0, 0, 0], [0, step, 0, 0, 0, 0]]
        else:                    # 가루: 한 방향(툴 +X)으로만 쓸고 들어서 돌아오기 대신 옆 줄로
            moves = [[0, -amp, 0, 0, 0, 0]]
            for _ in range(rows):
                moves += [[2 * amp, 0, 0, 0, 0, 0], [-2 * amp, step, 0, 0, 0, 0]]
        if self.dr is None:
            self.get_logger().info(f'(dry) {kind} 닦기 amp={amp:.0f}mm rows={rows} moves={len(moves)}')
            self.movel(above, va)
            return
        dr = self.dr
        F = min(float(self.limits.get('wipe_force_n', self.g('wipe_force_n'))), 5.0)
        try:
            dr.set_ref_coord(dr.DR_TOOL)
            dr.task_compliance_ctrl(stx=[float(v) for v in self.g('wipe_stiffness')])
            dr.set_desired_force(fd=[0.0, 0.0, F, 0.0, 0.0, 0.0], dir=[0, 0, 1, 0, 0, 0],
                                 mod=dr.DR_FC_MOD_REL)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 3.0:            # 테이블에 닿을 때까지
                self.check()
                if np.linalg.norm(self.tool_force()) > F * 0.7:
                    break
                time.sleep(0.05)
            if kind == 'liquid':
                time.sleep(self.g('dab_sec'))              # 눌러서 흡수
            for mv in moves:
                if np.linalg.norm(self.tool_force()) > self.g('max_contact_force_n'):
                    raise SafetyAbort()
                self.movel(mv, va, ref=dr.DR_TOOL, mod=dr.DR_MV_MOD_REL)
        finally:
            try:
                dr.release_force()
                dr.release_compliance_ctrl()
                dr.set_ref_coord(dr.DR_BASE)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'힘제어 해제 실패: {e}')
        self.movel(above, va)

    # ------------------------------------------------------------------ 메인 루프
    def run(self):
        self.connect_robot()
        try:
            self.home()
            self.grip(False)
            if self.g('start_observing'):
                self.go_rest()
        except (SafetyAbort, TaskCancelled, TaskFailed) as e:
            self.get_logger().warn(f'시작 자세 실패: {e}')
        if not self.at_observe:
            self.state_note('IDLE')
        while rclpy.ok():
            time.sleep(0.05)
            if self.request is None or self.stop_flag:
                continue
            req, self.request = self.request, None
            args = self.args_of(req)
            self.busy, self.cancel_flag = True, False
            self.state_note(req.task.upper(), f'{args} ({req.source})')
            try:
                TASKS[req.task](self, args)
                if req.task not in NO_HOME or (req.task == 'express' and args.get('motion') == 'retreat'):
                    self.go_rest()
                elif req.task in ('observe_table', 'scan_scene'):
                    self.state_note('OBSERVING', '— 손동작 기다리는 중 (RealSense)')
                self.done_pub.publish(String(data=json.dumps({'task': req.task, 'ok': True})))
            except SafetyAbort:
                self.state_note('SAFETY_STOP', '— /safety/reset 후 IDLE (로봇은 자동으로 움직이지 않음)')
                while rclpy.ok() and self.stop_flag:
                    time.sleep(0.1)
            except (TaskCancelled, TaskFailed) as e:
                if isinstance(e, TaskCancelled):
                    self.say('그만할게요')
                else:
                    self.get_logger().error(f'작업 실패: {e}')
                    self.say('잘 안 됐어요')
                self.cancel_flag = False
                try:
                    if self.gripper is not None and self.g('open_on_fail'):
                        self.grip(False)
                    self.go_rest()
                except (SafetyAbort, TaskCancelled, TaskFailed):
                    pass
            finally:
                self.busy, self.cancel_flag = False, False
                self.publish_work_point(None)
            if not self.at_observe:
                self.state_note('IDLE')


def main(args=None):
    rclpy.init(args=args)
    node = TaskManager()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    def spin():
        try:
            ex.spin()
        except Exception:  # noqa: BLE001  (종료 중)
            if rclpy.ok():
                raise
    threading.Thread(target=spin, daemon=True).start()
    try:
        node.run()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ex.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
