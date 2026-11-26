"""pour 모듈 (호연 작성, skill_server 안에서 사용) — Pour 액션 처리 로직.  [팀 설계 v1.1]

skill_server(현상) 가 하는 일
  robot = DsrRobot(dr=DSR_ROBOT2 모듈, gripper_open=..., gripper_close=...)   # 또는 테스트용 FakeRobot()
  pour = PourModule(robot, lock=shared_motion_lock, logger=node.get_logger())  # Skill 과 같은 잠금
  /cobot2/pour/level 콜백에서     pour.on_level(msg.ratio, msg.confidence)
  /cobot2/safety/stop 콜백에서    pour.request_stop(msg.data)
  Pour 액션 execute 콜백에서
      res = pour.execute(cup_pose=goal.cup_pose, bottle_pose=goal.bottle_pose, target_ratio=goal.target_ratio,
                         feedback=lambda r: goal_handle.publish_feedback(Pour.Feedback(current_ratio=r)),
                         cancelled=lambda: goal_handle.is_cancel_requested)
      → res = {'final_ratio', 'success', 'error_code'}  (error_code: OK / BUSY / STOPPED / FAILED / UNREACHABLE)
단위: 바깥(액션·토픽) m·rad·쿼터니언 ↔ 로봇(DSR) mm·deg(ZYZ) 변환은 이 모듈 경계(pose_to_posx)에서만.
흐름: 잠금 → 물병 집기 → 컵 위 따르는 자세 → 단계적으로 기울이기 (수위 보며) → 목표 근처 감속 → 도달하면 되세우기
      → 물병 제자리 → 놓기 → 잠금 해제.  수위 확신도가 낮거나 끊기면 시간·각도 기준 추정으로 대체, 전체 타임아웃.
      stop / cancel 이 오면 즉시 로봇 정지 → STOPPED (들고 있는 물병은 skill_server 의 복구 규칙 5.2 로)
"""
import threading
import time
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

DEFAULTS = {
    'vel_mms': 80.0,              # 일반 이동 속도 (mm/s)
    'slow_vel_mms': 30.0,         # 기울이기·접근 속도
    'pre_lift_m': 0.12,           # 집기·놓기 전 위쪽 접근 높이
    'grasp_dz_m': 0.0,            # bottle_pose 기준 파지 높이 보정
    'pour_offset_m': [-0.06, 0.0, 0.12],   # 컵 위치 기준 따르는 자세 (물병 입구가 컵 위에 오도록, base_link)
    'tilt_axis': [0.0, 1.0, 0.0],          # 기울이는 축 (base_link, 단위 벡터)
    'tilt_start_deg': 60.0,       # 물이 나오기 직전 각도까지는 한 번에
    'tilt_max_deg': 120.0,
    'tilt_step_deg': 4.0,
    'tilt_step_slow_deg': 1.5,    # 목표 근처
    'step_wait_s': 0.4,
    'slowdown_margin': 0.15,      # 목표 - 이 값 이후 감속
    'stop_lead': 0.03,            # 흐름이 남아서 목표보다 이만큼 먼저 되세움
    'min_conf': 0.4,
    'level_timeout_s': 0.6,       # 수위 메시지가 이보다 오래 없으면 '끊김'
    'fallback_rate_per_s': 0.06,  # 수위를 못 볼 때 가정하는 차오르는 속도 (최대 기울기 근처, 비율/초)
    'timeout_s': 30.0,
}


# ------------------------------------------------------------------ 단위 변환 (모듈 경계)
def pose_to_posx(pose):
    """geometry_msgs/Pose (m, 쿼터니언) → 두산 posx [x,y,z mm, rx,ry,rz deg ZYZ]"""
    p, q = pose.position, pose.orientation
    quat = [q.x, q.y, q.z, q.w]
    if np.allclose(quat, 0):
        quat = [0, 0, 0, 1]
    with warnings.catch_warnings():          # ry=180°(그리퍼 수직 아래) 는 짐벌락 경고가 나지만 값은 맞음
        warnings.simplefilter('ignore')
        zyz = Rotation.from_quat(quat).as_euler('ZYZ', degrees=True)
    return [p.x * 1000.0, p.y * 1000.0, p.z * 1000.0, *zyz]


def posx_to_matrix(posx):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('ZYZ', posx[3:6], degrees=True).as_matrix()
    T[:3, 3] = posx[:3]
    return T


def matrix_to_posx(T):
    return [*T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_euler('ZYZ', degrees=True)]


class StopRequested(Exception):
    pass


# ------------------------------------------------------------------ 로봇 인터페이스 (mm·deg)
class RobotInterface:
    def movel(self, posx, vel_mms):
        raise NotImplementedError

    def grip(self, close):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def get_posx(self):
        raise NotImplementedError


class DsrRobot(RobotInterface):
    """skill_server 용 어댑터: DSR_ROBOT2 모듈과 그리퍼 함수를 넘겨받음 (DR_init 설정은 skill_server 가)"""

    def __init__(self, dr, gripper_open, gripper_close, stop_mode=1):
        self.dr, self.go, self.gc, self.stop_mode = dr, gripper_open, gripper_close, stop_mode

    def movel(self, posx, vel_mms):
        self.dr.movel([float(v) for v in posx], vel=[vel_mms, 30.0], acc=[2 * vel_mms, 60.0])

    def grip(self, close):
        (self.gc if close else self.go)()

    def stop(self):
        """실제 감속 정지는 skill_server 가 /dsr01/.../motion/move_stop 서비스로 이미 처리한다고 가정.
        여기서는 다음 movel 을 보내지 않게만 함 (PourModule 이 StopRequested 로 빠져나감)."""

    def get_posx(self):
        return list(self.dr.get_current_posx()[0])


class FakeRobot(RobotInterface):
    """테스트용: 움직이는 척 + (선택) 기울기에 따른 수위 시뮬레이션"""

    def __init__(self, logger=None, speed=4.0, sim_level=True):
        self.log, self.speed, self.sim = logger, speed, sim_level
        self.pos = [400.0, 0.0, 400.0, 0.0, 180.0, 0.0]
        self.level, self.tilt_deg, self.holding = 0.0, 0.0, False
        self.stopped = False

    def _say(self, m):
        if self.log:
            self.log.info(f'(fake) {m}')

    def movel(self, posx, vel_mms):
        d = np.linalg.norm(np.array(posx[:3]) - np.array(self.pos[:3]))
        t_end = time.monotonic() + min(1.5, d / max(vel_mms, 1) / self.speed)
        while time.monotonic() < t_end:
            if self.stopped:
                raise StopRequested()
            time.sleep(0.02)
        self.pos = list(posx)
        self._say(f'movel {np.round(posx, 1).tolist()} v={vel_mms:.0f}')

    def grip(self, close):
        self.holding = close
        self._say('grip close' if close else 'grip open')

    def stop(self):
        self.stopped = True
        self._say('STOP')

    def get_posx(self):
        return list(self.pos)

    def step_level(self, dt):
        """기울기 70° 이상이면 물이 나옴 (시뮬레이션)"""
        if self.sim and self.tilt_deg > 70:
            self.level = min(1.0, self.level + dt * 0.08 * (self.tilt_deg - 70) / 40)
        return self.level


# ------------------------------------------------------------------ Pour 로직
class PourModule:
    def __init__(self, robot, lock=None, logger=None, params=None):
        self.robot = robot
        self.lock = lock or threading.Lock()
        self.log = logger
        self.p = {**DEFAULTS, **(params or {})}
        self.level = (0.0, 0.0, 0.0)          # ratio, conf, 받은 시각
        self.stop_flag = False

    def _say(self, m, warn=False):
        if self.log:
            (self.log.warn if warn else self.log.info)(f'[pour] {m}')

    # skill_server 가 연결
    def on_level(self, ratio, confidence):
        self.level = (float(ratio), float(confidence), time.monotonic())

    def request_stop(self, stop=True):
        self.stop_flag = bool(stop)
        if stop:
            self.robot.stop()

    def _check(self, cancelled):
        if self.stop_flag or (cancelled and cancelled()):
            raise StopRequested()

    def _level_ok(self):
        r, c, t = self.level
        return c >= self.p['min_conf'] and time.monotonic() - t < self.p['level_timeout_s']

    def _tilted(self, base_T, deg):
        axis = np.asarray(self.p['tilt_axis'], float)
        R = Rotation.from_rotvec(np.deg2rad(deg) * axis / np.linalg.norm(axis)).as_matrix()
        T = base_T.copy()
        T[:3, :3] = R @ base_T[:3, :3]
        return matrix_to_posx(T)

    def execute(self, cup_pose, bottle_pose, target_ratio, feedback=None, cancelled=None):
        p = self.p
        if not self.lock.acquire(blocking=False):
            return {'final_ratio': float(self.level[0]), 'success': False, 'error_code': 'BUSY'}
        if self.stop_flag:
            self.lock.release()
            return {'final_ratio': float(self.level[0]), 'success': False, 'error_code': 'STOPPED'}
        target = float(np.clip(target_ratio, 0.05, 1.0))
        t0 = time.monotonic()
        est = self.level[0]
        holding = False
        try:
            vel, slow = p['vel_mms'], p['slow_vel_mms']
            bottle = pose_to_posx(bottle_pose)
            bottle[2] += p['grasp_dz_m'] * 1000
            above_b = bottle[:2] + [bottle[2] + p['pre_lift_m'] * 1000] + bottle[3:]
            cup = pose_to_posx(cup_pose)
            pour_x = [cup[0] + p['pour_offset_m'][0] * 1000, cup[1] + p['pour_offset_m'][1] * 1000,
                      cup[2] + p['pour_offset_m'][2] * 1000] + bottle[3:]
            # 1) 물병 집기
            self._check(cancelled)
            self._say(f'물병 집기 → 목표 수위 {target:.2f}')
            self.robot.grip(False)
            self.robot.movel(above_b, vel)
            self._check(cancelled)
            self.robot.movel(bottle, slow)
            self.robot.grip(True)
            holding = True
            self.robot.movel(above_b, slow)
            # 2) 컵 위 따르는 자세
            self._check(cancelled)
            self.robot.movel(pour_x, vel)
            base_T = posx_to_matrix(pour_x)
            # 3) 단계적으로 기울이기
            angle = p['tilt_start_deg']
            self.robot.movel(self._tilted(base_T, angle), slow)
            self._set_tilt(angle)
            t_last = time.monotonic()
            reached = False
            while True:
                self._check(cancelled)
                now = time.monotonic()
                if now - t0 > p['timeout_s']:
                    self._say('타임아웃 → 되세움', warn=True)
                    break
                self._sim_step(now - t_last)
                if self._level_ok():
                    est = self.level[0]
                else:
                    est = min(1.0, est + p['fallback_rate_per_s'] * (now - t_last) * (angle / p['tilt_max_deg']))
                t_last = now
                if feedback:
                    feedback(float(est))
                if est >= target - p['stop_lead']:
                    reached = True
                    break
                step = p['tilt_step_slow_deg'] if est >= target - p['slowdown_margin'] else p['tilt_step_deg']
                if angle < p['tilt_max_deg']:
                    angle = min(p['tilt_max_deg'], angle + step)
                    self.robot.movel(self._tilted(base_T, angle), slow)
                    self._set_tilt(angle)
                time.sleep(p['step_wait_s'])
            # 4) 되세우기 → 물병 제자리 → 놓기
            self.robot.movel(pour_x, slow)
            self._set_tilt(0.0)
            self.robot.movel(above_b, vel)
            self.robot.movel(bottle, slow)
            self.robot.grip(False)
            holding = False
            self.robot.movel(above_b, slow)
            final = float(self.level[0]) if self._level_ok() else float(est)
            ok = reached and abs(final - target) <= 0.15
            self._say(f'끝: 수위 {final:.2f} (목표 {target:.2f}) {"성공" if ok else "실패"}')
            return {'final_ratio': final, 'success': bool(ok), 'error_code': 'OK' if ok else 'FAILED'}
        except StopRequested:
            self.robot.stop()
            self._say(f'정지 요청 → STOPPED (물병 {"들고 있음 — 복구 필요" if holding else "안 듦"})', warn=True)
            return {'final_ratio': float(est), 'success': False, 'error_code': 'STOPPED'}
        except Exception as e:  # noqa: BLE001
            self.robot.stop()
            self._say(f'실패: {e}', warn=True)
            return {'final_ratio': float(est), 'success': False, 'error_code': 'FAILED'}
        finally:
            self.lock.release()

    # FakeRobot 시뮬레이션 연결 (실제 로봇에서는 아무것도 안 함)
    def _set_tilt(self, deg):
        if isinstance(self.robot, FakeRobot):
            self.robot.tilt_deg = deg

    def _sim_step(self, dt):
        if isinstance(self.robot, FakeRobot) and self.robot.sim:
            self.on_level(self.robot.step_level(dt), 0.9)
