"""task_manager 가 쓰는 로봇 입출력 + 공통 동작(skill).

로봇 제어는 기존 robot_control/robot_control.py 방식 그대로:
  DR_init.__dsr__id/__dsr__model/__dsr__node 설정 후 DSR_ROBOT2 import, 그리퍼 robot_control.onrobot.RG,
  좌표 변환 T_gripper2camera + get_current_posx (robot_control.get_target_pos 와 같은 흐름).
차이: 정지/일시정지/취소에 반응하려고 movel/movej 대신 amovel/amovej + check_motion 폴링.
dry_run=True 면 DSR 을 import 하지 않고 로그만 남긴다.
"""
import json
import time

import numpy as np
from geometry_msgs.msg import PointStamped

from assist_arm.common.geometry import transform_to_base
from od_msg.srv import SrvDepthPosition


class SafetyAbort(Exception):
    """/safety/stop — 시나리오 즉시 중단, 로봇은 자동으로 다시 움직이지 않음"""


class TaskCancelled(Exception):
    """사용자 '그만/취소' — 현재 작업만 중단하고 홈으로"""


class TaskFailed(Exception):
    pass


class RobotIO:
    """TaskManager(Node) 가 상속. self.g(name) 파라미터, self.wp/self.layout 설정을 쓴다."""

    # ------------------------------------------------------------------ 연결
    def connect_robot(self):
        self.dr = None
        self.gripper = None
        self.work_pub = self.create_publisher(PointStamped, '/assist/work_point', 10)
        if self.g('dry_run'):
            self.get_logger().warn('dry_run=True: 로봇 명령을 보내지 않습니다')
            return
        import rclpy
        import DR_init
        DR_init.__dsr__id = self.g('robot_id')
        DR_init.__dsr__model = self.g('robot_model')
        self.dsr_node = rclpy.create_node('assist_task_dsr', namespace=self.g('robot_id'))
        DR_init.__dsr__node = self.dsr_node
        import DSR_ROBOT2 as dr  # noqa: N813  (DR_init 설정 후에만 import 가능)
        self.dr = dr
        from robot_control.onrobot import RG
        self.gripper = RG(self.g('gripper_name'), self.g('toolcharger_ip'), self.g('toolcharger_port'))

    # ------------------------------------------------------------------ 상태 확인
    def check(self):
        if self.stop_flag:
            raise SafetyAbort()
        if self.cancel_flag:
            raise TaskCancelled()

    def wait_unpaused(self):
        announced = False
        while self.pause_flag:
            self.check()
            if not announced:
                self.state_note('PAUSED', '손/팔이 가까워 일시정지 — 벗어나면 재개')
                self.say('손을 조금만 떼 주세요')
                announced = True
            time.sleep(0.05)

    def wait_motion(self):
        time.sleep(0.1)
        while True:
            self.check()
            if not self.pause_flag and self.dr.check_motion() == self.dr.DR_STATE_IDLE:
                return
            time.sleep(0.05)

    def publish_work_point(self, xyz):
        m = PointStamped()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), 'base'
        m.point.x, m.point.y, m.point.z = (float('nan'),) * 3 if xyz is None else map(float, xyz[:3])
        try:
            self.work_pub.publish(m)
        except Exception:  # noqa: BLE001  (종료 중 context 무효)
            pass

    # ------------------------------------------------------------------ 동작
    def movel(self, pos, vel, ref=None, mod=None, t=None):
        self.at_observe = False
        self.check()
        self.wait_unpaused()
        if mod is None:
            self.publish_work_point(pos)
        if self.dr is None:
            self.get_logger().info(f'(dry) movel {np.round(pos, 1).tolist()} v={vel:.0f}'
                                   + (f' t={t:.2f}' if t else ''))
            time.sleep(0.2 * (t or 0.0) + float(self.g('dry_move_sec')))   # dry_run: 이동 시간 흉내
            return
        kw = {}
        if ref is not None:
            kw['ref'] = ref
        if mod is not None:
            kw['mod'] = mod
        if t is not None:
            kw['time'] = float(t)
        self.dr.amovel([float(v) for v in pos], vel=[vel, 30.0], acc=[2 * vel, 60.0], **kw)
        self.wait_motion()

    def movej(self, q, vel=None):
        self.at_observe = False
        self.check()
        self.wait_unpaused()
        vel = vel or self.g('joint_vel_dps')
        if self.dr is None:
            self.get_logger().info(f'(dry) movej {q}')
            time.sleep(float(self.g('dry_move_sec')))
            return
        self.dr.amovej([float(v) for v in q], vel=vel, acc=vel)
        self.wait_motion()

    def posx(self):
        if self.dr is None:
            return list(self.wp.get('dry_posx', [400, 0, 400, 0, 180, 0]))
        return list(self.dr.get_current_posx()[0])

    def tool_force(self):
        if self.dr is None:
            return np.zeros(3)
        f = self.dr.get_tool_force()
        return np.array(f[:3]) if isinstance(f, list) else np.zeros(3)

    def grip(self, close, width=None):
        self.check()
        if not close:
            self.holding = None                   # 열면 빈손 (v2.2: 들고 있을 때 복구 판단용)
        if self.gripper is None:
            self.get_logger().info(f'(dry) gripper {"close" if close else "open"}'
                                   + (f' width={width}' if width else ''))
            return
        if width is not None and not close:
            self.gripper.move_gripper(int(width))
        else:
            (self.gripper.close_gripper if close else self.gripper.open_gripper)()
        t0 = time.monotonic()
        while self.gripper.get_status()[0] and time.monotonic() - t0 < 5.0:
            self.check()
            time.sleep(0.2)

    def home(self):
        self.publish_work_point(None)
        self.movej(self.wp['home_j'])

    # ------------------------------------------------------------------ skill
    def item_cfg(self, item):
        items = self.layout.get('items', {})
        return {**items.get('default', {}), **items.get(item, {})}

    def orient(self, item):
        return list(self.item_cfg(item).get('orientation', self.layout.get('place_orientation', [0, 180, 0])))

    def pick(self, item, xyz, z_offset=0.0):
        """xyz: 베이스 좌표(물체 중심 또는 위 표면). 아이템별 grasp_z(테이블 기준) / pre_lift 사용.
        z_offset: 보관 더미 맨 위에서 꺼낼 때 (아래 개수 × 포갬 높이, mm) — v2.2 실행기"""
        c = self.item_cfg(item)
        z = self.table_z + float(c.get('grasp_z', 40.0)) if c.get('use_table_z', True) else xyz[2]
        z += float(z_offset)
        grasp = [xyz[0], xyz[1], z] + self.orient(item)
        pre = grasp[:2] + [z + float(c.get('pre_lift', 120.0))] + grasp[3:]
        vf = self.speed('speed_free_mms')
        if c.get('open_width'):
            self.grip(False, width=c['open_width'])
        self.movel(pre, vf)
        self.movel(grasp, vf * 0.4)
        self.grip(True)
        self.holding = item
        self.movel(pre, vf * 0.6)

    def place(self, item, xy, slow=False, z_offset=0.0):
        """z_offset: 더미 위에 포갤 때 (이미 있는 개수 × 포갬 높이, mm). 접촉 감지 하강은 실물 단계에서 추가"""
        c = self.item_cfg(item)
        z = self.table_z + float(c.get('place_z', c.get('grasp_z', 40.0))) + float(z_offset)
        tgt = [xy[0], xy[1], z] + self.orient(item)
        pre = tgt[:2] + [z + float(c.get('pre_lift', 120.0))] + tgt[3:]
        v = self.speed('speed_near_human_mms') if slow else self.speed('speed_free_mms')
        self.movel(pre, v)
        self.movel(tgt, self.speed('speed_final_approach_mms') if slow else v * 0.4)
        self.grip(False)
        self.movel(pre, v)

    def pick_from_rack(self, item):
        r = self.layout['racks'][item]
        k = self.rack_used.get(item, 0)
        if k >= int(r.get('count', 1)):
            raise TaskFailed(f'거치대에 {item} 없음 (count={r.get("count", 1)})')
        xyz = np.array(r['pos'][:3]) + k * np.array(r.get('step', [0, 0, 0]))
        self.rack_used[item] = k + 1
        self.pick(item, xyz)

    def recover_holding(self):
        """v2.2 6.4: 물체를 쥔 채 작업이 끊겼을 때 → 비어 있는 임시 자리에 내려놓기. 내려놓았으면 True"""
        item = getattr(self, 'holding', None)
        if not item:
            return False
        slots = (self.layout.get('storage', {}) or {}).get('temp_slots', {}) or {}
        if not slots:
            raise TaskFailed(f'{item} 을(를) 들고 있지만 임시 자리 설정이 없음 — 사람이 직접 빼 주세요')
        name, cfg = next(iter(slots.items()))
        self.state_note('RECOVER', f'{item} → {name}')
        self.say('들고 있던 것을 먼저 내려놓을게요')
        self.place(item, list(cfg['pos'][:2]), slow=True)
        return True

    def clear_slot(self):
        st = self.layout['clear_station']
        slots = st['slots']
        k = self.clear_count % len(slots)
        self.clear_count += 1
        return slots[k]

    # ------------------------------------------------------------------ 인지
    def _call_detect(self, target):
        req = SrvDepthPosition.Request()
        req.target = target
        res = self.call(self.det_cli, req, timeout=20.0)
        return list(res.depth_position) if res else []

    def detect_one(self, target):
        if self.dr is None and self.g('dry_fake_detections'):
            dets = self._fake_detections([target])
            if dets:
                return dets[0]['xyz']
        for i in range(int(self.g('detect_retries'))):
            cam = self._call_detect(target)
            if len(cam) >= 3 and sum(cam[:3]) != 0:
                b = transform_to_base(cam[:3], self.T_g2c, self.posx())
                self.get_logger().info(f"'{target}' 베이스 {np.round(b, 1).tolist()}")
                return b
            self.get_logger().warn(f"'{target}' 못 찾음 ({i + 1})")
            time.sleep(0.5)
        raise TaskFailed(f"'{target}' 탐지 실패")

    def detect_all(self, classes):
        """→ [{'cls', 'xyz'(base), 'score', 'fill'}]. 관찰 자세(table_view)에서 멈춘 상태로 호출."""
        if self.dr is None and self.g('dry_fake_detections'):
            return self._fake_detections(classes)
        flat = self._call_detect('all:' + ','.join(classes))
        cur = self.posx()
        out = []
        for i in range(0, len(flat) - 5, 6):
            x, y, z, s, k, fill = flat[i:i + 6]
            out.append({'cls': classes[int(k)], 'xyz': transform_to_base([x, y, z], self.T_g2c, cur),
                        'score': s, 'fill': fill})
        return out

    def _fake_detections(self, classes):
        """dry_run 테스트용: waypoints.yaml 의 dry_detections 를 탐지 결과처럼 돌려줌 (치운 건 제외)."""
        out = []
        for d in self.wp.get('dry_detections', []):
            if d['cls'] in classes and d.get('id') not in self.dry_removed:
                out.append({'cls': d['cls'], 'xyz': np.array(d['xyz'], float), 'score': 0.9,
                            'fill': d.get('fill', -1), 'id': d.get('id')})
        return out

    def observe_table(self):
        """관찰 자세 (테이블 위 약 50~70cm, 손목 RealSense 가 식탁·손을 내려다봄). 이미 그 자세면 이동 생략."""
        if not self.at_observe:
            self.movej(self.wp['table_view_j'])
            time.sleep(float(self.g('settle_sec')))
        self.at_observe = True

    def user_front(self):
        """사용자 앞 위치: 최근 손바닥(웹캠 테이블 좌표)이 있으면 그 앞, 없으면 table_layout.user_front"""
        return list(self.layout.get('user_front', [550, 0]))

    def near_hand_spot(self, offset_mm):
        """손 옆 offset_mm (테이블 위). 손 → 로봇 베이스 방향으로 offset 만큼 (손에 닿지 않게)."""
        if self.palm is None or time.monotonic() - self.palm[0] > 2.0:
            self.get_logger().warn('최근 손 위치 없음 → user_front 사용')
            return self.user_front()
        h = np.array(self.palm[1][:2])
        d = -h / (np.linalg.norm(h) + 1e-6)
        spot = h + d * offset_mm
        lo, hi = np.array(self.wp['workspace']['min'][:2]), np.array(self.wp['workspace']['max'][:2])
        return list(np.clip(spot, lo, hi))

    @staticmethod
    def args_of(req):
        try:
            return json.loads(req.args or '{}')
        except ValueError:
            return {}
