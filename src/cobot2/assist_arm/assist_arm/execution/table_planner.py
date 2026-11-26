"""table_planner: 세팅·정리에서 '다음에 무엇을 어디로 옮길지' 정하는 결정 모듈 (v2.2, docs/ARCHITECTURE_v2.2.md 3.5·6.7).

ROS 에 의존하지 않는 순수 파이썬이라 로봇·카메라 없이 단위 시험과 RL 학습 환경에서 그대로 쓴다.
task_manager(skills.py) 가 관찰 → next_action() → 실행 → 다시 관찰을 반복한다.

결정 방식
  - 기본은 규칙 기반 (rule): 지금 코드처럼 '로봇에 가까운 것부터', 목적지는 종류별 보관 자리
  - RL 정책은 policy(obs, candidates) → 후보 번호 를 넘기면 같은 자리에서 바뀐다 (후보 = 가능한 행동만, 마스킹 효과)

반환 PlanResult(status, action, reason)
  ACT              action 하나 실행
  DONE             대상이 하나도 없고, 사라진 물체는 모두 집기 성공 기록이 있음
  BLOCKED          대상은 남았지만 가능한 행동이 없음 (작업 범위 밖, 보관 자리 가득, 재고 없음 …)
  PERCEPTION_FAIL  검출 실패, 또는 집은 기록 없이 사라진 물체가 있음 (task_manager 가 재관측 후 다시 부름)

좌표 단위: mm (두산 베이스 좌표, docs 1.1)
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

ACT, DONE, BLOCKED, PERCEPTION_FAIL = 'ACT', 'DONE', 'BLOCKED', 'PERCEPTION_FAIL'
EMPTY, EATING, SEAT_DONE = 'EMPTY', 'EATING', 'DONE'

BOWL_TYPES = ('rice_bowl', 'soup_bowl')
UNKNOWN_BOWL = 'bowl'          # YOLO 가 종류를 모를 때의 클래스 (위치로 판정, 안 되면 임시 자리)


@dataclass
class Action:
    item: str                  # 판정된 종류 (rice_bowl, soup_bowl, plate, spoon, fork, cup …)
    src: str                   # 'seat1' / 'table' / 보관 자리 이름 (stack_B, holder_spoon …)
    src_xy: list               # 집을 위치 [x, y] (mm)
    dst: str                   # 보관 자리 이름 또는 'seat1:rice_bowl' 같은 세팅 자리
    dst_xy: list               # 놓을 위치 [x, y] (mm)
    dst_site: str              # 'stack' | 'holder' | 'temp' | 'table'
    dst_count: int = 0         # 놓기 전 목적지에 있는 개수 (포개기 높이 계산용)
    dst_top_type: str = ''     # 놓기 전 목적지 맨 위 종류 ('' = 비어 있음)
    det: Optional[dict] = None  # 정리 때 원래 검출 결과


@dataclass
class PlanResult:
    status: str
    action: Optional[Action] = None
    reason: str = ''


@dataclass
class History:
    """한 작업(episode) 동안의 기록. task_manager 가 동작마다 갱신한다."""
    picked: list = field(default_factory=list)      # 집기에 성공한 위치 [(종류, [x, y])]
    placed: set = field(default_factory=set)        # 세팅에서 놓은 (자리 번호, 종류)
    prev_dets: list = field(default_factory=list)   # 직전 관찰의 검출 결과


def seat_frame(seat_xy):
    """자리 좌표계 (skills.seat_frame 과 같음): u = 자리 → 로봇 베이스 방향, v = u 를 왼쪽으로 90°"""
    s = np.array(seat_xy, float)
    u = -s / (np.linalg.norm(s) + 1e-6)
    return s, u, np.array([-u[1], u[0]])


def seat_slots(layout, seat_xy):
    """자리 템플릿 → [(종류, [x, y])]  (table_layout.seat_layout 의 from: rack 항목)"""
    s, u, v = seat_frame(seat_xy)
    out = []
    for it in layout.get('seat_layout', []):
        if it.get('from', 'rack') != 'rack':
            continue
        off = it.get('offset', [0.0, 0.0])
        out.append((it.get('rack', it['item']), list(s + off[0] * u + off[1] * v)))
    return out


def _xy(det):
    return np.array(det['xyz'][:2], float)


class TablePlanner:
    def __init__(self, layout, workspace=None, policy: Optional[Callable] = None):
        """layout: table_layout.yaml 내용, workspace: waypoints.yaml 의 workspace {min, max} (mm)"""
        self.layout = layout
        self.storage = layout.get('storage', {})
        self.radius = float(layout.get('type_match_radius_mm', 60.0))
        default_classes = list(BOWL_TYPES) + [UNKNOWN_BOWL, 'plate', 'spoon', 'fork']
        self.clear_classes = list(layout.get('clear_classes', default_classes))
        self.workspace = workspace
        self.policy = policy
        self.vanish_mm = 50.0

    # ------------------------------------------------------------------ 보조
    def reachable(self, xy):
        if not self.workspace:
            return True
        lo, hi = np.array(self.workspace['min'][:2], float), np.array(self.workspace['max'][:2], float)
        p = np.array(xy[:2], float)
        return bool(np.all(p >= lo) and np.all(p <= hi))

    def nearest_seat(self, xy, seats):
        if not seats:
            return None
        k = min(seats, key=lambda k: np.linalg.norm(np.array(seats[k]['xy'][:2], float) - xy))
        return k

    def classify(self, det, seats):
        """검출 클래스 → 종류. 'bowl' 이면 가장 가까운 자리 템플릿 위치(허용 반경 안)의 종류로 (docs 1.2)"""
        cls = det['cls']
        if cls != UNKNOWN_BOWL:
            return cls
        p, best, best_d = _xy(det), UNKNOWN_BOWL, self.radius
        for k in seats or {}:
            for item, xy in seat_slots(self.layout, seats[k]['xy']):
                if item in BOWL_TYPES:
                    d = float(np.linalg.norm(np.array(xy) - p))
                    if d <= best_d:
                        best, best_d = item, d
        return best

    def _places(self, kind):
        return self.storage.get(kind, {}) or {}

    def location(self, name):
        """보관 자리 이름 → 설정 (stacks / holders / temp_slots 어디든). 없으면 {}"""
        for kind in ('stacks', 'holders', 'temp_slots'):
            if name in self._places(kind):
                return self._places(kind)[name]
        return {}

    def storable(self):
        """보관 자리(더미·수저통)가 설정된 종류. 세팅은 이 종류만 맡는다 (예: 젓가락은 범위 밖 → 건너뜀)"""
        return {c.get('type') for kind in ('stacks', 'holders') for c in self._places(kind).values()}

    def clear_destination(self, item, storage):
        """정리할 종류 → (자리 이름, 사이트, 위치, 개수, 맨 위 종류) 또는 None (넣을 곳 없음)"""
        for site, kind in (('stack', 'stacks'), ('holder', 'holders')):
            for name, cfg in self._places(kind).items():
                if cfg.get('type') != item:
                    continue
                inv = storage.get(name, [])
                if len(inv) < int(cfg.get('max', 4)):
                    return name, site, list(cfg['pos'][:2]), len(inv), (inv[-1] if inv else '')
        for name, cfg in self._places('temp_slots').items():       # 종류 모름·가득 참 → 비어 있는 임시 자리
            if not storage.get(name):
                return name, 'temp', list(cfg['pos'][:2]), 0, ''
        return None

    def vanished(self, dets, history):
        """직전 관찰에 있었는데 지금 없는 물체 중 집기 기록이 없는 것"""
        out = []
        for p in history.prev_dets:
            if p['cls'] not in self.clear_classes:
                continue
            q = _xy(p)
            if any(d['cls'] == p['cls'] and np.linalg.norm(_xy(d) - q) < self.vanish_mm for d in dets):
                continue
            if any(c == p['cls'] or (p['cls'] == UNKNOWN_BOWL and c in BOWL_TYPES)
                   for c, xy in history.picked if np.linalg.norm(np.array(xy[:2]) - q) < self.vanish_mm):
                continue
            out.append(p)
        return out

    def _choose(self, cands, obs):
        """후보 중 하나 고르기: RL 정책이 있으면 정책, 없으면 규칙(로봇에 가까운 것부터)"""
        if self.policy is not None:
            i = int(self.policy(obs, cands))
            if 0 <= i < len(cands):
                return cands[i]
        return min(cands, key=lambda a: float(np.linalg.norm(np.array(a.src_xy[:2]))))

    # ------------------------------------------------------------------ 결정
    def next_action(self, dets, storage, seats=None, holding=None, mode='clear', history=None) -> PlanResult:
        """dets: detect_all() 결과 [{cls, xyz(mm), ...}] 또는 None(검출 실패)
        storage: {보관 자리 이름: [아래 → 위 종류]} (DB inventory)
        seats: {자리 번호: {'xy': [x, y], 'state': EMPTY|EATING|DONE}}
        holding: 들고 있는 종류 또는 None — 들고 있으면 결정하지 않음 (task_manager 가 6.4 복구 먼저)"""
        history = history or History()
        seats = seats or {}
        storage = storage or {}
        if holding:
            return PlanResult(BLOCKED, reason=f'holding:{holding} (복구 먼저)')
        if dets is None:
            return PlanResult(PERCEPTION_FAIL, reason='detect_failed')
        if mode == 'clear':
            return self._next_clear(dets, storage, seats, history)
        if mode == 'set_table':
            return self._next_set(storage, seats, history)
        raise ValueError(f'모르는 mode: {mode}')

    def _next_clear(self, dets, storage, seats, history):
        lost = self.vanished(dets, history)
        if lost:
            return PlanResult(PERCEPTION_FAIL, reason='vanished:' + ','.join(d['cls'] for d in lost))
        targets = []
        for d in dets:
            if d['cls'] not in self.clear_classes:
                continue
            k = self.nearest_seat(_xy(d), seats)
            if k is not None and seats[k].get('state') == EATING:      # 먹는 중인 자리는 건드리지 않음
                continue
            targets.append((d, k))
        if not targets:
            return PlanResult(DONE)
        cands, why = [], []
        for d, k in targets:
            item = self.classify(d, seats)
            if not self.reachable(_xy(d)):
                why.append(f'{item}:out_of_workspace')
                continue
            dst = self.clear_destination(item, storage)
            if dst is None:
                why.append(f'{item}:no_space')
                continue
            name, site, xy, n, top = dst
            cands.append(Action(item=item, src=f'seat{k}' if k is not None else 'table', src_xy=list(_xy(d)),
                                dst=name, dst_xy=xy, dst_site=site, dst_count=n, dst_top_type=top, det=d))
        if not cands:
            return PlanResult(BLOCKED, reason=';'.join(why) or 'no_action')
        return PlanResult(ACT, self._choose(cands, {'mode': 'clear', 'dets': dets, 'storage': storage}))

    def _next_set(self, storage, seats, history):
        # 먼 자리부터 놓는다 (놓인 그릇 위로 팔이 지나가지 않게, docs 7장)
        order = sorted(seats, key=lambda k: -float(np.linalg.norm(np.array(seats[k]['xy'][:2], float))))
        ok = self.storable()
        need = [(k, item, xy) for k in order for item, xy in seat_slots(self.layout, seats[k]['xy'])
                if item in ok and (k, item) not in history.placed]
        if not need:
            return PlanResult(DONE)
        cands, why = [], []
        for k, item, xy in need:
            src = next(((n, c) for kind in ('stacks', 'holders') for n, c in self._places(kind).items()
                        if c.get('type') == item and storage.get(n)), None)
            if src is None:
                why.append(f'seat{k}:{item}:out_of_stock')
                continue
            if not self.reachable(xy):
                why.append(f'seat{k}:{item}:out_of_workspace')
                continue
            name, cfg = src
            cands.append(Action(item=item, src=name, src_xy=list(cfg['pos'][:2]), dst=f'seat{k}:{item}',
                                dst_xy=list(xy), dst_site='table'))
        if not cands:
            return PlanResult(BLOCKED, reason=';'.join(why))
        if self.policy is None:          # 규칙: 먼 자리 순서를 지킨다 (need 순서의 첫 가능한 것)
            return PlanResult(ACT, cands[0])
        return PlanResult(ACT, self._choose(cands, {'mode': 'set_table', 'storage': storage}))
