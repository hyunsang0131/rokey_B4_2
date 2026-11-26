"""실행기 (v2.2 6.7·7·8장): 관찰 → table_planner 결정 → 스킬 실행 → DB 기록 → 반복.

skills.py 의 정리(clear_dish / clear_empty / clear_all)·세팅(seat_setting)이 run_plan() 을 부른다.
ROS 를 직접 쓰지 않고 tm(TaskManager) 의 메서드만 쓰므로, 가짜 tm 으로 단위 시험할 수 있다.

tm 에 필요한 것
  observe_table(), detect_all(classes), pick(item, xyz, z_offset=0), place(item, xy, slow=False, z_offset=0),
  recover_holding(), say(text), state_note(s, extra=''), check(), g(name), holding, layout,
  planner(TablePlanner), db(DataDB), dry_removed(set)
예외
  TaskFailed      한 동작 실패 → task_log 에 실패 기록, 다시 관찰해서 계속 (연속 실패가 많으면 중단)
  SafetyAbort / TaskCancelled → episode 를 ABORTED 로 닫고 그대로 위로 던짐 (task_manager 가 처리)
"""
import os
import time

import numpy as np
import yaml

from assist_arm.execution.robot_io import SafetyAbort, TaskCancelled, TaskFailed
from assist_arm.execution.table_planner import (ACT, BLOCKED, DONE, EATING, PERCEPTION_FAIL, SEAT_DONE,
                                                History)

REASON_KO = {'out_of_workspace': '손이 닿지 않아', 'no_space': '놓을 자리가 없어',
             'out_of_stock': '그릇이 모자라', 'holding': '들고 있는 것이 있어'}


def load_seats(tm, done=None):
    """자리 파일(seat_setting 이 저장) → {번호: {'xy', 'state'}}. done 에 있는 번호만 DONE, 나머지 EATING.
    done 이 None 이면 모든 자리 DONE (사람이 '치워' 라고 했으므로)."""
    path = os.path.expanduser(tm.g('seats_file'))
    try:
        with open(path) as f:
            pts = (yaml.safe_load(f) or {}).get('seats', [])
    except OSError:
        pts = []
    out = {}
    for i, xy in enumerate(pts, start=1):
        state = SEAT_DONE if done is None or i in done else EATING
        out[i] = {'xy': [float(v) for v in xy[:2]], 'state': state}
    return out


def _error_code(e):
    msg = str(e)
    if '탐지' in msg or '못 찾' in msg:
        return 'PERCEPTION_FAIL'
    if '1206' in msg or 'workspace' in msg.lower():
        return 'OUT_OF_WORKSPACE'
    return 'TASK_FAILED'


def _stack_height(tm, item):
    return float((tm.layout.get('storage', {}) or {}).get('stack_height_mm', {}).get(item, 15.0))


def _blocked_text(reason, mode):
    kinds = {r.split(':')[-1] for r in reason.split(';') if r}
    why = ', '.join(sorted({REASON_KO.get(k, k) for k in kinds}))
    what = '정리' if mode == 'clear' else '세팅'
    return f'일부는 {why} {what}하지 못했어요' if why else f'더 {what}할 수 없어요'


class Executor:
    def __init__(self, tm, mode, episode_id, max_fail=3):
        self.tm, self.mode, self.ep = tm, mode, episode_id
        self.history = History()
        self.n_ok, self.n_fail, self.max_fail = 0, 0, max_fail

    # ------------------------------------------------------------------ 기록 보조
    def _log(self, skill, a, dst='', success=True, code='OK', t0=None):
        self.tm.db.log_task(self.ep, skill, a.item, a.src, dst, a.dst_top_type if dst else '', success, code,
                            None if t0 is None else round(time.monotonic() - t0, 2))

    # ------------------------------------------------------------------ 한 동작 = 집기 + 놓기
    def execute(self, a):
        tm = self.tm
        tm.state_note(self.mode.upper(), f'{a.item}: {a.src} → {a.dst}')
        # 집기
        t0 = time.monotonic()
        if a.det is not None:                          # 테이블 위 물체 (정리)
            xyz = np.array(a.det['xyz'], float)
            z_off = 0.0
        else:                                          # 보관 자리 맨 위 (세팅)
            xyz = np.array(list(a.src_xy) + [0.0], float)
            z_off = max(0, len(tm.db.inventory(a.src)) - 1) * _stack_height(tm, a.item)
        try:
            tm.pick(a.item, xyz, z_offset=z_off)
        except TaskFailed as e:
            self._log('pick', a, success=False, code=_error_code(e), t0=t0)
            raise
        self._log('pick', a, t0=t0)
        self.history.picked.append((a.item, list(xyz[:2])))
        if a.det is not None and a.det.get('id') is not None:
            tm.dry_removed.add(a.det['id'])             # dry_run 가짜 검출에서 치운 것 빼기
        if a.det is None:
            tm.db.inventory_pop(a.src)
        # 놓기
        t1 = time.monotonic()
        skill = {'stack': 'place_stack', 'holder': 'place_holder'}.get(a.dst_site, 'place')
        z_off = a.dst_count * _stack_height(tm, a.item) if a.dst_site == 'stack' else 0.0
        try:
            tm.place(a.item, list(a.dst_xy), z_offset=z_off)
        except TaskFailed as e:
            self._log(skill, a, a.dst, success=False, code=_error_code(e), t0=t1)
            raise
        self._log(skill, a, a.dst, t0=t1)
        if a.dst_site in ('stack', 'holder', 'temp'):
            tm.db.inventory_push(a.dst, a.item)
        else:                                          # 세팅: '자리번호:종류'
            seat = int(a.dst.split(':')[0].replace('seat', ''))
            self.history.placed.add((seat, a.item))

    # ------------------------------------------------------------------ 반복
    def run(self, seats, target_xy=None, radius=120.0, max_actions=None, max_steps=30):
        tm, planner = self.tm, self.tm.planner
        recheck = 0
        for _ in range(max_steps):
            tm.check()
            if tm.holding:                             # 6.4 들고 있으면 복구 먼저
                tm.recover_holding()
            dets = None
            if self.mode == 'clear':
                tm.observe_table()
                try:
                    dets = tm.detect_all(planner.clear_classes)
                except TaskFailed:
                    dets = None
                if dets is not None and target_xy is not None:      # 콕: 가리킨 물체만
                    p = np.array(target_xy[:2], float)
                    near = [d for d in dets if np.linalg.norm(np.array(d['xyz'][:2]) - p) <= radius]
                    dets = sorted(near, key=lambda d: np.linalg.norm(np.array(d['xyz'][:2]) - p))[:1]
                    if not dets and self.n_ok == 0:
                        return self._finish(BLOCKED, 'point:no_object', '가리킨 곳 근처에 그릇이 없어요')
            res = planner.next_action(dets if self.mode == 'clear' else [], tm.db.inventory(), seats=seats,
                                      holding=tm.holding, mode=self.mode, history=self.history)
            if dets is not None:
                self.history.prev_dets = [dict(d) for d in dets]
            if res.status == ACT:
                recheck = 0
                try:
                    self.execute(res.action)
                    self.n_ok += 1
                except TaskFailed as e:
                    self.n_fail += 1
                    tm.say('잘 안 됐어요. 다시 볼게요')
                    if self.n_fail >= self.max_fail:
                        return self._finish(BLOCKED, f'too_many_failures:{e}', '여러 번 실패해서 멈출게요')
                if max_actions is not None and self.n_ok >= max_actions:
                    return self._finish(DONE, '', '치웠어요' if self.mode == 'clear' else '놓았어요')
                continue
            if res.status == PERCEPTION_FAIL and recheck < 2:
                recheck += 1
                tm.at_observe = False                 # 다시 관찰 (6.7)
                continue
            return self._finish(res.status, res.reason)
        return self._finish(BLOCKED, 'max_steps', '너무 오래 걸려서 멈출게요')

    def _finish(self, status, reason='', text=None):
        if text is None:
            text = {DONE: '다 치웠어요' if self.mode == 'clear' else '세팅 다 했어요',
                    PERCEPTION_FAIL: '그릇이 잘 안 보여요. 확인해 주세요'}.get(status) or \
                _blocked_text(reason, self.mode)
        self.tm.say(text)
        self.tm.state_note(f'PLAN_{status}', reason)
        self.tm.db.finish_episode(self.ep, status, reason=reason, n_actions=self.n_ok)
        return status


def run_plan(tm, mode, seats=None, target_xy=None, max_actions=None):
    """정리(mode='clear') 또는 세팅(mode='set_table') 한 번. 반환: DONE / BLOCKED / PERCEPTION_FAIL"""
    planner_kind = 'rl' if tm.planner.policy is not None else 'rule'
    seats = seats if seats is not None else load_seats(tm)
    ep = tm.db.start_episode(mode, planner_kind, people=len(seats) or None)
    ex = Executor(tm, mode, ep)
    try:
        return ex.run(seats, target_xy=target_xy, radius=float(tm.g('poke_radius_mm')), max_actions=max_actions)
    except (SafetyAbort, TaskCancelled) as e:
        tm.db.finish_episode(ep, 'ABORTED', reason=type(e).__name__, n_actions=ex.n_ok)
        raise
