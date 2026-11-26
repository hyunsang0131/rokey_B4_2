"""decision_node: 장면을 보고 로봇이 먼저 제안하고, 사용자 반응으로 '언제 무엇을 제안할지' 배운다.

1) 상황 판단 (규칙 점수)
   /assist/scene (task_manager scan_scene 결과: 그릇·컵 채움, 얼룩) + /user/hand_table (손 움직임) + 식사 경과 시간
   → 행동별 규칙 점수 rule(a) ∈ [0,1]
       offer_water   : 컵이 비었음          → pour_water   "물 더 드릴까요?"
       offer_clear   : 빈 그릇 있음 + 손 쉼  → clear_empty  "빈 그릇 치울까요?"
       offer_wipe    : 얼룩 있음            → wipe_table   "테이블 닦을까요?"
       offer_tidy    : 식사 오래됨 + 손 오래 쉼 + 빈 그릇 많음 → clear_all "다 드셨으면 정리할까요?"
2) 스스로 배우기 (문맥 밴딧, LinUCB — 행동마다 A, b)
   x = 문맥 특징, score(a) = w_rule * rule(a) + θ_aᵀx + α·sqrt(xᵀA_a⁻¹x)
   가장 높은 점수가 offer_threshold 이상이고 쿨다운이 지났으면 제안
   반응: 엄지척/"네" = +1 (작업 요청), 손등 밀기/"아니" = -1, 무응답 = no_answer_reward
   A_a += xxᵀ, b_a += r·x  →  ~/data/assist_arm/decision/<user_id>.npz 에 사람별 저장
3) 표현 모션: 제안할 때 대상 쪽 바라보기(look), 승낙 끄덕(nod), 거절 물러나기(retreat) — task_manager 'express'
4) claude_planner_node 의 /assist/suggest (확신 낮음 / API 실패) 도 같은 '제안 → 확인' 흐름으로 처리
"""
import json
import os
import time

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from assist_arm_msgs.msg import Gesture, HandTable, TaskRequest
from assist_arm.common.datadb import DataDB
from assist_arm.common.heartbeat import attach_heartbeat

ACTIONS = {   # 행동 → (작업, 안내 문구)
    'offer_water': ('pour_water', '물 더 드릴까요?'),
    'offer_clear': ('clear_empty', '빈 그릇 치울까요?'),
    'offer_wipe': ('wipe_table', '테이블에 얼룩이 있어요. 닦을까요?'),
    'offer_tidy': ('clear_all', '다 드셨으면 정리할까요?'),
}
TASK_TO_ACTION = {v[0]: k for k, v in ACTIONS.items()}
YES_WORDS = ['네', '응', '좋아', '그래', '해줘', '부탁']
NO_WORDS = ['아니', '싫어', '괜찮아', '됐어', '나중에']
FEATURES = ['bias', 'cup_empty', 'n_empty', 'spill', 'hand_idle', 'meal_elapsed', 'since_offer', 'recent_reject',
            'drinking']   # drinking: 최근 hold_cup(실제 컵을 쥔 손) 제스처 — 명령이 아니라 '마시는 중' 상태


class LinUCB:
    def __init__(self, actions, d, alpha):
        self.actions, self.d, self.alpha = list(actions), d, alpha
        self.A = {a: np.eye(d) for a in self.actions}
        self.b = {a: np.zeros(d) for a in self.actions}
        self.n = {a: 0 for a in self.actions}

    def score(self, a, x):
        Ainv = np.linalg.inv(self.A[a])
        theta = Ainv @ self.b[a]
        return float(theta @ x), float(self.alpha * np.sqrt(x @ Ainv @ x))

    def update(self, a, x, r):
        self.A[a] += np.outer(x, x)
        self.b[a] += r * x
        self.n[a] += 1

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, actions=np.array(self.actions), A=np.stack([self.A[a] for a in self.actions]),
                 b=np.stack([self.b[a] for a in self.actions]), n=np.array([self.n[a] for a in self.actions]))

    def load(self, path):
        if not os.path.exists(path):
            return False
        d = np.load(path, allow_pickle=False)
        for i, a in enumerate(d['actions']):
            a = str(a)
            if a in self.A and d['A'][i].shape == (self.d, self.d):
                self.A[a], self.b[a], self.n[a] = d['A'][i], d['b'][i], int(d['n'][i])
        return True


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')
        p = self.declare_parameter
        p('user_id', 'default')
        p('store_dir', '~/data/assist_arm/decision')
        p('enabled_actions', list(ACTIONS))
        p('alpha', 0.6)                   # 탐험 정도
        p('rule_weight', 1.0)
        p('offer_threshold', 0.7)
        p('offer_cooldown_sec', 60.0)      # 제안 사이 최소 간격
        p('same_offer_cooldown_sec', 180.0)
        p('response_window_sec', 8.0)
        p('no_answer_reward', -0.3)
        p('scan_period_sec', 90.0)         # 한가할 때 장면 관찰 요청 간격 (0 이면 끔)
        p('scene_max_age_sec', 120.0)
        p('hand_still_mm', 20.0)
        p('hand_idle_full_sec', 60.0)
        p('meal_full_min', 30.0)
        p('cup_empty_th', 0.2)
        p('dish_empty_th', 0.3)
        p('use_express', True)
        p('use_db', True)                  # ~/data/assist_arm/assist_arm.db 에 제안·반응 기록 (실패해도 동작은 그대로)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.bandit = LinUCB(g('enabled_actions'), len(FEATURES), float(g('alpha')))
        self.db = DataDB(quiet=False) if g('use_db') else None
        self._load_user(g('user_id'))
        self.scene, self.scene_t = None, 0.0
        self.palm, self.last_move_t = None, time.monotonic()
        self.meal_start = time.monotonic()
        self.last_offer_t, self.last_offer_by = -1e9, {}
        self.recent_reject_t = -1e9
        self.drink_t, self.drink_start = -1e9, None    # hold_cup 이 마지막으로 보인 시각 / 이번에 마시기 시작한 시각
        self.last_scan_req = time.monotonic()
        self.task_busy = False
        self.queue, self.last_send_t = [], -1e9   # task_manager 로 보낼 요청 (하나씩, 한가할 때)
        self.offer = None        # {'action', 'x', 't', 'point', 'reason'}
        self.req_pub = self.create_publisher(TaskRequest, '/assist/request', 10)
        self.say_pub = self.create_publisher(String, '/assist/announce', 10)
        self.active_pub = self.create_publisher(Bool, '/assist/offer_active', 10)
        self.dbg_pub = self.create_publisher(String, '/assist/decision', 10)
        self.create_subscription(String, '/assist/scene', self.on_scene, 10)
        self.create_subscription(HandTable, '/user/hand_table', self.on_hand, 10)
        self.create_subscription(Gesture, '/user/gesture', self.on_gesture, 10)
        self.create_subscription(String, '/voice/command', self.on_voice, 10)
        self.create_subscription(String, '/assist/request_text', self.on_voice, 10)
        self.create_subscription(String, '/assist/state', self.on_state, 10)
        self.create_subscription(String, '/assist/suggest', self.on_suggest, 10)
        self.create_subscription(String, '/assist/user', lambda m: self._load_user(m.data), 10)
        self.create_timer(0.5, self.tick)
        attach_heartbeat(self)

    # ---------------------------------------------------------------- 사람별 저장
    def _path(self, user):
        return os.path.join(os.path.expanduser(self.g('store_dir')), f'{user}.npz')

    def _load_user(self, user):
        self.user = user or 'default'
        self.bandit = LinUCB(self.g('enabled_actions'), len(FEATURES), float(self.g('alpha')))
        ok = self.bandit.load(self._path(self.user))
        self.get_logger().info(f"사용자 '{self.user}' 선호 {'불러옴' if ok else '새로 시작'}")

    # ---------------------------------------------------------------- 입력
    def on_scene(self, m):
        try:
            self.scene, self.scene_t = json.loads(m.data), time.monotonic()
        except ValueError:
            pass

    def on_hand(self, m):
        if not m.valid:
            return
        p = np.array([m.palm.x, m.palm.y])
        if self.palm is None or np.linalg.norm(p - self.palm) > self.g('hand_still_mm'):
            self.last_move_t = time.monotonic()
            self.palm = p

    def on_state(self, m):
        s = m.data.split(' ', 1)[0]
        self.task_busy = s not in ('IDLE', 'HOME', 'OBSERVING')
        if s == 'SEAT_SETTING':
            self.meal_start = time.monotonic()     # 상 차리면 식사 시작으로 봄

    def on_gesture(self, m):
        if m.label == 'hold_cup':
            now = time.monotonic()
            if now - self.drink_t > 5.0:
                self.drink_start = now
            self.drink_t = now
            return
        if self.offer is None or not m.confirmed:
            return
        if m.label == 'thumbs_up':
            self.answer(+1.0, 'thumbs_up')
        elif m.label == 'push':
            self.answer(-1.0, 'push')

    def on_voice(self, m):
        if self.offer is None:
            return
        t = m.data.replace(' ', '')
        if any(w in t for w in NO_WORDS):
            self.answer(-1.0, m.data)
        elif any(w in t for w in YES_WORDS):
            self.answer(+1.0, m.data)

    def on_suggest(self, m):
        """claude_planner_node: 확신 낮음 / API 실패 → 바로 실행하지 않고 '제안 → 확인'"""
        try:
            d = json.loads(m.data)
        except ValueError:
            return
        a = TASK_TO_ACTION.get(d.get('task'))
        if a is None or self.offer is not None or self.task_busy:
            return
        self.make_offer(a, self.features(), reason=d.get('reason', 'planner'), say=d.get('say'))

    # ---------------------------------------------------------------- 특징 / 규칙
    def features(self):
        now = time.monotonic()
        sc = self.scene if (self.scene and now - self.scene_t < self.g('scene_max_age_sec')) else {}
        objs = sc.get('objects', [])
        cups = [o for o in objs if o['cls'] == 'cup' and o.get('fill', -1) >= 0]
        cup_empty = 1.0 if cups and min(o['fill'] for o in cups) < self.g('cup_empty_th') else 0.0
        dishes = [o for o in objs if o['cls'] != 'cup' and 0 <= o.get('fill', -1) < self.g('dish_empty_th')]
        spill = 1.0 if sc.get('spill', {}).get('detected') else 0.0
        idle = min(1.0, (now - self.last_move_t) / self.g('hand_idle_full_sec'))
        meal = min(1.0, (now - self.meal_start) / 60.0 / self.g('meal_full_min'))
        since = min(1.0, (now - self.last_offer_t) / 600.0)
        rej = 1.0 if now - self.recent_reject_t < 300.0 else 0.0
        # 마시는 중(최근 5초 안 hold_cup) = 1, 마시고 내려놓은 직후 1분 동안 0.5 → 0 으로 줄어듦
        dt = now - self.drink_t
        drinking = 1.0 if dt < 5.0 else max(0.0, 0.5 * (1 - (dt - 5.0) / 60.0))
        return np.array([1.0, cup_empty, min(1.0, len(dishes) / 3.0), spill, idle, meal, since, rej, drinking])

    @staticmethod
    def rule_scores(x):
        _, cup_empty, n_empty, spill, idle, meal, _, rej, drinking = x
        if drinking >= 1.0:          # 컵을 들고 마시는 중이면 아무것도 제안하지 않음 (방해 금지)
            return {a: 0.0 for a in ('offer_water', 'offer_clear', 'offer_wipe', 'offer_tidy')}
        # 마시고 내려놓은 직후엔 컵이 비었을 가능성이 커서 물 제안 점수를 조금 올림 (컵 채움 판정 보조)
        r = {'offer_water': 0.8 * cup_empty + 0.25 * drinking,
             'offer_clear': (0.5 + 0.4 * idle) * (n_empty > 0) * min(1.0, n_empty * 3),
             'offer_wipe': 0.85 * spill,
             'offer_tidy': 0.9 * meal * idle * (n_empty >= 0.66)}
        return {a: v * (0.7 if rej else 1.0) for a, v in r.items()}

    # ---------------------------------------------------------------- 제안 / 반응
    def make_offer(self, action, x, reason='', say=None):
        task, text = ACTIONS[action]
        point = None
        if self.scene and action in ('offer_water', 'offer_clear', 'offer_wipe'):
            objs = self.scene.get('objects', [])
            if action == 'offer_water':
                cand = [o for o in objs if o['cls'] == 'cup']
            elif action == 'offer_clear':
                cand = [o for o in objs if o['cls'] != 'cup' and 0 <= o.get('fill', -1) < self.g('dish_empty_th')]
            else:
                sp = self.scene.get('spill', {})
                cand = [{'xy': sp.get('position', [0, 0])[:2]}] if sp.get('position_valid') else []
            point = cand[0]['xy'] if cand else None
        self.offer = {'action': action, 'x': x, 't': time.monotonic(), 'point': point, 'reason': reason}
        self.last_offer_t = self.last_offer_by[action] = time.monotonic()
        self.active_pub.publish(Bool(data=True))
        if self.g('use_express'):
            self.request('express', {'motion': 'look', 'point': point})
        self.say(say or text)
        self.publish_debug('offer', action=action, reason=reason)

    def answer(self, r, raw):
        o, self.offer = self.offer, None
        self.active_pub.publish(Bool(data=False))
        self.bandit.update(o['action'], o['x'], r)
        self.bandit.save(self._path(self.user))
        if self.db is not None:
            resp = 'accept' if r > 0 else ('ignore' if raw == 'no_answer' else 'reject')
            ctx = {'user': self.user, 'features': dict(zip(FEATURES, map(float, o['x']))), 'reason': o.get('reason', ''),
                   'point': o.get('point'), 'answer_raw': raw}
            self.db.add_decision_event(ctx, o['action'], True, resp, float(r),
                                       model_ver=f"linucb_a{self.bandit.alpha}_d{self.bandit.d}:{self.user}")
        self.publish_debug('answer', action=o['action'], reward=r, raw=raw)
        if r > 0:
            if self.g('use_express'):
                self.request('express', {'motion': 'nod'})
            self.request(ACTIONS[o['action']][0], {})
            self.say('네, 할게요')
        else:
            self.recent_reject_t = time.monotonic()
            if self.g('use_express'):
                self.request('express', {'motion': 'retreat'})
            self.say('알겠어요')

    def request(self, task, args):
        """task_manager 는 한 번에 하나만 받으므로 큐에 넣고 한가할 때 하나씩 보낸다."""
        m = TaskRequest(task=task, args=json.dumps(args, ensure_ascii=False), source='decision', raw='')
        self.queue.append(m)
        self._flush()

    def _flush(self):
        now = time.monotonic()
        if self.queue and not self.task_busy and now - self.last_send_t > 1.5:
            m = self.queue.pop(0)
            m.header.stamp = self.get_clock().now().to_msg()
            self.req_pub.publish(m)
            self.last_send_t = now

    def say(self, text):
        self.say_pub.publish(String(data=text))
        self.get_logger().info(f'[제안] {text}')

    def publish_debug(self, kind, **kw):
        d = {'kind': kind, 'user': self.user, **kw}
        self.dbg_pub.publish(String(data=json.dumps(d, ensure_ascii=False, default=float)))
        self.get_logger().info(f'{kind}: {kw}')

    # ---------------------------------------------------------------- 주기
    def tick(self):
        now = time.monotonic()
        self._flush()
        if self.offer is not None:
            if now - self.offer['t'] > self.g('response_window_sec'):
                self.answer(float(self.g('no_answer_reward')), 'no_answer')
            return
        if self.task_busy or self.queue or now - self.last_send_t < 2.0:
            return
        if self.g('scan_period_sec') > 0 and now - self.last_scan_req > self.g('scan_period_sec'):
            self.last_scan_req = now
            self.request('scan_scene', {})
            return
        if now - self.last_offer_t < self.g('offer_cooldown_sec'):
            return
        x = self.features()
        rules = self.rule_scores(x)
        best, best_s, detail = None, -1e9, {}
        for a in self.bandit.actions:
            if now - self.last_offer_by.get(a, -1e9) < self.g('same_offer_cooldown_sec'):
                continue
            mean, bonus = self.bandit.score(a, x)
            s = self.g('rule_weight') * rules.get(a, 0.0) + mean + (bonus if rules.get(a, 0.0) > 0 else 0.0)
            detail[a] = round(s, 3)
            if s > best_s:
                best, best_s = a, s
        if best is not None and best_s >= self.g('offer_threshold') and rules.get(best, 0) > 0:
            self.make_offer(best, x, reason=f'score={best_s:.2f} {detail}')


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
