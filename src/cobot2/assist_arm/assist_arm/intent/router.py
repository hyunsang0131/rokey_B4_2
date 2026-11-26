"""제스처 · 음성 · 콕(poke) → 작업 요청 (/assist/request, assist_arm_msgs/TaskRequest).

입력
  /user/gesture        (confirmed 만)  stop 은 safety_monitor 가 직접 처리
  /voice/command       STT 원문 (voice_command_node)
  /user/poke           finger_pointer 의 '콕' (테이블 좌표)
  /assist/request_text 키보드/테스트용 (음성과 같은 규칙)

규칙
  음식 콕 → poke_followup_sec 동안 후속 명령 대기
      · '치워' 음성 또는 push(손등 밀기)      → clear_dish {point}
      · '줘/가져와' 음성 또는 같은 곳 다시 콕  → bring_dish {point}
      · 아무것도 없으면 poke_default (기본 bring_dish = 가리키면 사용자 앞으로)
  push (콕 대기 없음) 또는 '치워' 음성         → clear_empty (빈 그릇 스스로 찾아 정리)
  thumbs_up 또는 '다 먹었어'                   → clear_all (그릇 → 컵 → 수저)
  '물 줘' 음성 + cup_grasp 제스처 (combo_window_sec 안, 순서 무관) → pour_water
  '<음식> 줘' (table_layout.yaml foods 키워드)  → bring_dish {food}
  'N명이야' → 콕 N번(seat_poke_timeout_sec 안) → seat_setting {seats: [[x,y],...]}
  v_sign 또는 '노래/음악/춤'                   → music {mood}
  '닦아/얼룩/흘렸'                             → wipe_table
  '그만/취소'                                  → cancel
  그 밖의 말                                   → /assist/unmatched_text (claude_planner_node 가 해석)
  decision_node 제안 중(/assist/offer_active)에는 엄지척/손등 밀기를 명령으로 쓰지 않음 (승낙/거절)
"""
import json
import os
import re
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PointStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, String

from assist_arm_msgs.msg import Gesture, TaskRequest
from assist_arm.common.datadb import DataDB
from assist_arm.common.heartbeat import attach_heartbeat

KOR_NUM = {'한': 1, '하나': 1, '두': 2, '둘': 2, '세': 3, '셋': 3, '네': 4, '넷': 4, '다섯': 5, '여섯': 6}
W = {  # 음성 키워드 (공백 무시 비교)
    'cancel': ['그만', '취소', '됐어'],
    'clear_all': ['다먹었', '잘먹었', '식사끝', '다치워'],
    'clear': ['치워', '정리', '빼줘'],
    'give': ['줘', '가져와', '갖다줘', '주세요'],
    'water': ['물줘', '물좀', '물따라', '목말라'],
    'wipe': ['닦아', '얼룩', '흘렸'],
    'music': ['노래', '음악', '춤', '율동'],
}

YES_NO = ['네', '응', '좋아', '그래', '아니', '싫어', '괜찮아', '나중에']


def norm(t):
    return t.replace(' ', '')


class RequestRouter(Node):
    def __init__(self):
        super().__init__('request_router')
        p = self.declare_parameter
        p('table_layout_file', '')
        p('music_library_file', '')
        p('poke_followup_sec', 3.0)
        p('poke_default', 'bring_dish')      # bring_dish / none
        p('same_poke_mm', 60.0)
        p('combo_window_sec', 5.0)
        p('water_require_both', True)        # False 면 '물 줘' 만으로도 pour_water
        p('seat_poke_timeout_sec', 30.0)
        p('cooldown_sec', 3.0)
        p('use_db', True)                    # 음성/텍스트 요청을 DB voice_utterances 에 기록 (의도 학습 데이터)
        p('person', 'default')
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        share = get_package_share_directory('assist_arm')
        layout = self._load(g('table_layout_file') or os.path.join(share, 'config', 'table_layout.yaml'))
        self.foods = layout.get('foods', {})
        self.moods = self._load(g('music_library_file') or os.path.join(share, 'config', 'music_library.yaml')
                                ).get('moods', {})
        self.pending_poke = None     # (t, [x,y,z])
        self.water_voice_t = self.cup_gesture_t = -1e9
        self.seat_mode = None        # {'n': N, 'seats': [], 't': start}
        self.last = {}
        self.db = DataDB() if g('use_db') else None
        self.offer_active = False     # decision_node 가 제안 중이면 엄지척/손등 밀기는 '대답'으로 쓰임
        self.pub = self.create_publisher(TaskRequest, '/assist/request', 10)
        self.unmatched_pub = self.create_publisher(String, '/assist/unmatched_text', 10)
        self.create_subscription(Bool, '/assist/offer_active', lambda m: setattr(self, 'offer_active', m.data), 10)
        self.announce = self.create_publisher(String, '/assist/announce', 10)
        self.create_subscription(Gesture, '/user/gesture', self.on_gesture, 10)
        self.create_subscription(String, '/voice/command', lambda m: self.on_text(m.data, 'voice'), 10)
        self.create_subscription(String, '/assist/request_text', lambda m: self.on_text(m.data, 'manual'), 10)
        self.create_subscription(PointStamped, '/user/poke', self.on_poke, 10)
        self.create_timer(0.1, self.tick)
        attach_heartbeat(self)

    @staticmethod
    def _load(path):
        try:
            with open(os.path.expanduser(path)) as f:
                return yaml.safe_load(f) or {}
        except OSError:
            return {}

    def say(self, text):
        self.announce.publish(String(data=text))
        self.get_logger().info(f'[안내] {text}')

    def log_text(self, intent, args=None, conf=None):
        """음성/텍스트 한 문장당 한 번 DB 에 (전사, 의도, 슬롯, 확신도). verified 는 사람이 나중에 확인"""
        cur = getattr(self, '_cur_text', None)
        if self.db is None or cur is None or getattr(self, '_text_logged', True):
            return
        self._text_logged = True
        self.db.add_voice_utterance(cur[0], intent, {**(args or {}), 'source': cur[1]}, conf,
                                    person=self.get_parameter('person').value)

    def emit(self, task, args, source, raw=''):
        if getattr(self, '_cur_text', None) is not None:      # 음성/텍스트 처리 중에 나온 요청만 기록
            self.log_text(task, args, 1.0)
        now = time.monotonic()
        key = task + json.dumps(args, sort_keys=True)
        if task != 'cancel' and now - self.last.get(key, -1e9) < self.get_parameter('cooldown_sec').value:
            return
        self.last[key] = now
        m = TaskRequest(task=task, args=json.dumps(args, ensure_ascii=False), source=source, raw=raw)
        m.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(m)
        self.get_logger().info(f'요청: {task} {args} ({source}: {raw})')

    # ---------------------------------------------------------------- 입력
    def on_poke(self, m):
        pt = [m.point.x, m.point.y, m.point.z]
        now = time.monotonic()
        if self.seat_mode is not None:
            self.seat_mode['seats'].append(pt[:2])
            k, n = len(self.seat_mode['seats']), self.seat_mode['n']
            self.say(f'{k}번 자리 저장')
            if k >= n:
                self.emit('seat_setting', {'seats': self.seat_mode['seats']}, 'poke', f'{n}명')
                self.seat_mode = None
            return
        if self.pending_poke is not None:
            d = sum((a - b) ** 2 for a, b in zip(pt[:2], self.pending_poke[1][:2])) ** 0.5
            if d < self.get_parameter('same_poke_mm').value:
                self.emit('bring_dish', {'point': self.pending_poke[1]}, 'poke', 'double_poke')
                self.pending_poke = None
                return
        self.pending_poke = (now, pt)
        self.say('이거요?')

    def on_gesture(self, m):
        if not m.confirmed:
            return
        if self.offer_active and m.label in ('push', 'thumbs_up'):
            return                    # 제안에 대한 승낙/거절 → decision_node 가 처리
        now = time.monotonic()
        if m.label == 'push':
            if self.pending_poke is not None:
                self.emit('clear_dish', {'point': self.pending_poke[1]}, 'combo', 'poke+push')
                self.pending_poke = None
            else:
                self.emit('clear_empty', {}, 'gesture', 'push')
        elif m.label == 'thumbs_up':
            self.emit('clear_all', {}, 'gesture', 'thumbs_up')
        elif m.label == 'cup_grasp':
            self.cup_gesture_t = now
            self._check_water('gesture')
        elif m.label == 'hold_cup':
            pass                      # 실제 컵을 쥐고 있음 = 명령 아님 (물 따르기 트리거 금지). decision_node 가 '마시는 중' 으로 사용
        elif m.label == 'v_sign':
            self.emit('music', {'mood': 'any'}, 'gesture', 'v_sign')

    def on_text(self, text, source):
        self._text_logged = False
        self._cur_text = (text, source)
        try:
            self._on_text(text, source)
        finally:
            # 요청을 바로 만들지 않은 말 (예: 'N명이야', 컵 손동작 기다리는 '물 줘') 도 기록
            self.log_text('pending' if not self._text_logged else '', {}, None)
            self._cur_text = None

    def _on_text(self, text, source):
        t = norm(text)
        now = time.monotonic()
        has = lambda k: any(w in t for w in W[k])  # noqa: E731
        if has('cancel'):
            self.pending_poke = self.seat_mode = None
            self.emit('cancel', {}, source, text)
            return
        n = self._seat_count(t)
        if n:
            self.seat_mode = {'n': n, 'seats': [], 't': now}
            self.say(f'{n}명 자리를 차례로 콕 짚어 주세요')
            return
        if has('clear_all'):
            self.emit('clear_all', {}, source, text)
            return
        if has('water'):
            self.water_voice_t = now
            if not self.get_parameter('water_require_both').value:
                self.emit('pour_water', {}, source, text)
            else:
                self._check_water(source, text)
            return
        if has('wipe'):
            self.emit('wipe_table', {}, source, text)
            return
        if has('music'):
            self.emit('music', {'mood': self._mood(t)}, source, text)
            return
        if has('clear'):
            if self.pending_poke is not None:
                self.emit('clear_dish', {'point': self.pending_poke[1]}, 'combo', text)
                self.pending_poke = None
            else:
                self.emit('clear_empty', {}, source, text)
            return
        food = self._food(t)
        if food:
            self.emit('bring_dish', {'food': food}, source, text)
            return
        if has('give') and self.pending_poke is not None:
            self.emit('bring_dish', {'point': self.pending_poke[1]}, 'combo', text)
            self.pending_poke = None
            return
        if len(t) <= 4 and any(w in t for w in YES_NO):
            self.log_text('answer', {}, 1.0)
            return                    # "네/아니" 같은 단답은 decision_node 제안에 대한 대답 → planner 로 안 보냄
        # 키워드로 못 알아들은 말 → claude_planner_node 가 해석 (없으면 무시됨)
        self.log_text('unmatched', {}, 0.0)
        self.unmatched_pub.publish(String(data=text))
        self.get_logger().info(f'키워드 매칭 없음 → planner 로: {text}')

    # ---------------------------------------------------------------- 보조
    def _check_water(self, source, raw=''):
        w = self.get_parameter('combo_window_sec').value
        if abs(self.water_voice_t - self.cup_gesture_t) <= w:
            self.emit('pour_water', {}, 'combo', raw or 'voice+cup_grasp')
            self.water_voice_t = self.cup_gesture_t = -1e9
        elif source != 'gesture':
            self.say('컵 잡는 손동작도 보여 주세요')

    @staticmethod
    def _seat_count(t):
        m = re.search(r'(\d+)명', t)
        if m:
            return int(m.group(1))
        for k, v in sorted(KOR_NUM.items(), key=lambda kv: -len(kv[0])):
            if f'{k}명' in t or f'{k}사람' in t:
                return v
        return 0

    def _food(self, t):
        for name, f in self.foods.items():
            if any(norm(k) in t for k in f.get('keywords', [])):
                return name
        return None

    def _mood(self, t):
        for mood, kws in self.moods.items():
            if any(k in t for k in kws):
                return mood
        return 'any'

    def tick(self):
        now = time.monotonic()
        if self.pending_poke is not None and now - self.pending_poke[0] > self.get_parameter('poke_followup_sec').value:
            if self.get_parameter('poke_default').value == 'bring_dish':
                self.emit('bring_dish', {'point': self.pending_poke[1]}, 'poke', 'poke')
            self.pending_poke = None
        if self.seat_mode is not None and now - self.seat_mode['t'] > self.get_parameter('seat_poke_timeout_sec').value:
            self.say(f"자리 {len(self.seat_mode['seats'])}/{self.seat_mode['n']}곳만 받아 취소했어요")
            self.seat_mode = None


def main(args=None):
    rclpy.init(args=args)
    node = RequestRouter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
