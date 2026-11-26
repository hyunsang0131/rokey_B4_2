"""claude_planner_node (planner): 키워드로 못 알아들은 말("배불러 좀 정리해 줘", "목이 칼칼해") + 장면 요약
→ 허용된 서브태스크 목록 중에서만 하나(또는 none / ask)를 고른다.

백엔드 (backend 파라미터)
  local  (기본): config/intents.yaml 예시 문장 ↔ 글자 n-gram 유사도 (scikit-learn, 비용 0, 오프라인)
  claude        : Claude API (키가 있을 때만). 키/SDK 가 없거나 API 오류면 local 로 대체

흐름
  /assist/unmatched_text (request_router 가 키워드 매칭 실패한 음성/텍스트)
  + 최근 /assist/scene, /safety/status, /assist/state
  → Claude (anthropic SDK, client.messages.parse + Pydantic 구조화 출력)
  → 안전 검사 (정지/일시정지 중이면 실행 안 함, 꺼진 기능·허용 밖 skill·모르는 target 거부)
  → confidence ≥ min_confidence  : /assist/request (source='claude') + 안내 한 줄
     confidence 낮음               : decision_node 로 '제안 → 확인' (/assist/suggest)
     skill=ask                    : 되묻기 (/assist/announce)
     refusal / none              : 아무것도 안 함
  API 실패 (키 없음, RateLimit, APIStatus, 연결 실패) → 로컬 규칙으로 추정해 decision_node 에 제안만 넘김
키: ANTHROPIC_API_KEY 환경변수 또는 `ant auth login` 프로필(~/.config/anthropic). 코드/yaml 에 키를 쓰지 않는다.
의존성: anthropic, pydantic — assist_mp venv 에 (README). mock=true 면 API 를 부르지 않고 가짜 응답.
"""
import json
import os
import time
from typing import Literal, Optional

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String

from assist_arm_msgs.msg import SafetyStatus, TaskRequest
from assist_arm.common.datadb import DataDB
from assist_arm.common.heartbeat import attach_heartbeat
from assist_arm.common.venv_import import import_from_venv

try:
    from pydantic import BaseModel, Field
except ImportError:  # 시스템에 없으면 venv 에서
    _p = import_from_venv('pydantic')
    BaseModel, Field = _p.BaseModel, _p.Field

# 플래너가 고를 수 있는 것 (콕/자리 좌표가 필요한 clear_dish, seat_setting 은 제외 — 손으로 지정해야 함)
SKILLS = ('pour_water', 'bring_dish', 'clear_empty', 'clear_all', 'wipe_table', 'music', 'none', 'ask')


class PlannerDecision(BaseModel):
    skill: Literal['pour_water', 'bring_dish', 'clear_empty', 'clear_all', 'wipe_table', 'music', 'none', 'ask']
    target: Optional[str] = Field(None, description='bring_dish 면 음식 이름(foods 키), music 이면 분위기. 그 외 null')
    say: str = Field(description='사용자에게 할 한국어 한 줄 (되묻기면 질문)')
    confidence: float = Field(ge=0.0, le=1.0)


SYSTEM_PROMPT = """너는 식탁 옆 가정용 로봇팔의 작업 선택기다. 사용자의 말과 현재 장면을 보고, 아래 작업 중 하나만 고른다.
로봇은 사람 몸에 닿지 않는다. 물건은 테이블 위에 내려놓기만 한다.

작업:
- pour_water: 컵에 물을 따라 사용자 손 옆 테이블에 둔다 (목마름, 목이 칼칼함, 물 요청)
- bring_dish: 음식 그릇 하나를 사용자 앞으로 옮긴다. target 은 아래 음식 목록의 키 중 하나
- clear_empty: 빈 그릇만 치운다 (조금 정리, 다 먹은 그릇)
- clear_all: 테이블 전체를 치운다 (식사 끝, 배부름, 다 먹음)
- wipe_table: 테이블 얼룩을 걸레로 닦는다 (흘림, 지저분함)
- music: 노래 추천과 율동. target 은 분위기 (아래 목록 중 하나 또는 any)
- none: 로봇이 할 일이 아님 (잡담, 로봇 밖의 일, 사람 몸을 만져야 하는 일, 위험한 일)
- ask: 무엇을 원하는지 애매해서 되물어야 함 (say 에 짧은 질문)

규칙: 목록 밖의 일은 none. 확신이 없으면 confidence 를 낮게 주거나 ask. say 는 존댓말 한 문장."""


class ClaudePlanner(Node):
    def __init__(self):
        super().__init__('claude_planner_node')
        p = self.declare_parameter
        p('enabled', True)
        p('backend', 'local')                # local (무료, 기본) / claude (키 필요)
        p('intents_file', '')                # 비우면 share/assist_arm/config/intents.yaml
        p('local_ask_below', 0.45)           # 이 유사도 미만이면 되묻기
        p('local_sure', 0.65)                # 이 유사도 이상이면 바로 실행 (사이는 '~할까요?' 확인)
        p('use_db', True)                    # 해석 결과를 DB voice_utterances 에 기록
        p('mock', False)                     # backend=claude 일 때 API 호출 없이 가짜 응답 (dry_run / 테스트)
        p('model', 'claude-opus-5-5')
        p('effort', 'low')                   # 지연을 줄이기 위해 low (low/medium/high/xhigh/max)
        p('max_tokens', 1024)
        p('timeout_sec', 20.0)
        p('min_confidence', 0.6)
        p('enabled_skills', ['pour_water', 'bring_dish', 'clear_empty', 'clear_all', 'wipe_table', 'music'])
        p('table_layout_file', '')
        p('music_library_file', '')
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        from ament_index_python.packages import get_package_share_directory
        import yaml
        share = get_package_share_directory('assist_arm')

        def load(path):
            try:
                with open(os.path.expanduser(path)) as f:
                    return yaml.safe_load(f) or {}
            except OSError:
                return {}
        self.foods = load(g('table_layout_file') or os.path.join(share, 'config', 'table_layout.yaml')).get('foods', {})
        self.moods = list(load(g('music_library_file') or os.path.join(share, 'config', 'music_library.yaml'))
                          .get('moods', {}))
        self.db = DataDB() if g('use_db') else None
        self.scene, self.safety, self.state = None, None, 'IDLE'
        self.offer_active = False
        from assist_arm.common.local_intent import LocalIntentClassifier
        self.local = LocalIntentClassifier(
            g('intents_file') or os.path.join(share, 'config', 'intents.yaml'),
            float(g('local_ask_below')), float(g('local_sure')))
        self.client, self.anthropic = None, None
        if g('enabled') and g('backend') == 'claude' and not g('mock'):
            self._init_client()
            if self.client is None:
                self.get_logger().warn('Claude 를 쓸 수 없어 local 백엔드로 동작합니다')
        self.get_logger().info(f"planner backend={g('backend')}{' (mock)' if g('mock') else ''}, "
                               f"local 예시 {len(self.local.texts)}개")

        cbg = MutuallyExclusiveCallbackGroup()   # 한 번에 한 요청만 (API 호출 중 다음 말은 대기)
        self.req_pub = self.create_publisher(TaskRequest, '/assist/request', 10)
        self.say_pub = self.create_publisher(String, '/assist/announce', 10)
        self.suggest_pub = self.create_publisher(String, '/assist/suggest', 10)
        self.log_pub = self.create_publisher(String, '/assist/planner', 10)
        self.create_subscription(String, '/assist/unmatched_text', self.on_text, 10, callback_group=cbg)
        self.create_subscription(String, '/assist/scene', lambda m: setattr(self, 'scene', m.data), 10)
        self.create_subscription(SafetyStatus, '/safety/status', lambda m: setattr(self, 'safety', m), 10)
        self.create_subscription(String, '/assist/state', lambda m: setattr(self, 'state', m.data), 10)
        self.create_subscription(Bool, '/assist/offer_active', lambda m: setattr(self, 'offer_active', m.data), 10)
        attach_heartbeat(self)

    # ---------------------------------------------------------------- 클라이언트
    def _init_client(self):
        has_profile = os.path.isdir(os.path.expanduser('~/.config/anthropic'))
        if not os.environ.get('ANTHROPIC_API_KEY') and not os.environ.get('ANTHROPIC_AUTH_TOKEN') and not has_profile:
            self.get_logger().error(
                'Claude 키 없음 → planner 비활성 (규칙 기반 제안만). 설정: export ANTHROPIC_API_KEY=... '
                '또는 `ant auth login`. 키를 코드/yaml 에 쓰지 마세요.')
            return
        try:
            self.anthropic = import_from_venv('anthropic')
        except ImportError:
            self.get_logger().error('anthropic SDK 없음 → planner 비활성. README 의 venv 설치 명령 참고 '
                                    '(~/venvs/assist_mp/bin/pip install anthropic pydantic)')
            return
        self.client = self.anthropic.Anthropic(timeout=float(self.g('timeout_sec')), max_retries=1)
        self.get_logger().info(f"Claude planner 준비 (model={self.g('model')}, effort={self.g('effort')})")

    # ---------------------------------------------------------------- 장면 요약
    def scene_summary(self):
        lines = []
        try:
            sc = json.loads(self.scene) if self.scene else {}
        except ValueError:
            sc = {}
        objs = sc.get('objects', [])
        if objs:
            desc = ', '.join(f"{o['cls']}(채움 {o['fill']:.1f})" if o.get('fill', -1) >= 0 else o['cls'] for o in objs)
            lines.append(f'테이블 위: {desc}')
        else:
            lines.append('테이블 위: (최근 관찰 없음)')
        sp = sc.get('spill', {})
        lines.append(f"얼룩: {sp.get('label', '있음') if sp.get('detected') else '없음'}")
        lines.append(f"음식 목록(bring_dish target): {', '.join(self.foods) or '없음'}")
        lines.append(f"음악 분위기(music target): {', '.join(self.moods + ['any'])}")
        lines.append(f"켜진 작업: {', '.join(self.g('enabled_skills'))}")
        lines.append(f'로봇 상태: {self.state}')
        return '\n'.join(lines)

    # ---------------------------------------------------------------- 결정
    def decide(self, text):
        """→ (PlannerDecision | None, source). None 이면 API 실패 → fallback"""
        if self.g('backend') != 'claude':
            return self.local_decide(text), 'local'
        if self.g('mock'):
            return self.mock_decide(text), 'mock'
        if self.client is None:
            return None, 'disabled'
        a = self.anthropic
        user = f'사용자 말: "{text}"\n\n현재 장면:\n{self.scene_summary()}'
        try:
            response = self.client.messages.parse(
                model=self.g('model'),
                max_tokens=int(self.g('max_tokens')),
                output_config={'effort': self.g('effort')},
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': user}],
                output_format=PlannerDecision,
            )
        except a.RateLimitError as e:
            self.get_logger().warn(f'Claude rate limit → 규칙 기반 대체: {e}')
            return None, 'rate_limit'
        except a.APIStatusError as e:
            self.get_logger().error(f'Claude API 오류 {e.status_code} → 규칙 기반 대체: {e.message}')
            return None, f'api_status_{e.status_code}'
        except a.APIConnectionError as e:
            self.get_logger().error(f'Claude 연결 실패 → 규칙 기반 대체: {e}')
            return None, 'connection'
        if response.stop_reason == 'refusal':
            self.get_logger().warn('Claude 가 요청을 거절함 → 아무것도 안 함')
            return PlannerDecision(skill='none', say='', confidence=1.0), 'refusal'
        if response.parsed_output is None:
            return None, f'no_parse({response.stop_reason})'
        return response.parsed_output, 'claude'

    @staticmethod
    def mock_decide(text):
        """dry_run 용 가짜 응답 (API 호출 없음). 실제 Claude 응답 형식과 같은 PlannerDecision."""
        t = text.replace(' ', '')
        if any(w in t for w in ('배불러', '배부르', '그만먹')):
            return PlannerDecision(skill='clear_all', say='다 드셨으면 정리해 드릴게요', confidence=0.8)
        if any(w in t for w in ('칼칼', '목말', '갈증', '목이')):
            return PlannerDecision(skill='pour_water', say='물 한 잔 드릴까요?', confidence=0.55)
        if any(w in t for w in ('지저분', '끈적')):
            return PlannerDecision(skill='wipe_table', say='테이블 닦아 드릴게요', confidence=0.75)
        if any(w in t for w in ('심심', '우울')):
            return PlannerDecision(skill='music', target='신나는', say='신나는 노래 틀어 드릴게요', confidence=0.7)
        if '뭐' in t or '?' in text:
            return PlannerDecision(skill='ask', say='무엇을 도와드릴까요?', confidence=0.3)
        return PlannerDecision(skill='none', say='', confidence=0.9)

    def local_decide(self, text):
        skill, target, say, conf, ex, sim = self.local.classify(text)
        self.get_logger().info(f'local: "{text}" → {skill} (sim {sim:.2f}, 예시 "{ex}")')
        return PlannerDecision(skill=skill, target=target, say=say or '', confidence=min(1.0, conf))

    def rule_fallback(self, text):
        """Claude 를 못 쓸 때: local 분류기로 추정 → decision_node 가 '제안 → 확인' 으로 처리"""
        d = self.local_decide(text)
        return d if d.skill in ('pour_water', 'clear_all', 'clear_empty', 'wipe_table', 'music') else None

    # ---------------------------------------------------------------- 안전 검사
    def safety_check(self, d):
        if d.skill not in SKILLS:
            return False, f'허용 밖 skill {d.skill}'
        if d.skill in ('none', 'ask'):
            return True, ''
        if d.skill not in self.g('enabled_skills'):
            return False, f'{d.skill} 꺼짐'
        if self.safety is not None and (self.safety.stop or self.safety.paused):
            return False, '안전 정지/일시정지 중'
        if d.skill == 'bring_dish' and d.target not in self.foods:
            return False, f'모르는 음식 {d.target}'
        if d.skill == 'music' and d.target not in self.moods + ['any', None]:
            d.target = 'any'
        return True, ''

    # ---------------------------------------------------------------- 처리
    def on_text(self, m):
        text = m.data.strip()
        if not text or not self.g('enabled'):
            return
        t0 = time.monotonic()
        d, src = self.decide(text)
        if d is None:
            fb = self.rule_fallback(text)
            self.log(text, None, src, time.monotonic() - t0)
            if fb is not None:
                self.suggest_pub.publish(String(data=json.dumps(
                    {'task': fb.skill, 'reason': f'fallback:{src}', 'say': fb.say}, ensure_ascii=False)))
            return
        ok, why = self.safety_check(d)
        self.log(text, d, src, time.monotonic() - t0, ok, why)
        if not ok:
            self.get_logger().warn(f'planner 결과 거부: {why}')
            return
        if d.skill == 'none':
            return
        if d.skill == 'ask':
            self.say_pub.publish(String(data=d.say or '무엇을 도와드릴까요?'))     # 되묻기
            return
        if d.confidence < self.g('min_confidence'):
            # 확신 낮음 → 바로 하지 않고 decision_node 가 "~할까요?" 로 묻고 엄지척/손등 밀기로 확인
            self.suggest_pub.publish(String(data=json.dumps(
                {'task': d.skill, 'reason': f'claude_low_conf:{d.confidence:.2f}', 'say': d.say},
                ensure_ascii=False)))
            return
        args = {'food': d.target} if d.skill == 'bring_dish' else ({'mood': d.target or 'any'} if d.skill == 'music' else {})
        r = TaskRequest(task=d.skill, args=json.dumps(args, ensure_ascii=False), source=f'planner_{self.g("backend")}', raw=text)
        r.header.stamp = self.get_clock().now().to_msg()
        self.req_pub.publish(r)
        if d.say:
            self.say_pub.publish(String(data=d.say))

    def log(self, text, d, src, dt, ok=None, why=''):
        rec = {'text': text, 'source': src, 'latency_sec': round(dt, 2), 'ok': ok, 'why': why,
               'decision': d.model_dump() if d is not None else None}
        self.log_pub.publish(String(data=json.dumps(rec, ensure_ascii=False)))
        if self.db is not None:
            self.db.add_voice_utterance(text, d.skill if d is not None else f'fail:{src}',
                                        {'target': getattr(d, 'target', None), 'say': getattr(d, 'say', ''),
                                         'source': f'planner_{src}', 'accepted': ok, 'why': why},
                                        None if d is None else float(d.confidence))
        self.get_logger().info(f'planner: {rec}')


def main(args=None):
    rclpy.init(args=args)
    node = ClaudePlanner()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001  (종료 신호 중 context 무효)
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
