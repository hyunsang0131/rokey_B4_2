"""assist_arm 데이터 DB (파이썬 내장 sqlite3, 설치 없음).

DB 파일: ~/data/assist_arm/assist_arm.db  (환경변수 ASSIST_ARM_DB 로 바꿀 수 있음)
원칙: 데이터 파일(npz, mp4 …)은 원래 자리에 그대로 두고, DB 에는 메타데이터와 파일 경로만 저장.
      DB 기록이 실패해도 원래 파일 저장은 그대로 (모든 기록 함수는 실패 시 None 을 돌려주고 경고만 출력).
여러 노드가 동시에 쓰므로 WAL 모드 + 호출마다 짧게 연결.

테이블
  sessions         녹화 세션 (사람, 카메라 source, 장치)
  gesture_samples  손동작 시퀀스 파일 (라벨, 프레임 수, fps, 손, 전처리 버전, train/val/test 분할)
  clips            원본 영상 파일
  decision_events  decision_node 제안·반응 (문맥 JSON, 행동, 수락/거절/무시, 보상, 모델 버전)
  voice_utterances 음성/텍스트 요청 (전사, 의도, 슬롯 JSON, 확신도, 사람 확인 여부)
  models           학습된 모델 파일 (종류, 경로, 학습 데이터 조건, 지표 JSON)
  yolo_images      식탁 사진 (객체인식 학습용, 라벨 상태 none/auto/human, 박스 수, 분할)
  -- v2.2 (세팅·정리 RL, 설계 문서 docs/ARCHITECTURE_v2.2.md 9장) --
  episodes          작업 한 번 (모드, 계획 방식 rl/rule, 인원, 결과 상태 DONE/BLOCKED/…, 동작 수, 시간)
  task_log          동작 한 번 (집기/놓기, 물체 종류, 출발·도착, 도착 더미 맨 위 종류, 성공, 오류 코드, 시간)
  storage_inventory 보관 더미·수저통·임시 자리 안의 물체 (카메라로 안 보이는 것의 기준, level 0 = 맨 아래)
  user_pref         사용자 선호 (자리, 왼손잡이 등)

사용 예
  from assist_arm.common.datadb import DataDB
  db = DataDB()
  sid = db.start_session('lee', 'realsense', device='D435i')
  db.add_gesture_sample(sid, 'clear', '/home/lee/data/assist_arm/gestures/clear/x.npz', n_frames=30, fps=15)
"""
import hashlib
import json
import os
import sqlite3
import time

DEFAULT_DB = os.path.expanduser(os.environ.get('ASSIST_ARM_DB', '~/data/assist_arm/assist_arm.db'))
SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  person TEXT, source TEXT, device TEXT, started_at TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS gesture_samples (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER REFERENCES sessions(id),
  label TEXT NOT NULL, file_path TEXT NOT NULL UNIQUE, n_frames INTEGER, fps REAL, handedness TEXT,
  preprocess_ver INTEGER, split TEXT CHECK (split IN ('train', 'val', 'test')), created_at TEXT);
CREATE TABLE IF NOT EXISTS clips (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER REFERENCES sessions(id),
  video_path TEXT NOT NULL UNIQUE, label TEXT, duration_s REAL);
CREATE TABLE IF NOT EXISTS decision_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, context_json TEXT, action TEXT, proposed INTEGER,
  user_response TEXT CHECK (user_response IN ('accept', 'reject', 'ignore') OR user_response IS NULL),
  reward REAL, model_ver TEXT);
CREATE TABLE IF NOT EXISTS voice_utterances (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT, person TEXT, audio_path TEXT, transcript TEXT, intent TEXT, slots_json TEXT,
  confidence REAL, verified INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS models (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT CHECK (kind IN ('gesture', 'intent', 'decision', 'yolo')), path TEXT NOT NULL,   -- 같은 경로 재학습은 새 줄 (이력)
  dataset_query TEXT, metrics_json TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS yolo_images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id INTEGER REFERENCES sessions(id),
  image_path TEXT NOT NULL UNIQUE, width INTEGER, height INTEGER, source TEXT,
  label_path TEXT, label_state TEXT DEFAULT 'none',   -- none / auto(초벌) / human(사람이 고침)
  n_boxes INTEGER, split TEXT, created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_gs_label ON gesture_samples(label);
CREATE INDEX IF NOT EXISTS idx_gs_session ON gesture_samples(session_id);
CREATE INDEX IF NOT EXISTS idx_de_action ON decision_events(action);
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT, ended_at TEXT,
  mode TEXT CHECK (mode IN ('clear', 'set_table')), planner TEXT CHECK (planner IN ('rl', 'rule')),
  people INTEGER,
  status TEXT CHECK (status IN ('DONE', 'BLOCKED', 'PERCEPTION_FAIL', 'ABORTED') OR status IS NULL),
  success INTEGER, reason TEXT, n_actions INTEGER DEFAULT 0, duration_s REAL);
CREATE TABLE IF NOT EXISTS task_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id INTEGER REFERENCES episodes(id), ts TEXT,
  skill TEXT, object_type TEXT, src TEXT, dst TEXT, dst_top_type TEXT,
  success INTEGER, error_code TEXT, duration_s REAL);
CREATE TABLE IF NOT EXISTS storage_inventory (
  location TEXT NOT NULL, level INTEGER NOT NULL, object_type TEXT NOT NULL, updated_at TEXT,
  PRIMARY KEY (location, level));
CREATE TABLE IF NOT EXISTS user_pref (
  user TEXT NOT NULL, key TEXT NOT NULL, value TEXT, updated_at TEXT, PRIMARY KEY (user, key));
CREATE INDEX IF NOT EXISTS idx_tl_episode ON task_log(episode_id);
"""


def now_iso():
    return time.strftime('%Y-%m-%dT%H:%M:%S')


def split_for(path, val=0.1, test=0.1):
    """파일 이름 해시로 정해지는 고정 분할 (다시 등록해도 같은 결과)"""
    h = int(hashlib.md5(os.path.basename(path).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return 'test' if h < test else ('val' if h < test + val else 'train')


class DataDB:
    def __init__(self, path=DEFAULT_DB, quiet=False):
        self.path = os.path.expanduser(path)
        self.quiet = quiet
        self.ok = False
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with self._conn() as c:
                c.execute('PRAGMA journal_mode=WAL')
                self._migrate(c)
                c.executescript(SCHEMA)
                c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)",
                          (str(SCHEMA_VERSION),))
            self.ok = True
        except Exception as e:  # noqa: BLE001
            self._warn(f'DB 열기 실패 ({self.path}): {e}')

    @staticmethod
    def _migrate(c):
        """예전 DB 호환: models.kind 에 'yolo' 가 없던 버전 → 표를 새로 만들고 내용 복사 (기존 줄 유지)"""
        row = c.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='models'").fetchone()
        if row and "'yolo'" not in row[0]:
            c.execute('ALTER TABLE models RENAME TO models_old')
            c.executescript(SCHEMA)
            c.execute('INSERT INTO models (id, kind, path, dataset_query, metrics_json, created_at) '
                      'SELECT id, kind, path, dataset_query, metrics_json, created_at FROM models_old')
            c.execute('DROP TABLE models_old')

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=5.0)
        c.execute('PRAGMA foreign_keys=ON')
        c.execute('PRAGMA busy_timeout=5000')
        return c

    def _warn(self, msg):
        if not self.quiet:
            print(f'[datadb] {msg} — 파일 저장은 그대로 진행')

    def _insert(self, table, row, ignore=False):
        if not self.ok:
            return None
        cols = ', '.join(row)
        q = f"INSERT {'OR IGNORE ' if ignore else ''}INTO {table} ({cols}) VALUES ({', '.join('?' * len(row))})"
        try:
            with self._conn() as c:
                cur = c.execute(q, list(row.values()))
                return cur.lastrowid if cur.rowcount else None
        except Exception as e:  # noqa: BLE001
            self._warn(f'{table} 기록 실패: {e}')
            return None

    def query(self, sql, params=()):
        """읽기 전용 질의 → list[dict]"""
        if not self.ok:
            return []
        try:
            with self._conn() as c:
                c.row_factory = sqlite3.Row
                return [dict(r) for r in c.execute(sql, params).fetchall()]
        except Exception as e:  # noqa: BLE001
            self._warn(f'질의 실패: {e}')
            return []

    def has_path(self, table, col, path):
        return bool(self.query(f'SELECT 1 FROM {table} WHERE {col} = ? LIMIT 1', (os.path.abspath(path),)))

    # ------------------------------------------------------------ 기록
    def start_session(self, person, source, device='', notes=''):
        return self._insert('sessions', {'person': person, 'source': source, 'device': str(device),
                                         'started_at': now_iso(), 'notes': notes})

    def find_or_create_session(self, person, source, notes):
        r = self.query('SELECT id FROM sessions WHERE person=? AND source=? AND notes=? LIMIT 1', (person, source, notes))
        return r[0]['id'] if r else self.start_session(person, source, notes=notes)

    def add_gesture_sample(self, session_id, label, file_path, n_frames=None, fps=None, handedness='',
                           preprocess_ver=None, split=None):
        p = os.path.abspath(file_path)
        return self._insert('gesture_samples', {
            'session_id': session_id, 'label': label, 'file_path': p, 'n_frames': n_frames,
            'fps': None if fps is None else float(fps), 'handedness': handedness, 'preprocess_ver': preprocess_ver,
            'split': split or split_for(p), 'created_at': now_iso()}, ignore=True)

    def add_clip(self, session_id, video_path, label='', duration_s=None):
        return self._insert('clips', {'session_id': session_id, 'video_path': os.path.abspath(video_path),
                                      'label': label, 'duration_s': duration_s}, ignore=True)

    def add_decision_event(self, context, action, proposed, user_response=None, reward=None, model_ver=''):
        return self._insert('decision_events', {
            'ts': now_iso(), 'context_json': json.dumps(context, ensure_ascii=False, default=float),
            'action': action, 'proposed': int(bool(proposed)), 'user_response': user_response,
            'reward': reward, 'model_ver': model_ver})

    def add_voice_utterance(self, transcript, intent, slots=None, confidence=None, person='', audio_path='',
                            verified=False):
        return self._insert('voice_utterances', {
            'ts': now_iso(), 'person': person, 'audio_path': audio_path, 'transcript': transcript, 'intent': intent,
            'slots_json': json.dumps(slots or {}, ensure_ascii=False), 'confidence': confidence,
            'verified': int(bool(verified))})

    def add_model(self, kind, path, dataset_query='', metrics=None):
        return self._insert('models', {'kind': kind, 'path': os.path.abspath(os.path.expanduser(path)),
                                       'dataset_query': dataset_query,
                                       'metrics_json': json.dumps(metrics or {}, ensure_ascii=False, default=float),
                                       'created_at': now_iso()})

    def add_yolo_image(self, session_id, image_path, width=None, height=None, source=''):
        p = os.path.abspath(os.path.expanduser(image_path))
        return self._insert('yolo_images', {'session_id': session_id, 'image_path': p, 'width': width,
                                            'height': height, 'source': source, 'split': split_for(p, 0.15, 0.05),
                                            'created_at': now_iso()}, ignore=True)

    def set_yolo_label(self, image_path, label_path, state, n_boxes):
        if not self.ok:
            return
        try:
            with self._conn() as c:
                c.execute('UPDATE yolo_images SET label_path=?, label_state=?, n_boxes=? WHERE image_path=?',
                          (os.path.abspath(label_path), state, int(n_boxes), os.path.abspath(image_path)))
        except Exception as e:  # noqa: BLE001
            self._warn(f'yolo 라벨 기록 실패: {e}')

    def remove_files(self, gesture_path=None, video_path=None):
        """녹화 취소(u) 때: 해당 파일의 DB 줄 삭제"""
        if not self.ok:
            return
        try:
            with self._conn() as c:
                if gesture_path:
                    c.execute('DELETE FROM gesture_samples WHERE file_path=?', (os.path.abspath(gesture_path),))
                if video_path:
                    c.execute('DELETE FROM clips WHERE video_path=?', (os.path.abspath(video_path),))
        except Exception as e:  # noqa: BLE001
            self._warn(f'삭제 실패: {e}')

    # ------------------------------------------------------------ v2.2: 작업 기록 (세팅·정리)
    def _exec(self, sql, params=(), what='기록'):
        if not self.ok:
            return None
        try:
            with self._conn() as c:
                return c.execute(sql, params).rowcount
        except Exception as e:  # noqa: BLE001
            self._warn(f'{what} 실패: {e}')
            return None

    def start_episode(self, mode, planner='rule', people=None):
        """작업 한 번 시작 → episode id (실패 시 None)"""
        return self._insert('episodes', {'started_at': now_iso(), 'mode': mode, 'planner': planner,
                                         'people': people})

    def finish_episode(self, episode_id, status, reason='', n_actions=None, duration_s=None):
        """status: DONE / BLOCKED / PERCEPTION_FAIL / ABORTED. success 는 DONE 일 때만 1"""
        if episode_id is None:
            return None
        return self._exec('UPDATE episodes SET ended_at=?, status=?, success=?, reason=?, '
                          'n_actions=COALESCE(?, n_actions), duration_s=? WHERE id=?',
                          (now_iso(), status, int(status == 'DONE'), reason, n_actions, duration_s, episode_id),
                          'episode 종료 기록')

    def log_task(self, episode_id, skill, object_type, src='', dst='', dst_top_type='', success=True,
                 error_code='OK', duration_s=None):
        """동작 한 번 (pick / place / place_stack …). episodes.n_actions(옮긴 물체 수)는 finish_episode 에서 정함"""
        return self._insert('task_log', {
            'episode_id': episode_id, 'ts': now_iso(), 'skill': skill, 'object_type': object_type, 'src': src,
            'dst': dst, 'dst_top_type': dst_top_type or '', 'success': int(bool(success)),
            'error_code': error_code, 'duration_s': duration_s})

    # ------------------------------------------------------------ v2.2: 보관 재고 (포개진 그릇은 안 보이므로 DB 가 기준)
    def inventory(self, location=None):
        """location 을 주면 그 자리의 [아래 → 위] 종류 목록, 안 주면 {location: [...]}"""
        if location is not None:
            return [r['object_type'] for r in self.query(
                'SELECT object_type FROM storage_inventory WHERE location=? ORDER BY level', (location,))]
        out = {}
        for r in self.query('SELECT location, object_type FROM storage_inventory ORDER BY location, level'):
            out.setdefault(r['location'], []).append(r['object_type'])
        return out

    def inventory_push(self, location, object_type):
        """맨 위에 하나 올림 → 새 level (실패 시 None)"""
        n = len(self.inventory(location))
        return n if self._insert('storage_inventory', {'location': location, 'level': n, 'object_type': object_type,
                                                       'updated_at': now_iso()}) is not None else None

    def inventory_pop(self, location):
        """맨 위 하나 꺼냄 → 그 종류 (비어 있거나 실패면 None)"""
        items = self.inventory(location)
        if not items:
            return None
        ok = self._exec('DELETE FROM storage_inventory WHERE location=? AND level=?', (location, len(items) - 1),
                        '재고 꺼내기')
        return items[-1] if ok else None

    def inventory_set(self, location, object_types):
        """한 자리의 내용을 통째로 바꿈 (시작 전 초기 재고 입력, 사람이 정리한 뒤 맞추기)"""
        if not self.ok:
            return False
        try:
            with self._conn() as c:
                c.execute('DELETE FROM storage_inventory WHERE location=?', (location,))
                c.executemany('INSERT INTO storage_inventory (location, level, object_type, updated_at) '
                              'VALUES (?, ?, ?, ?)',
                              [(location, i, t, now_iso()) for i, t in enumerate(object_types)])
            return True
        except Exception as e:  # noqa: BLE001
            self._warn(f'재고 설정 실패: {e}')
            return False

    # ------------------------------------------------------------ v2.2: 사용자 선호
    def set_pref(self, user, key, value):
        return self._exec('INSERT INTO user_pref (user, key, value, updated_at) VALUES (?, ?, ?, ?) '
                          'ON CONFLICT(user, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at',
                          (user, key, json.dumps(value, ensure_ascii=False), now_iso()), '선호 기록')

    def get_pref(self, user, key, default=None):
        r = self.query('SELECT value FROM user_pref WHERE user=? AND key=?', (user, key))
        if not r:
            return default
        try:
            return json.loads(r[0]['value'])
        except ValueError:
            return r[0]['value']

    def counts(self):
        out = {}
        for t in ('sessions', 'gesture_samples', 'clips', 'decision_events', 'voice_utterances', 'models',
                  'yolo_images', 'episodes', 'task_log', 'storage_inventory', 'user_pref'):
            r = self.query(f'SELECT COUNT(*) AS n FROM {t}')
            out[t] = r[0]['n'] if r else 0
        return out
