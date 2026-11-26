"""손 랜드마크 → 특징 벡터 / 규칙 기반 제스처.

gesture(intent_node), tools/live_record.py, tools/train_gesture_sklearn.py·Colab 노트북이 *같은* 전처리를 써야 하므로
이 파일 하나에 모아둔다. (training 스크립트는 Colab 에서 이 파일을 함께 업로드해서 import)
"""
import numpy as np

WRIST = 0
THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP = 4, 8, 12, 16, 20
MCP = {'index': 5, 'middle': 9, 'ring': 13, 'pinky': 17}
PIP = {'index': 6, 'middle': 10, 'ring': 14, 'pinky': 18}
TIP = {'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20}
PALM_IDS = [0, 5, 9, 13, 17]

FEATURE_DIM = 63
# 제스처 라벨 (규칙 기반 fallback 으로 모두 나옴; 학습/kNN 으로 사람마다 다듬기)
#   stop      손바닥을 카메라로 펴서 내밀기         → 안전 정지
#   push      손등을 보이며 손 펴기 (밀어내기)       → '치워' (빈 그릇 정리 / 콕 선택 후 치우기)
#   cup_grasp 컵 잡는 C자                           → 물 (음성 "물 줘" 와 함께일 때만)
#   point     검지만 펴기                            → finger_pointer 가 1초 머물면 poke 이벤트
#   thumbs_up 엄지 척                                → 식사 끝 (테이블 전체 정리)
#   v_sign    검지+중지                              → 음악/율동
DEFAULT_LABELS = ['idle', 'stop', 'push', 'cup_grasp', 'point', 'thumbs_up', 'v_sign']


def hand_feature(pts, handedness='Right'):
    """(21,3) 정규화좌표 → 63차원. 손목 원점, 손 크기(손목-중지MCP) 정규화, 왼손은 x 반전."""
    p = np.asarray(pts, dtype=np.float32).reshape(21, 3).copy()
    p -= p[WRIST]
    if handedness == 'Left':
        p[:, 0] *= -1.0
    scale = np.linalg.norm(p[MCP['middle'], :2])
    if scale < 1e-6:
        scale = 1.0
    p /= scale
    return p.reshape(-1)


def hand_feature_world(world, handedness='Right'):
    """MediaPipe hand world landmarks (21,3, 미터) → 63차원. 손목 원점, 손목-중지MCP 3D 길이로 정규화,
    왼손은 x 반전. 화면 위치·카메라 거리와 무관한 '실제 손 모양' 이라 학습에 유리."""
    p = np.asarray(world, dtype=np.float32).reshape(21, 3).copy()
    p -= p[WRIST]
    if handedness == 'Left':
        p[:, 0] *= -1.0
    scale = np.linalg.norm(p[MCP['middle']])
    p /= scale if scale > 1e-6 else 1.0
    return p.reshape(-1)


def shape_feature(pts, handedness='Right', world=None, space='world'):
    """space='world' 이고 world 가 있으면 world 특징, 아니면 이미지 좌표 특징. → (63차원, 실제 사용 space)"""
    if space == 'world' and world is not None and np.all(np.isfinite(world)):
        return hand_feature_world(world, handedness), 'world'
    return hand_feature(pts, handedness), 'image'


def finger_curl_ratios(pts):
    """손가락별 (손목-끝 거리)/(손목-MCP 거리). 펴면 ~1.8+, 주먹 ~1.0 이하."""
    p = np.asarray(pts, dtype=np.float32).reshape(21, 3)[:, :2]
    w = p[WRIST]
    out = {}
    for f in ('index', 'middle', 'ring', 'pinky'):
        d_mcp = np.linalg.norm(p[MCP[f]] - w) + 1e-6
        out[f] = float(np.linalg.norm(p[TIP[f]] - w) / d_mcp)
    return out


def hand_size(pts):
    p = np.asarray(pts, dtype=np.float32).reshape(21, 3)[:, :2]
    return float(np.linalg.norm(p[MCP['middle']] - p[WRIST]) + 1e-6)


def palm_facing_camera(pts, handedness='Right', palm_sign=1.0):
    """손바닥이 카메라를 향하면 True (손등이면 False). 손목·검지MCP·소지MCP 로 만든 법선의 z 부호.
    카메라/거울 설정에 따라 부호가 반대일 수 있음 → gesture_node 파라미터 palm_sign 으로 보정."""
    p = np.asarray(pts, dtype=np.float32).reshape(21, 3)
    n = np.cross(p[MCP['index']] - p[WRIST], p[MCP['pinky']] - p[WRIST])
    side = 1.0 if handedness == 'Right' else -1.0
    return bool(n[2] * side * palm_sign < 0)


def rule_gesture(pts, handedness='Right', palm_sign=1.0, open_th=1.75, curl_lo=1.05, curl_hi=1.6,
                 c_gap_min=0.35, bent_th=1.3):
    """규칙 기반 fallback.  반환: (label, confidence)"""
    r = finger_curl_ratios(pts)
    vals = np.array(list(r.values()))
    p = np.asarray(pts, dtype=np.float32).reshape(21, 3)[:, :2]
    gap = float(np.linalg.norm(p[THUMB_TIP] - p[INDEX_TIP]) / hand_size(pts))
    ext = {f: r[f] > open_th for f in r}
    bent = {f: r[f] < bent_th for f in r}
    if all(ext.values()):
        if palm_facing_camera(pts, handedness, palm_sign):
            return 'stop', float(min(1.0, (vals.min() - open_th) / 0.3 + 0.6))
        return 'push', 0.7
    if np.all(vals < curl_lo):                                   # 주먹 (네 손가락 완전히 굽힘)
        hs = hand_size(pts)
        if p[THUMB_TIP][1] < p[MCP['index']][1] - 0.5 * hs:      # 엄지 끝이 손 위로 확실히 (이미지 y 작음)
            return 'thumbs_up', 0.7
        return 'idle', 0.5
    if ext['index'] and bent['middle'] and bent['ring'] and bent['pinky']:
        return 'point', 0.7
    if ext['index'] and ext['middle'] and bent['ring'] and bent['pinky']:
        return 'v_sign', 0.7
    if np.all((vals > curl_lo) & (vals < curl_hi)) and gap > c_gap_min:
        return 'cup_grasp', 0.7
    return 'idle', 0.5


def is_pointing(pts, open_th=1.75, bent_th=1.3):
    r = finger_curl_ratios(pts)
    return r['index'] > open_th and all(r[f] < bent_th for f in ('middle', 'ring', 'pinky'))


def is_grasping(pts, curl_th=1.35):
    """핸드오버용: 손가락이 (컵을) 감싸 굽혀져 있는가. 검지~약지 3개 이상 굽힘."""
    r = finger_curl_ratios(pts)
    bent = sum(1 for f in ('index', 'middle', 'ring', 'pinky') if r[f] < curl_th)
    return bent >= 3


# ---------------------------------------------------------------------- 학습·추론 공통 특징 (버전 2)
# 카메라가 바뀌어도(노트북 웹캠 / USB 캠 / RealSense) 비슷하게 나오도록 '손 기준 좌표'로 정규화한다.
#   손 기준 좌표: 손목 원점, x축 = 손목→중지 MCP, z축 = 손바닥 법선(검지MCP × 새끼MCP), y = z × x,
#               크기 = 손목-중지MCP 길이로 나눔. 왼손은 x 반전 후 계산 → 왼손/오른손 같은 모양이면 같은 값.
#   (world landmarks(미터 3D)가 있으면 그것으로, 없으면 이미지 좌표로)
# 정지 특징(STATIC_DIM 66) = 손 기준 21점(63) + 손바닥 법선의 카메라 방향(3: 손바닥/손등 구분용)
# 시퀀스 특징(SEQ_FEATURE_DIM 71) = 정지 66 + 손 중심 속도를 손바닥 좌표계로(3) + 속력(1, 초당 손 크기) + 손 있음(1)
FEATURE_VERSION = 2
SEQ_LEN = 30                       # 2초 (≈15fps)
STATIC_DIM = FEATURE_DIM + 3
SEQ_FEATURE_DIM = STATIC_DIM + 5
MAX_SPEED = 20.0                   # 초당 손 크기, 이상치 자르기

# 녹화 라벨 → 시스템 라벨 (request_router / safety_monitor 가 쓰는 이름)
GESTURE_ALIASES = {'poke': 'point', 'point_seat': 'point', 'clear': 'push', 'meal_done': 'thumbs_up',
                   'cup_grab': 'cup_grasp'}
RECORD_LABELS = ['idle', 'poke', 'clear', 'meal_done', 'cup_grab', 'stop', 'point_seat', 'v_sign', 'hold_cup']
# cup_grab : 빈손으로 컵 잡는 시늉 (C자) = '물 줘' 요청 제스처
# hold_cup : 실제 컵을 쥐고/들고 있음 (마시는 중) = 명령 아님 → decision_node 가 '마시는 중' 으로만 사용
RECORD_HINTS = {'idle': 'no command: rest, move, touch things', 'poke': 'index finger, hold 1s on a spot',
                'clear': 'push away with back of hand', 'meal_done': 'thumbs up', 'cup_grab': 'EMPTY hand C-shape (ask water)',
                'stop': 'open palm to camera', 'point_seat': 'point at a seat', 'v_sign': 'V sign',
                'hold_cup': 'hold the REAL cup (drinking)'}


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def palm_frame(P):
    """P (21,3) → (R 3x3: 행 = 손바닥 x,y,z 축 (원래 좌표계 기준), scale)"""
    x = _unit(P[MCP['middle']] - P[WRIST])
    z = _unit(np.cross(P[MCP['index']] - P[WRIST], P[MCP['pinky']] - P[WRIST]))
    z = _unit(z - np.dot(z, x) * x)
    y = np.cross(z, x)
    return np.stack([x, y, z]), float(np.linalg.norm(P[MCP['middle']] - P[WRIST]) + 1e-9)


def _source_points(pts, handedness, world):
    use_world = world is not None and np.all(np.isfinite(world))
    P = np.asarray(world if use_world else pts, np.float64).reshape(21, 3).copy()
    if not use_world:
        P[:, 2] *= 1.0          # 이미지 z 는 상대 깊이 (손목 기준) 그대로
    if handedness == 'Left':
        P[:, 0] *= -1.0
    return P


def static_feature(pts, handedness='Right', world=None):
    """정지 자세 특징 (STATIC_DIM). 카메라 회전·거리에 덜 민감한 손 기준 좌표 + 손바닥 방향."""
    P = _source_points(pts, handedness, world)
    R, scale = palm_frame(P)
    local = ((P - P[WRIST]) @ R.T) / scale
    return np.concatenate([local.reshape(-1), R[2]]).astype(np.float32)


class SeqFeaturizer:
    """프레임마다 step() → SEQ_FEATURE_DIM. 손 중심 속도를 손바닥 좌표계로 바꿔 넣어 카메라 위치와 덜 관련되게."""

    def __init__(self):
        self.prev = None

    def reset(self):
        self.prev = None

    def step(self, pts, handedness='Right', world=None, t=0.0):
        out = np.zeros(SEQ_FEATURE_DIM, np.float32)
        if pts is None:
            self.prev = None
            return out
        pts = np.asarray(pts, np.float32).reshape(21, 3)
        out[:STATIC_DIM] = static_feature(pts, handedness, world)
        P = _source_points(pts, handedness, world)
        R, _ = palm_frame(P)
        c = pts[PALM_IDS, :2].mean(axis=0).astype(np.float64)
        if handedness == 'Left':
            c[0] = -c[0]
        size = hand_size(pts)
        if self.prev is not None and t - self.prev[0] > 1e-3:
            v = (c - self.prev[1]) / (t - self.prev[0]) / size          # 초당 손 크기 (이미지 기준)
            v3 = np.array([v[0], v[1], 0.0])
            sp = float(np.linalg.norm(v3))
            if sp > MAX_SPEED:
                v3 *= MAX_SPEED / sp
                sp = MAX_SPEED
            out[STATIC_DIM:STATIC_DIM + 3] = R @ v3
            out[STATIC_DIM + 3] = sp
        out[STATIC_DIM + 4] = 1.0
        self.prev = (t, c)
        return out


def mirror_frame(hand, handedness, pose=None, world=None):
    """좌우 반전 증강: 이미지 x → 1-x (world 는 x → -x), 왼손↔오른손, Pose 좌우 관절 교환.
    world 를 주면 (hand, handedness, pose, world) 4개를 돌려줌."""
    h = None if hand is None else np.asarray(hand, np.float32).reshape(21, 3).copy()
    if h is not None:
        h[:, 0] = 1.0 - h[:, 0]
    w = None if world is None else np.asarray(world, np.float32).reshape(21, 3).copy()
    if w is not None:
        w[:, 0] = -w[:, 0]
    p = None
    if pose is not None:
        p = np.asarray(pose, np.float32).reshape(33, 4).copy()
        p[:, 0] = 1.0 - p[:, 0]
        p[[11, 12]] = p[[12, 11]]
    hd = 'Left' if handedness == 'Right' else 'Right'
    if world is not None:
        return h, hd, p, w
    return h, hd, p


def random_rotation(rng, max_deg=40.0):
    """증강용 3D 랜덤 회전 (각 축 ±max_deg)"""
    ax, ay, az = np.deg2rad(rng.uniform(-max_deg, max_deg, 3))
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx, az


def augment_frames(hands, worlds, rng, max_deg=40.0, scale=(0.85, 1.15), noise_img=0.003, noise_world=0.003):
    """시퀀스 전체에 같은 변환: world 3D 회전 + 노이즈, 이미지 좌표 roll 회전·스케일(손목 기준) + 노이즈.
    hands/worlds: list of (21,3) | None. → (hands, worlds) 새 리스트"""
    R, roll = random_rotation(rng, max_deg)
    s = rng.uniform(*scale)
    c, sn = np.cos(roll), np.sin(roll)
    R2 = np.array([[c, -sn], [sn, c]])
    out_h, out_w = [], []
    for h, w in zip(hands, worlds):
        if h is None:
            out_h.append(None)
            out_w.append(None)
            continue
        h = np.asarray(h, np.float32).copy()
        o = h[WRIST, :2].copy()
        h[:, :2] = (h[:, :2] - o) @ R2.T * s + o + rng.normal(0, noise_img, (21, 2))
        out_h.append(h)
        if w is not None and np.all(np.isfinite(w)):
            out_w.append((np.asarray(w, np.float64) @ R.T * s + rng.normal(0, noise_world, (21, 3))).astype(np.float32))
        else:
            out_w.append(None)
    return out_h, out_w
