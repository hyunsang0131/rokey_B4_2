"""녹화한 관절 시퀀스(npz) → 학습 데이터. (노트북 sklearn / Colab GRU / 교차 평가 공용)

  ~/data/assist_arm/gestures/<label>/*.npz   (tools/live_record.py 로 녹화)
    hand (T,21,3) 화면 좌표, hand_world (T,21,3) 미터 3D (없을 수 있음), handedness (T,), hand_valid (T,),
    fps, label, person, source(webcam/realsense, 없으면 webcam)
전처리는 assist_arm/common/features.py (static_feature / SeqFeaturizer, 버전 2) — 추론(gesture_node)과 똑같음.
  손 기준 좌표(손목 원점 + 손바닥 축 회전 정규화) → 카메라(웹캠·USB 캠·RealSense)가 바뀌어도 비슷한 값.
증강: 좌우 반전, 3D 랜덤 회전(±40°), 스케일, 노이즈 (augment 배수만큼 추가 샘플).
Colab 에서는 features.py 와 이 파일을 같은 폴더에 두고 import.
"""
import glob
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if os.path.isdir(os.path.join(_HERE, '..', 'assist_arm')):
    sys.path.insert(0, os.path.join(_HERE, '..'))       # 노트북: 패키지 소스 (빌드·source 안 해도 됨)
try:
    from assist_arm.common.features import (SEQ_LEN, SeqFeaturizer, augment_frames, mirror_frame,
                                            static_feature)
except ImportError:          # Colab: features.py 를 같은 폴더에 둔 경우
    sys.path.insert(0, _HERE)
    from features import SEQ_LEN, SeqFeaturizer, augment_frames, mirror_frame, static_feature  # noqa: E402


def load_samples(root, sources=None):
    """sources: None(전부) 또는 ['webcam', 'realsense'] 처럼 고르기"""
    out = []
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(root), '*', '*.npz'))):
        d = np.load(path, allow_pickle=False)
        src = str(d['source']) if 'source' in d.files else 'webcam'
        if sources and src not in sources:
            continue
        out.append({'label': str(d['label']), 'person': str(d['person']), 'source': src, 'path': path,
                    'hand': d['hand'], 'hand_valid': d['hand_valid'].astype(bool),
                    'handedness': d['handedness'].astype(str),
                    'hand_world': d['hand_world'] if 'hand_world' in d.files else None,
                    'fps': float(d['fps']) if 'fps' in d.files else 15.0})
    return out


def _frames(s, mirror=False):
    hands, worlds, hds = [], [], []
    for t in range(len(s['hand'])):
        h = s['hand'][t] if s['hand_valid'][t] else None
        w = None
        if h is not None and s['hand_world'] is not None and np.isfinite(s['hand_world'][t]).all():
            w = s['hand_world'][t]
        hd = s['handedness'][t] or 'Right'
        if mirror and h is not None:
            if w is not None:
                h, hd, _, w = mirror_frame(h, hd, None, w)
            else:
                h, hd, _ = mirror_frame(h, hd, None)
        hands.append(h)
        worlds.append(w)
        hds.append(hd)
    return hands, worlds, hds


def _variants(s, mirror, augment, rng):
    """원본(+반전) + 증강 n개 → [(hands, worlds, hds)]"""
    base = [_frames(s, False)] + ([_frames(s, True)] if mirror else [])
    out = list(base)
    for _ in range(augment):
        h, w, hd = base[rng.integers(len(base))]
        h2, w2 = augment_frames(h, w, rng)
        out.append((h2, w2, hd))
    return out


def sequence_dataset(samples, seq_len=SEQ_LEN, mirror=True, augment=0, seed=0):
    """→ X (N, seq_len, SEQ_FEATURE_DIM), y (N,), persons (N,), sources (N,)"""
    rng = np.random.default_rng(seed)
    X, y, who, src = [], [], [], []
    for s in samples:
        dt = 1.0 / max(1.0, s['fps'])
        for hands, worlds, hds in _variants(s, mirror, augment, rng):
            f = SeqFeaturizer()
            seq = np.stack([f.step(h, hd, w, i * dt) for i, (h, w, hd) in enumerate(zip(hands, worlds, hds))])
            if len(seq) < seq_len:
                seq = np.vstack([np.zeros((seq_len - len(seq), seq.shape[1]), np.float32), seq])
            X.append(seq[-seq_len:])
            y.append(s['label'])
            who.append(s['person'])
            src.append(s['source'])
    return np.array(X, np.float32), np.array(y), np.array(who), np.array(src)


def frame_dataset(samples, mirror=True, augment=0, seed=0):
    """정지 자세용: 손이 보이는 프레임마다 static_feature (STATIC_DIM). → X, y, persons, sources"""
    rng = np.random.default_rng(seed)
    X, y, who, src = [], [], [], []
    for s in samples:
        for hands, worlds, hds in _variants(s, mirror, augment, rng):
            for h, w, hd in zip(hands, worlds, hds):
                if h is None:
                    continue
                X.append(static_feature(h, hd, w))
                y.append(s['label'])
                who.append(s['person'])
                src.append(s['source'])
    return np.array(X, np.float32), np.array(y), np.array(who), np.array(src)


def make_fake_samples(root, labels=('idle', 'clear', 'stop', 'poke'), per_label=8, sources=('webcam', 'realsense'),
                      seed=0):
    """실제 데이터가 없을 때 동작 확인용 가짜 녹화 (같은 손 모양을 카메라마다 다른 각도로)"""
    rng = np.random.default_rng(seed)
    base = {}
    for lab in labels:
        p = rng.normal(0, 0.03, (21, 3)).astype(np.float32)
        p[0] = 0
        for k in range(1, 21):
            p[k] += [0.01 * (k % 4), -0.02 * (k // 4 + 1), 0.0]
        base[lab] = p
    for src in sources:
        R_cam = np.linalg.qr(rng.normal(size=(3, 3)))[0] if src == 'realsense' else np.eye(3)
        for lab in labels:
            os.makedirs(os.path.join(root, lab), exist_ok=True)
            for i in range(per_label):
                T = 30
                world = np.stack([(base[lab] + rng.normal(0, 0.002, (21, 3))) @ R_cam.T for _ in range(T)])
                hand = np.stack([np.c_[0.5 + w[:, 0] * 2, 0.6 + w[:, 1] * 2, w[:, 2]] for w in world])
                if lab == 'clear':                       # 손을 옆으로 휘젓는 움직임
                    hand[:, :, 0] += np.linspace(-0.15, 0.15, T)[:, None]
                np.savez_compressed(os.path.join(root, lab, f'fake_{src}_{i}.npz'), hand=hand.astype(np.float32),
                                    hand_world=world.astype(np.float32), hand_valid=np.ones(T, bool),
                                    handedness=np.array(['Right'] * T), pose=np.zeros((T, 33, 4), np.float32),
                                    pose_valid=np.zeros(T, bool), fps=15.0, label=lab, person=f'p{i % 2}',
                                    source=src)
