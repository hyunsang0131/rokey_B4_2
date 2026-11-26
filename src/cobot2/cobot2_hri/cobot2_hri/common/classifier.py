"""손동작 분류기 (assist_arm 에서 가져온 것과 같은 전처리·모델 형식).

순서 (있는 것만): seq(Colab GRU ONNX, 모션) → model(노트북 sklearn .joblib, 손 모양) → rule(규칙)
학습 데이터·모델은 assist_arm/tools 의 live_record / train_gesture_sklearn / Colab 노트북으로 만든 것을 그대로 씀
  (~/data/assist_arm/models/gesture_static.joblib, gesture_seq.onnx + gesture_seq_labels.json)
녹화 라벨 → 시스템 라벨: poke·point_seat→point, clear→push, meal_done→thumbs_up, cup_grab→cup_grasp (hold_cup 그대로)
"""
import json
import os
from collections import deque

import numpy as np

from cobot2_hri.common.features import (FEATURE_VERSION, GESTURE_ALIASES, SEQ_FEATURE_DIM, SEQ_LEN, SeqFeaturizer,
                                        rule_gesture, static_feature)


class GestureClassifier:
    def __init__(self, static_model='', seq_model='', seq_labels='', palm_sign=1.0, logger=None):
        self.log = logger
        self.palm_sign = palm_sign
        self.static = self.seq = None
        self.labels = []
        sm = os.path.expanduser(static_model or '')
        if sm and os.path.exists(sm):
            import joblib
            self.static = joblib.load(sm)
            if getattr(self.static, 'feature_space_', '') != 'canonical':
                self._say(f'손 모양 모델이 예전 특징 형식이라 안 씀: {sm}')
                self.static = None
            else:
                self._say(f'손 모양 모델: {sm} {list(self.static.classes_)}')
        qm, ql = os.path.expanduser(seq_model or ''), os.path.expanduser(seq_labels or '')
        if qm and ql and os.path.exists(qm) and os.path.exists(ql):
            meta = json.load(open(ql))
            if isinstance(meta, dict) and int(meta.get('feature_version', 1)) == FEATURE_VERSION:
                import onnxruntime as ort
                self.seq = ort.InferenceSession(qm, providers=['CPUExecutionProvider'])
                self.labels = meta['labels']
                inp = self.seq.get_inputs()[0]
                self.T = inp.shape[1] if isinstance(inp.shape[1], int) else SEQ_LEN
                self._say(f'모션 모델: {qm} {self.labels}')
            else:
                self._say(f'모션 모델 라벨 형식이 예전 버전이라 안 씀: {ql}')
        self.feat = SeqFeaturizer()
        self.buf = deque(maxlen=getattr(self, 'T', SEQ_LEN))
        self._dim = SEQ_FEATURE_DIM

    def _say(self, m):
        if self.log:
            self.log.info(m)

    def source(self):
        return 'seq' if self.seq is not None else ('sklearn' if self.static is not None else 'rule')

    def step(self, pts, handedness='Right', world=None, t=0.0):
        """한 프레임 → (라벨, 확신도). pts: (21,3) 정규화 좌표 | None"""
        if self.seq is not None:
            self.buf.append(self.feat.step(pts, handedness, world, t))
            if len(self.buf) == self.buf.maxlen:
                x = np.stack(self.buf)[None].astype(np.float32)
                pr = np.asarray(self.seq.run(None, {self.seq.get_inputs()[0].name: x})[0]).reshape(-1)
                if pr.min() < 0 or abs(pr.sum() - 1) > 1e-3:
                    pr = np.exp(pr - pr.max())
                    pr /= pr.sum()
                i = int(pr.argmax())
                lab = self.labels[i]
                return GESTURE_ALIASES.get(lab, lab), float(pr[i])
        if pts is None:
            return 'idle', 1.0
        if self.static is not None:
            f = static_feature(pts, handedness, world).reshape(1, -1)
            pr = self.static.predict_proba(f)[0]
            i = int(pr.argmax())
            lab = str(self.static.classes_[i])
            return GESTURE_ALIASES.get(lab, lab), float(pr[i])
        return rule_gesture(pts, handedness, self.palm_sign)
