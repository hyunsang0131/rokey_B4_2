"""/user/landmarks → 제스처 분류 → /user/gesture (assist_arm_msgs/Gesture)

분류 우선순위 (priority 파라미터, 파일이 없는 것은 건너뜀)
  1) seq   : Colab GRU 모션 모델 ONNX (최근 30프레임 손 21관절 + 상체) — tools/colab_gesture_seq_gru.ipynb
  2) knn   : 즉석 등록 샘플 (사용자 개인화)
  3) model : 정지 자세 sklearn(.joblib) 또는 ONNX — tools/train_gesture_sklearn.py
  4) rule  : 규칙 기반 fallback (features.rule_gesture)
녹화 라벨(poke/clear/meal_done/cup_grab/point_seat)은 시스템 라벨(point/push/thumbs_up/cup_grasp)로 바뀜
연속 confirm_frames 프레임 같은 라벨이면 confirmed=True.

즉석 등록:
  ros2 service call /gesture/register assist_arm_msgs/srv/RegisterGesture \
    "{label: push, duration_sec: 5.0, clear: true}"
  → 그 동안 손 모양을 보여주면 특징을 모아 knn_store_path(npz)에 저장
"""
import json
import os
import threading
import time
from collections import Counter, deque

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Bool, String

from assist_arm_msgs.msg import Gesture, UserLandmarks
from assist_arm_msgs.srv import RegisterGesture
from assist_arm.common.features import (DEFAULT_LABELS, FEATURE_VERSION, GESTURE_ALIASES, SEQ_FEATURE_DIM,
                                        SEQ_LEN, STATIC_DIM, SeqFeaturizer, rule_gesture, shape_feature,
                                        static_feature)
from assist_arm.common.heartbeat import attach_heartbeat


class ModelClassifier:
    def __init__(self, path, labels):
        self.labels = labels
        self.kind = None
        self.space = 'image'
        ext = os.path.splitext(path)[1].lower()
        if ext == '.onnx':
            import onnxruntime as ort
            self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
            self.inp = self.sess.get_inputs()[0].name
            self.kind = 'onnx'
        elif ext in ('.joblib', '.pkl'):
            import joblib
            self.clf = joblib.load(path)
            self.kind = 'sklearn'
            self.space = getattr(self.clf, 'feature_space_', 'image')   # 예전 모델은 이미지 좌표
            if hasattr(self.clf, 'classes_'):
                self.labels = [str(c) for c in self.clf.classes_]
        else:
            raise ValueError(f'지원하지 않는 모델 형식: {path}')

    def predict(self, feat):
        x = feat.reshape(1, -1).astype(np.float32)
        if self.kind == 'onnx':
            out = np.asarray(self.sess.run(None, {self.inp: x})[0]).reshape(-1)
            if out.min() < 0 or abs(out.sum() - 1.0) > 1e-3:
                e = np.exp(out - out.max())
                out = e / e.sum()
            i = int(out.argmax())
            return self.labels[i], float(out[i])
        if hasattr(self.clf, 'predict_proba'):
            pr = self.clf.predict_proba(x)[0]
            i = int(pr.argmax())
            return self.labels[i], float(pr[i])
        return str(self.clf.predict(x)[0]), 1.0


class SeqModel:
    """GRU 시퀀스 ONNX (입력 (1, T, F)) — 최근 T 프레임의 frame_feature 로 '모션' 분류"""

    def __init__(self, path, labels, space='image'):
        import onnxruntime as ort
        self.space = space
        self.sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        shp = inp.shape
        self.T = shp[1] if isinstance(shp[1], int) else SEQ_LEN
        self.F = shp[2] if isinstance(shp[2], int) else SEQ_FEATURE_DIM
        self.labels = labels
        self.bufs = {}               # 카메라 소스별 (webcam / realsense) 최근 T 프레임

    def push(self, feat, src='webcam'):
        self.bufs.setdefault(src, deque(maxlen=self.T)).append(feat)

    def predict(self, src='webcam'):
        buf = self.bufs.get(src)
        if buf is None or len(buf) < self.T:
            return None, 0.0
        x = np.stack(buf)[None].astype(np.float32)
        pr = np.asarray(self.sess.run(None, {self.name: x})[0]).reshape(-1)
        if pr.min() < 0 or abs(pr.sum() - 1.0) > 1e-3:
            e = np.exp(pr - pr.max())
            pr = e / e.sum()
        i = int(pr.argmax())
        return self.labels[i], float(pr[i])


class KnnStore:
    def __init__(self, path, k=3, reject_dist=1.5):
        self.path = os.path.expanduser(path)
        self.k, self.reject = k, reject_dist
        self.X = np.zeros((0, STATIC_DIM), np.float32)
        self.y = np.zeros((0,), dtype=object)
        if self.path and os.path.exists(self.path):
            d = np.load(self.path, allow_pickle=True)
            if d['X'].ndim == 2 and d['X'].shape[1] == STATIC_DIM:
                self.X, self.y = d['X'].astype(np.float32), d['y']
            else:
                print(f'[gesture_node] kNN 파일이 예전 특징 형식이라 무시합니다 (다시 등록하면 덮어씀): {self.path}')

    def __len__(self):
        return len(self.y)

    def add(self, label, feats, clear=False):
        if clear:
            keep = self.y != label
            self.X, self.y = self.X[keep], self.y[keep]
        feats = np.asarray(feats, np.float32)
        self.X = np.vstack([self.X, feats])
        self.y = np.concatenate([self.y, np.array([label] * len(feats), dtype=object)])
        if self.path:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            np.savez(self.path, X=self.X, y=self.y)

    def predict(self, feat):
        d = np.linalg.norm(self.X - feat[None], axis=1)
        idx = np.argsort(d)[:self.k]
        if d[idx[0]] > self.reject:
            return 'idle', 0.0
        lab, cnt = Counter(self.y[idx]).most_common(1)[0]
        return str(lab), cnt / len(idx)


class GestureNode(Node):
    def __init__(self):
        super().__init__('gesture_node')
        p = self.declare_parameter
        p('landmarks_topic', '/user/landmarks')
        p('gesture_topic', '/user/gesture')
        p('rs_landmarks_topic', '/rs/landmarks')   # RealSense 손 (rs_hand_node) — 관찰 자세에서 주 인식
        p('rs_timeout_sec', 0.5)
        p('seq_model_path', '~/data/assist_arm/models/gesture_seq.onnx')        # Colab GRU (모션), 없으면 건너뜀
        p('seq_labels_path', '~/data/assist_arm/models/gesture_seq_labels.json')
        p('model_path', '~/data/assist_arm/models/gesture_static.joblib')       # 정지 자세 (.joblib/.onnx), 없으면 건너뜀
        p('labels_path', '')            # 정지 자세 ONNX 용 라벨 json
        p('priority', ['seq', 'knn', 'model', 'rule'])   # 위에서부터 쓸 수 있는 것 사용
        p('confirm_frames', 15)         # 15fps 기준 약 1초
        p('min_confidence', 0.6)
        p('use_knn', True)
        p('knn_store_path', '~/data/assist_arm/knn_gestures.npz')
        p('knn_k', 3)
        p('knn_reject_dist', 1.5)
        p('register_default_sec', 5.0)
        p('palm_sign', 1.0)             # stop(손바닥) / push(손등) 가 뒤바뀌면 -1.0
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        labels = list(DEFAULT_LABELS)
        if g('labels_path') and os.path.exists(os.path.expanduser(g('labels_path'))):
            with open(os.path.expanduser(g('labels_path'))) as f:
                labels = json.load(f)
        self.knn = KnnStore(g('knn_store_path'), g('knn_k'), g('knn_reject_dist')) if g('use_knn') else None
        if self.knn is not None and len(self.knn):
            self.get_logger().info(f'kNN 샘플 {len(self.knn)}개 로드')
        self.model = None
        mp = os.path.expanduser(g('model_path'))
        if mp and os.path.exists(mp):
            self.model = ModelClassifier(mp, labels)
            self.get_logger().info(f'정지 자세 모델: {mp} ({self.model.kind}) labels={self.model.labels}')
        self.seq = None
        sp, sl = os.path.expanduser(g('seq_model_path')), os.path.expanduser(g('seq_labels_path'))
        if sp and os.path.exists(sp) and os.path.exists(sl):
            with open(sl) as f:
                meta = json.load(f)
            if isinstance(meta, list):          # 예전 형식: 라벨 목록만 → 이미지 좌표 특징
                meta = {'labels': meta, 'feature_space': 'image'}
            if int(meta.get('feature_version', 1)) != FEATURE_VERSION:
                self.get_logger().error(f'모션 모델이 예전 특징 형식(v{meta.get("feature_version", 1)})이라 안 씀 — '
                                        f'Colab 노트북으로 다시 학습하세요 (v{FEATURE_VERSION})')
            else:
                self.seq = SeqModel(sp, meta['labels'], 'canonical')
            self.get_logger().info(f'모션(GRU) 모델: {sp} T={self.seq.T} 특징={self.seq.space} labels={self.seq.labels}')
        self.priority = list(g('priority'))
        avail = [k for k in self.priority if (k == 'seq' and self.seq) or (k == 'model' and self.model)
                 or (k == 'knn' and self.knn is not None) or k == 'rule']
        self.get_logger().info(f'분류 순서: {avail} (kNN 은 등록 샘플이 있을 때만)')

        self.hist = {'webcam': deque(maxlen=int(g('confirm_frames'))),
                     'realsense': deque(maxlen=int(g('confirm_frames')))}
        self.observing, self.busy, self.rs_t = False, False, -1e9
        self.featurizers = {}      # 카메라별 SeqFeaturizer (속도 계산용 이전 프레임)
        self.min_conf = float(g('min_confidence'))
        self._rec_lock = threading.Lock()
        self._rec_buf = None
        self._handed = 'Right'

        cbg = ReentrantCallbackGroup()
        self.pub = self.create_publisher(Gesture, g('gesture_topic'), 10)
        self.create_subscription(UserLandmarks, g('landmarks_topic'), lambda m: self.on_lm(m, 'webcam'), 10,
                                 callback_group=cbg)
        self.create_subscription(UserLandmarks, g('rs_landmarks_topic'), lambda m: self.on_lm(m, 'realsense'), 10,
                                 callback_group=cbg)
        self.create_subscription(Bool, '/assist/observing', lambda m: setattr(self, 'observing', m.data), 10,
                                 callback_group=cbg)
        self.create_subscription(String, '/assist/state', self.on_state, 10, callback_group=cbg)
        self.create_service(RegisterGesture, '/gesture/register', self.on_register,
                            callback_group=cbg)
        attach_heartbeat(self)

    def on_state(self, m):
        s = m.data.split(' ', 1)[0]
        self.busy = s not in ('IDLE', 'HOME', 'OBSERVING', 'SAFETY_STOP')

    def classify(self, feats, pts, src='webcam'):
        """feats: {'world': 63 | None, 'image': 63 | None} — 분류기마다 학습 때 쓴 특징 공간을 사용."""
        def pick(space):
            return feats.get(space) if feats.get(space) is not None else feats.get('image')
        # priority 순서대로 쓸 수 있는 분류기 사용. 녹화 라벨(poke, clear...)은 시스템 라벨로 변환
        for k in self.priority:
            if k == 'seq' and self.seq is not None:
                lab, c = self.seq.predict(src)
                if lab is not None:
                    return GESTURE_ALIASES.get(lab, lab), c, 'seq'
            elif k == 'knn' and self.knn is not None and len(self.knn) and pts is not None:
                lab, c = self.knn.predict(feats['canonical'])
                return GESTURE_ALIASES.get(lab, lab), c, 'knn'
            elif k == 'model' and self.model is not None and pts is not None:
                lab, c = self.model.predict(pick(self.model.space))
                return GESTURE_ALIASES.get(lab, lab), c, self.model.kind
            elif k == 'rule' and pts is not None:
                lab, c = rule_gesture(pts, self._handed, float(self.get_parameter('palm_sign').value))
                return lab, c, 'rule'
        return 'idle', 1.0, 'none'

    def on_lm(self, msg, src='webcam'):
        """src: webcam / realsense.
        관찰 자세 + RealSense 손 있음 → RealSense 가 제스처 담당, 웹캠은 'stop' 만 (안전)
        로봇 이동 중 (작업 중, 관찰 아님)   → 웹캠은 'stop' 만
        그 외 (홈에서 대기 등)               → 웹캠 제스처 전부"""
        now = time.monotonic()
        if src == 'realsense':
            if not self.observing:
                return
            if msg.hand_valid:
                self.rs_t = now
        rs_active = self.observing and now - self.rs_t < self.get_parameter('rs_timeout_sec').value
        stop_only = src == 'webcam' and (rs_active or (self.busy and not self.observing))
        out = Gesture()
        out.header = msg.header
        pts = np.asarray(msg.hand, np.float32).reshape(21, 3) if msg.hand_valid else None
        world = (np.asarray(msg.hand_world, np.float32).reshape(21, 3)
                 if msg.hand_valid and msg.hand_world_valid else None)
        hd = msg.handedness or 'Right'
        if self.seq is not None:
            fz = self.featurizers.setdefault(src, SeqFeaturizer())
            self.seq.push(fz.step(pts, hd, world, now), src)
        feats = {'world': None, 'image': None, 'canonical': None}
        if pts is not None:
            feats['canonical'] = static_feature(pts, hd, world)       # 손 기준 좌표 (버전 2, 카메라 무관)
            feats['image'] = shape_feature(pts, hd, None, 'image')[0]  # 예전 모델 호환
            if world is not None:
                feats['world'] = shape_feature(pts, hd, world, 'world')[0]
            self._handed = msg.handedness
            with self._rec_lock:
                if self._rec_buf is not None:
                    self._rec_buf.append(feats['canonical'])
        out.label, out.confidence, out.source = self.classify(feats, pts, src)
        if stop_only and out.label != 'stop':
            out.label, out.confidence = 'idle', 1.0        # 이동 중/RealSense 담당 중 웹캠은 '멈춰'만
        lab = out.label if out.confidence >= self.min_conf else 'idle'
        hist = self.hist[src]
        hist.append(lab)
        out.confirmed = (len(hist) == hist.maxlen and len(set(hist)) == 1 and lab != 'idle')
        out.source = f'{out.source}@{src}'
        if src == 'webcam' and rs_active and lab == 'idle':
            return          # RealSense 가 담당 중이면 웹캠의 idle 은 내보내지 않음 (확정 판정 섞이지 않게)
        self.pub.publish(out)

    def on_register(self, req, res):
        if self.knn is None:
            res.success, res.message = False, 'use_knn=false'
            return res
        dur = req.duration_sec if req.duration_sec > 0 else float(
            self.get_parameter('register_default_sec').value)
        self.get_logger().info(f"제스처 '{req.label}' 등록: {dur:.1f}초 동안 보여주세요")
        with self._rec_lock:
            self._rec_buf = []
        time.sleep(dur)
        with self._rec_lock:
            feats, self._rec_buf = self._rec_buf, None
        if len(feats) < 5:
            res.success, res.message = False, f'손이 거의 안 보였습니다 ({len(feats)} 프레임)'
            return res
        self.knn.add(req.label, feats, clear=req.clear)
        for h in self.hist.values():
            h.clear()
        res.success, res.num_samples = True, len(feats)
        res.message = f"'{req.label}' {len(feats)}개 저장 (총 {len(self.knn)}개)"
        self.get_logger().info(res.message)
        return res


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
