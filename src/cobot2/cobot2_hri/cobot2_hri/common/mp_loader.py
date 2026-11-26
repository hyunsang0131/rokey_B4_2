"""MediaPipe 로더.

시스템 파이썬/도커는 건드리지 않기 위해 MediaPipe 는 별도 venv 에 설치하는 것을 전제로 한다.
ros2 run 으로 실행되는 스크립트는 /usr/bin/python3 로 돌기 때문에, 시스템에 mediapipe 가 없으면
환경변수 ASSIST_MP_VENV (기본 ~/venvs/assist_mp) 의 site-packages 를 sys.path *뒤쪽*에 붙여서
다시 import 한다. (뒤쪽에 붙이므로 시스템 numpy/cv2/rclpy 가 우선 사용된다.)

legacy API(mp.solutions)가 있으면 그것을, 없으면 Tasks API(.task 모델 파일 필요)를 쓴다.
"""
import os
import time

import numpy as np

MP_INSTALL_HINT = (
    "mediapipe 를 찾을 수 없습니다. 시스템 파이썬에는 설치하지 말고 별도 venv 를 권장합니다:\n"
    "  python3 -m venv --system-site-packages ~/venvs/assist_mp\n"
    "  ~/venvs/assist_mp/bin/pip install 'mediapipe==0.10.21' 'numpy<2'\n"
    "venv 경로가 다르면 export ASSIST_MP_VENV=<venv 경로> 후 노드를 실행하세요."
)


class MediaPipeMissing(RuntimeError):
    pass


def import_mediapipe():
    from cobot2_hri.common.venv_import import import_from_venv
    try:
        return import_from_venv('mediapipe')
    except ImportError as e:
        raise MediaPipeMissing(f"{MP_INSTALL_HINT}\n(원인: {e})") from e


def _has_legacy(mp):
    return hasattr(mp, 'solutions') and hasattr(mp.solutions, 'hands')


class HandTracker:
    """rgb(np.uint8 HxWx3) → 손 목록. 손마다 {'pts': (21,3) 정규화 이미지좌표, 'world': (21,3) 미터 3D | None,
    'handedness': 'Left'/'Right', 'score'}. world = MediaPipe hand world landmarks (손 중심 원점, 실제 크기)."""

    def __init__(self, max_hands=1, min_det=0.5, min_track=0.5, model_complexity=0,
                 task_model_path='', min_presence=0.5):
        self.mp = import_mediapipe()
        self.legacy = _has_legacy(self.mp)
        if self.legacy:
            self.impl = self.mp.solutions.hands.Hands(
                static_image_mode=False, max_num_hands=max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_det, min_tracking_confidence=min_track)
        else:
            task_model_path = os.path.expanduser(task_model_path or '')
            if not task_model_path or not os.path.exists(task_model_path):
                raise MediaPipeMissing(
                    "설치된 mediapipe 에 legacy API(mp.solutions)가 없습니다. "
                    "hand_landmarker.task 파일 경로를 hand_task_model 파라미터로 주세요 "
                    "(~/data/assist_arm/models/hand_landmarker.task).")
            from mediapipe.tasks.python import BaseOptions, vision
            opts = vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=task_model_path),
                running_mode=vision.RunningMode.VIDEO, num_hands=max_hands,
                min_hand_detection_confidence=min_det, min_hand_presence_confidence=min_presence,
                min_tracking_confidence=min_track)
            self.impl = vision.HandLandmarker.create_from_options(opts)
            self._t0 = time.monotonic()

    def process_all(self, rgb):
        out = []
        if self.legacy:
            res = self.impl.process(rgb)
            if not res.multi_hand_landmarks:
                return out
            worlds = getattr(res, 'multi_hand_world_landmarks', None) or []
            for i, hl in enumerate(res.multi_hand_landmarks):
                cls = res.multi_handedness[i].classification[0]
                w = np.array([[p.x, p.y, p.z] for p in worlds[i].landmark], np.float32) if i < len(worlds) else None
                out.append({'pts': np.array([[p.x, p.y, p.z] for p in hl.landmark], np.float32), 'world': w,
                            'handedness': cls.label, 'score': float(cls.score)})
        else:
            img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = self.impl.detect_for_video(img, int((time.monotonic() - self._t0) * 1000))
            worlds = getattr(res, 'hand_world_landmarks', None) or []
            for i, hl in enumerate(res.hand_landmarks or []):
                cat = res.handedness[i][0]
                w = np.array([[p.x, p.y, p.z] for p in worlds[i]], np.float32) if i < len(worlds) else None
                out.append({'pts': np.array([[p.x, p.y, p.z] for p in hl], np.float32), 'world': w,
                            'handedness': cat.category_name, 'score': float(cat.score)})
        out.sort(key=lambda h: -h['score'])        # 첫 번째 = 가장 확실한 손
        return out

    def process(self, rgb):
        """(예전 호환) 가장 확실한 손 하나 → (pts | None, handedness, score)"""
        hands = self.process_all(rgb)
        if not hands:
            return None, '', 0.0
        return hands[0]['pts'], hands[0]['handedness'], hands[0]['score']

    def close(self):
        try:
            self.impl.close()
        except Exception:
            pass


class PoseTracker:
    """rgb → landmarks (33,4) [x, y, z, visibility] | None"""

    def __init__(self, min_det=0.5, min_track=0.5, model_complexity=0, task_model_path=''):
        self.mp = import_mediapipe()
        self.legacy = _has_legacy(self.mp)
        if self.legacy:
            self.impl = self.mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=model_complexity,
                min_detection_confidence=min_det, min_tracking_confidence=min_track)
        else:
            task_model_path = os.path.expanduser(task_model_path or '')
            if not task_model_path or not os.path.exists(task_model_path):
                raise MediaPipeMissing(
                    "legacy API 가 없는 mediapipe 입니다. pose_landmarker_lite.task 경로를 "
                    "pose_task_model 파라미터로 주거나 mediapipe==0.10.21 을 설치하세요.")
            from mediapipe.tasks.python import BaseOptions, vision
            opts = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=task_model_path),
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=min_det, min_tracking_confidence=min_track)
            self.impl = vision.PoseLandmarker.create_from_options(opts)
            self._t0 = time.monotonic()

    def process(self, rgb):
        if self.legacy:
            res = self.impl.process(rgb)
            if not res.pose_landmarks:
                return None
            return np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark],
                            dtype=np.float32)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self.impl.detect_for_video(img, int((time.monotonic() - self._t0) * 1000))
        if not res.pose_landmarks:
            return None
        return np.array([[p.x, p.y, p.z, getattr(p, 'visibility', 1.0) or 0.0]
                         for p in res.pose_landmarks[0]], dtype=np.float32)

    def close(self):
        try:
            self.impl.close()
        except Exception:
            pass

