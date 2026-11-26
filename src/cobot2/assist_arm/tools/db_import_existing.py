"""이미 있는 데이터 파일을 DB(~/data/assist_arm/assist_arm.db)에 등록 (읽기만, 파일은 옮기거나 지우지 않음).

  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/db_import_existing.py [--root ~/data/assist_arm] [--dry]
등록 대상 (이미 등록된 경로는 건너뜀 → 여러 번 돌려도 중복 없음)
  gestures/<라벨>/*.npz  → gesture_samples (사람·카메라별 'imported' 세션)
  clips/**/*.mp4         → clips (gestures 와 같은 이름이면 같은 세션·라벨)
  models/*.joblib|*.onnx → models (gesture / intent 추정)
  decision/*.npz         → models (kind=decision, LinUCB 사람별 학습 상태)
"""
import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from assist_arm.common.datadb import DataDB  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='~/data/assist_arm')
    ap.add_argument('--db', default='')
    ap.add_argument('--dry', action='store_true', help='등록하지 않고 무엇을 등록할지만 보여주기')
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    db = DataDB(a.db) if a.db else DataDB()
    if not db.ok:
        sys.exit('DB 를 열 수 없어요')
    new = {'gesture_samples': 0, 'clips': 0, 'models': 0}
    skipped = 0
    sessions = {}
    stem_info = {}

    for path in sorted(glob.glob(os.path.join(root, 'gestures', '*', '*.npz'))):
        if db.has_path('gesture_samples', 'file_path', path):
            skipped += 1
            continue
        try:
            d = np.load(path, allow_pickle=False)
            label = str(d['label']) if 'label' in d.files else os.path.basename(os.path.dirname(path))
            person = str(d['person']) if 'person' in d.files else 'unknown'
            source = str(d['source']) if 'source' in d.files else 'webcam'
            hd = [x for x in d['handedness'].astype(str) if x] if 'handedness' in d.files else []
            n = int(len(d['hand'])) if 'hand' in d.files else None
            fps = float(d['fps']) if 'fps' in d.files else None
            ver = 2 if 'source' in d.files else 1
        except Exception as e:  # noqa: BLE001
            print('읽기 실패, 건너뜀:', path, e)
            continue
        stem_info[os.path.splitext(os.path.basename(path))[0]] = (person, source, label)
        if a.dry:
            print('[dry] gesture', label, person, source, path)
            new['gesture_samples'] += 1
            continue
        key = (person, source)
        if key not in sessions:
            sessions[key] = db.find_or_create_session(person, source, 'imported')
        if db.add_gesture_sample(sessions[key], label, path, n_frames=n, fps=fps,
                                 handedness=max(set(hd), key=hd.count) if hd else '', preprocess_ver=ver):
            new['gesture_samples'] += 1

    for path in sorted(glob.glob(os.path.join(root, 'clips', '**', '*.mp4'), recursive=True)):
        if db.has_path('clips', 'video_path', path):
            skipped += 1
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        person, source, label = stem_info.get(stem, ('unknown', 'webcam', ''))
        if not label:
            parent = os.path.basename(os.path.dirname(path))
            label = '' if parent == 'clips' else parent
        dur = None
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
            dur = float(n / fps) if fps else None
            cap.release()
        except Exception:  # noqa: BLE001
            pass
        if a.dry:
            print('[dry] clip', label or '(라벨 없음: 벤치마크 클립)', path)
            new['clips'] += 1
            continue
        key = (person, source)
        if key not in sessions:
            sessions[key] = db.find_or_create_session(person, source, 'imported')
        if db.add_clip(sessions[key], path, label=label, duration_s=dur):
            new['clips'] += 1

    for path in sorted(glob.glob(os.path.join(root, 'models', '*.joblib')) + glob.glob(os.path.join(root, 'models', '*.onnx'))
                       + glob.glob(os.path.join(root, 'decision', '*.npz'))):
        if db.has_path('models', 'path', path):
            skipped += 1
            continue
        kind = 'decision' if '/decision/' in path else ('intent' if 'intent' in os.path.basename(path) else 'gesture')
        if a.dry:
            print('[dry] model', kind, path)
        elif db.add_model(kind, path, dataset_query='imported'):
            new['models'] += 1
        if a.dry:
            new['models'] += 1

    print(f"{'[dry] ' if a.dry else ''}새로 등록: {new}, 이미 있어서 건너뜀: {skipped}")
    print('DB 전체:', db.counts(), '→', db.path)


if __name__ == '__main__':
    main()
