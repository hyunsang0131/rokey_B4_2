"""노트북에서 바로 (이미 설치된 scikit-learn): 손 '모양'(정지 자세) 분류 MLP → .joblib

  source ~/cobot2_ws/install/setup.bash
  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/train_gesture_sklearn.py
      [--data ~/data/assist_arm/gestures] [--out ~/data/assist_arm/models/gesture_static.joblib]
      [--sources webcam realsense] [--augment 3] [--holdout 사람이름]
특징: 손 기준 좌표(회전 정규화) + 손바닥 방향 (common/features.static_feature, 버전 2). 웹캠·RealSense 녹화를 섞어서 학습 가능.
gesture_node 의 model_path 에 결과 파일을 주면 됨 (ONNX 시퀀스 모델이 있으면 그쪽이 우선).
"""
import argparse
import os
import sys

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from gesture_dataset import frame_dataset, load_samples  # noqa: E402

try:
    from assist_arm.common.features import FEATURE_VERSION
except ImportError:
    from features import FEATURE_VERSION  # noqa: E402


def make_clf():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400,
                                                         early_stopping=True, random_state=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='~/data/assist_arm/gestures')
    ap.add_argument('--out', default='~/data/assist_arm/models/gesture_static.joblib')
    ap.add_argument('--sources', nargs='*', default=None, help='webcam / realsense (기본: 전부)')
    ap.add_argument('--augment', type=int, default=3, help='샘플당 증강 개수 (3D 회전·스케일·노이즈)')
    ap.add_argument('--holdout', default='', help='이 사람 데이터로만 평가')
    a = ap.parse_args()

    samples = load_samples(a.data, a.sources)
    if not samples:
        sys.exit(f'샘플 없음: {a.data} (live_record.py 로 먼저 녹화)')
    X, y, who, src = frame_dataset(samples, mirror=True, augment=a.augment)
    print(f'샘플 {len(samples)}개 → 프레임 {len(X)}개, 클래스 {sorted(set(y))}, 카메라 {sorted(set(src))}')
    groups = np.array([f'{w}_{i // 60}' for i, w in enumerate(who)])
    if a.holdout:
        te = who == a.holdout
    else:
        _, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(X, y, groups))
        te = np.zeros(len(X), bool)
        te[te_i] = True
    clf = make_clf().fit(X[~te], y[~te])
    metrics = {}
    if te.any():
        pred = clf.predict(X[te])
        metrics = {'test_acc': float((pred == y[te]).mean()), 'n_test': int(te.sum()), 'n_train': int((~te).sum())}
        print(classification_report(y[te], pred, zero_division=0))
        labs = sorted(set(y))
        print('confusion (행=정답, 열=예측)', labs)
        print(confusion_matrix(y[te], pred, labels=labs))
    clf.fit(X, y)
    clf.feature_space_ = 'canonical'          # gesture_node 가 같은 특징(static_feature)을 쓰도록
    clf.feature_version_ = FEATURE_VERSION
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    joblib.dump(clf, out)
    print('저장:', out)
    try:                                   # DB 에 모델 등록 (실패해도 모델 파일은 저장됨)
        from assist_arm.common.datadb import DataDB
        DataDB().add_model('gesture', out, dataset_query=f'data={a.data} sources={a.sources} augment={a.augment} '
                           f'holdout={a.holdout} feature_v={FEATURE_VERSION}',
                           metrics={**metrics, 'labels': sorted(set(y)), 'samples': len(samples)})
    except Exception as e:  # noqa: BLE001
        print('DB 등록 생략:', e)


if __name__ == '__main__':
    main()
