"""카메라(source)별 교차 평가: 한 카메라로 학습 → 다른 카메라로 테스트 정확도 표.

  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/eval_cross_source.py            # 실제 녹화 데이터
  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/eval_cross_source.py --fake     # 가짜 데이터로 동작만 확인
정지 자세(손 모양, MLP)와 모션(30프레임 시퀀스를 평탄화한 MLP — GRU 대신 빠른 근사) 두 가지.
예: train=realsense → test=webcam 이 높으면 웹캠·USB 캠만으로도 어느 정도 인식됨.
"""
import argparse
import os
import sys
import tempfile

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gesture_dataset import frame_dataset, load_samples, make_fake_samples, sequence_dataset  # noqa: E402


def clf():
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=500, random_state=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='~/data/assist_arm/gestures')
    ap.add_argument('--fake', action='store_true')
    ap.add_argument('--augment', type=int, default=3)
    a = ap.parse_args()
    root = os.path.expanduser(a.data)
    if a.fake:
        root = tempfile.mkdtemp(prefix='fake_gestures_')
        make_fake_samples(root)
        print('가짜 데이터:', root)
    samples = load_samples(root)
    sources = sorted({s['source'] for s in samples})
    if not samples:
        sys.exit('샘플 없음')
    print(f"샘플 {len(samples)}개, 카메라 {sources}, 동작 {sorted({s['label'] for s in samples})}\n")
    for kind in ('정지 자세', '모션(시퀀스)'):
        print(f'[{kind}] 행=학습 카메라, 열=테스트 카메라 (정확도)')
        print(f"{'':12}" + ''.join(f'{t:>12}' for t in sources + ['전체']))
        for tr in sources + ['전체']:
            row = f'{tr:12}'
            for te in sources + ['전체']:
                trs = [s for s in samples if tr == '전체' or s['source'] == tr]
                tes = [s for s in samples if te == '전체' or s['source'] == te]
                # 같은 카메라끼리는 사람/파일 단위로 절반씩 나눔 (같은 파일이 학습·테스트에 동시에 들어가지 않게)
                if tr == te or '전체' in (tr, te):
                    paths = sorted({s['path'] for s in tes})
                    test_paths = set(paths[1::2])
                    trs = [s for s in trs if s['path'] not in test_paths]
                    tes = [s for s in tes if s['path'] in test_paths]
                if not trs or not tes:
                    row += f"{'-':>12}"
                    continue
                if kind == '정지 자세':
                    Xtr, ytr, _, _ = frame_dataset(trs, augment=a.augment)
                    Xte, yte, _, _ = frame_dataset(tes, mirror=False)
                else:
                    Xtr, ytr, _, _ = sequence_dataset(trs, augment=a.augment)
                    Xte, yte, _, _ = sequence_dataset(tes, mirror=False)
                    Xtr, Xte = Xtr[:, ::3].reshape(len(Xtr), -1), Xte[:, ::3].reshape(len(Xte), -1)
                m = clf().fit(Xtr, ytr)
                row += f'{(m.predict(Xte) == yte).mean():>12.0%}'
            print(row)
        print()


if __name__ == '__main__':
    main()
