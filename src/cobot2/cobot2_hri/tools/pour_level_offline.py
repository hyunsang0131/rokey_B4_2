"""녹화 영상으로 수위 추정 오프라인 테스트 (ROS 불필요).

  python3 ~/cobot2_ws/src/cobot2/cobot2_hri/tools/pour_level_offline.py pour.mp4 [--config .../pour_level.yaml]
        [--empty_frames 5] [--out_video result.mp4]
→ 프레임별 ratio·confidence CSV (영상 옆에 _level.csv), 결과 영상(수면 선 그림). --empty_frames N: 처음 N 프레임 평균을 빈 컵 기준으로
"""
import argparse
import csv
import os
import sys

import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from cobot2_hri.common.level import LevelEstimator  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('video')
ap.add_argument('--config', default=os.path.join(HERE, '..', 'config', 'pour_level.yaml'))
ap.add_argument('--empty_frames', type=int, default=0)
ap.add_argument('--out_video', default='')
a = ap.parse_args()
c = yaml.safe_load(open(a.config))
est = LevelEstimator(c['roi'], c['y_bottom'], c['y_full'])
cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS) or 15
out_csv = os.path.splitext(a.video)[0] + '_level.csv'
vw, rows, i = None, [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    if i < a.empty_frames:
        est.set_empty(f)
    r, conf = est.update(f)
    rows.append((i, round(i / fps, 3), round(r, 4), round(conf, 3)))
    if a.out_video:
        if vw is None:
            vw = cv2.VideoWriter(a.out_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (f.shape[1], f.shape[0]))
        vw.write(est.draw(f))
    i += 1
with open(out_csv, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['frame', 't_sec', 'ratio', 'confidence'])
    w.writerows(rows)
if vw:
    vw.release()
low = sum(1 for r in rows if r[3] < 0.35)
print(f'{len(rows)} 프레임, 마지막 수위 {rows[-1][2] if rows else "-"}, 확신도 낮은 프레임 {low} → {out_csv}')
