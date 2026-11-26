"""수위 측정 영역 클릭 설정 → config/pour_level.yaml (ROS 불필요).

  python3 ~/cobot2_ws/src/cobot2/cobot2_hri/tools/pour_level_roi.py --device 2 [--out ~/cobot2_ws/src/cobot2/cobot2_hri/config/pour_level.yaml]
  (영상 파일로: --video cup.mp4)
순서: 창에서 ① 컵 안쪽 왼쪽 위 ② 컵 안쪽 오른쪽 아래(=바닥) ③ '가득' 높이 를 클릭 → s 저장, r 다시, q 취소
저장 후 colcon build --packages-select cobot2_hri (또는 pour_level_node 에 -p config_file:=<경로>)
"""
import argparse
import os

import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('--device', type=int, default=2)
ap.add_argument('--video', default='')
ap.add_argument('--out', default=os.path.join(HERE, '..', 'config', 'pour_level.yaml'))
a = ap.parse_args()
cap = cv2.VideoCapture(a.video or a.device)
clicks = []
cv2.namedWindow('pour_level_roi')
cv2.setMouseCallback('pour_level_roi', lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3 else None)
hints = ['click: cup inner TOP-LEFT', 'click: cup inner BOTTOM-RIGHT (bottom)', "click: 'FULL' level height", 's=save r=reset q=quit']
while True:
    ok, f = cap.read()
    if not ok:
        if a.video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    for p in clicks:
        cv2.circle(f, p, 4, (0, 0, 255), -1)
    if len(clicks) >= 2:
        cv2.rectangle(f, clicks[0], clicks[1], (255, 200, 0), 1)
    if len(clicks) == 3:
        cv2.line(f, (clicks[0][0], clicks[2][1]), (clicks[1][0], clicks[2][1]), (0, 255, 0), 1)
    cv2.putText(f, hints[len(clicks)], (10, 30), 0, 0.7, (0, 255, 255), 2)
    cv2.imshow('pour_level_roi', f)
    k = cv2.waitKey(30) & 0xFF
    if k == ord('r'):
        clicks.clear()
    elif k == ord('q'):
        break
    elif k == ord('s') and len(clicks) == 3:
        (x1, y1), (x2, y2), (_, yf) = clicks
        c = yaml.safe_load(open(a.out)) if os.path.exists(a.out) else {}
        c.update({'roi': [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], 'y_bottom': max(y1, y2) - 1,
                  'y_full': int(yf), 'device': a.device})
        yaml.safe_dump(c, open(a.out, 'w'), allow_unicode=True)
        print('저장:', a.out, c)
        break
cap.release()
cv2.destroyAllWindows()
