"""식탁 사진 찍기 (YOLO 객체인식 학습용). 로봇은 움직이지 않음 — 관찰 자세에 둔 채로 찍기.

  RealSense (먼저 다른 터미널에서 realsense):
    source /opt/ros/jazzy/setup.bash
    ~/venvs/assist_mp/bin/python ~/cobot2_ws/src/cobot2/assist_arm/tools/capture_table_images.py --source realsense
  웹캠/USB 캠:
    ~/venvs/assist_mp/bin/python ~/cobot2_ws/src/cobot2/assist_arm/tools/capture_table_images.py --device 0
키: space 한 장, a 자동(--interval 초마다) 켜기/끄기, u 방금 사진 지우기, q 종료
저장: ~/corecode/yolo_train/assist_table/images/<source>_<시각>.jpg  (컨테이너에서는 /yolo_train/assist_table — 자동 라벨용)
      DB yolo_images 에도 기록
팁: 그릇·컵·물병·수저·젓가락·휴지·행주를 매번 다르게 놓고, 조명·손·가려짐을 섞어서 200~500장. 빈 그릇/채운 그릇 둘 다.
"""
import argparse
import os
import sys
import time

import cv2
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from assist_arm.common.datadb import DataDB  # noqa: E402

CFG = os.path.join(HERE, '..', 'config', 'yolo_classes.yaml')


def main():
    cfg = yaml.safe_load(open(CFG))
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['webcam', 'realsense'], default='realsense')
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--interval', type=float, default=2.0, help='a(자동) 간격 초')
    ap.add_argument('--out', default=os.path.join(cfg['dataset_dir'], 'images'))
    ap.add_argument('--no_db', action='store_true')
    ap.add_argument('--preview', action='store_true', help='창 없이 한 장 저장 후 종료 (테스트용)')
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(out, exist_ok=True)
    if a.source == 'realsense':
        from live_record import RealsenseSource
        cam = RealsenseSource('/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw',
                              '/camera/color/camera_info')
    else:
        from live_record import WebcamSource
        cam = WebcamSource(a.device)
    db = None if a.no_db else DataDB()
    sid = None
    auto, t_last, last, n = False, 0.0, None, 0

    def save(frame):
        nonlocal sid, last, n
        path = os.path.join(out, f'{a.source}_{time.strftime("%Y%m%d_%H%M%S")}_{int(time.time() * 1000) % 1000:03d}.jpg')
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if db is not None and db.ok:
            if sid is None:
                sid = db.start_session('table', a.source, device=a.device if a.source == 'webcam' else 'D435i',
                                       notes='capture_table_images')
            db.add_yolo_image(sid, path, frame.shape[1], frame.shape[0], a.source)
        last, n = path, n + 1
        print(f'[{n}] {path}')

    while True:
        frame, _ = cam.read()
        if frame is None:
            continue
        if a.preview:
            save(frame)
            break
        now = time.monotonic()
        if auto and now - t_last > a.interval:
            save(frame)
            t_last = now
        vis = frame.copy()
        cv2.putText(vis, f"[{a.source}] saved {n}  space=shot a=auto({'on' if auto else 'off'}) u=undo q=quit",
                    (10, 30), 0, 0.7, (0, 255, 255), 2)
        cv2.imshow('capture_table_images', vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord(' '):
            save(frame)
        elif k == ord('a'):
            auto, t_last = not auto, 0.0
        elif k == ord('u') and last and os.path.exists(last):
            os.remove(last)
            print('삭제:', last)
            last, n = None, max(0, n - 1)
    cam.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
