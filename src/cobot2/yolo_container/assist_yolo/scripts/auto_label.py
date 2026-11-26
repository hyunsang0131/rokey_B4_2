"""[컨테이너 안에서 실행] COCO 기본 모델로 식탁 사진에 초벌 박스 → YOLO 라벨(txt). 새 패키지 설치 없음 (이미 있는 ultralytics).

호스트에서:
  docker start yolo-detection
  docker exec yolo-detection python3 /ros2_ws/src/assist_yolo/scripts/auto_label.py \
      --dataset /yolo_train/assist_table --weights /yolo_train/Fruits/yolov8n.pt
데이터 폴더(호스트 ~/corecode/yolo_train/assist_table)에 필요한 것: images/*.jpg, coco_map.json, data.yaml
  (호스트에서 tools/yolo_dataset.py prepare 가 만들어 줌)
결과: labels/<사진이름>.txt (YOLO: 클래스 cx cy w h, 0~1), auto_labels.json (초벌로 만든 파일 목록)
이미 라벨이 있는 사진은 건드리지 않음 (사람이 고친 것 보호). 다시 하려면 --overwrite
"""
import argparse
import glob
import json
import os

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='/yolo_train/assist_table')
    ap.add_argument('--weights', default='/yolo_train/Fruits/yolov8n.pt')
    ap.add_argument('--overwrite', action='store_true')
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.dataset, 'coco_map.json')))
    names, cmap, conf = cfg['classes'], cfg['coco_map'], float(cfg.get('prelabel_conf', 0.25))
    model = YOLO(a.weights)
    lab_dir = os.path.join(a.dataset, 'labels')
    os.makedirs(lab_dir, exist_ok=True)
    man_path = os.path.join(a.dataset, 'auto_labels.json')
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    imgs = sorted(glob.glob(os.path.join(a.dataset, 'images', '*.jpg')) + glob.glob(os.path.join(a.dataset, 'images', '*.png')))
    done = skipped = boxes = 0
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        out = os.path.join(lab_dir, stem + '.txt')
        if os.path.exists(out) and not a.overwrite:
            skipped += 1
            continue
        res = model(img, conf=conf, verbose=False)[0]
        lines = []
        for b, c in zip(res.boxes.xywhn.tolist(), res.boxes.cls.tolist()):
            coco_name = res.names[int(c)]
            if coco_name in cmap and cmap[coco_name] in names:
                lines.append(f'{names.index(cmap[coco_name])} ' + ' '.join(f'{v:.6f}' for v in b))
        with open(out, 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        manifest[stem] = len(lines)
        done += 1
        boxes += len(lines)
    json.dump(manifest, open(man_path, 'w'), indent=1)
    print(f'초벌 라벨: {done}장 (박스 {boxes}개), 이미 라벨 있어 건너뜀 {skipped}장 → {lab_dir}')


if __name__ == '__main__':
    main()
