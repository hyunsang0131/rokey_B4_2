"""YOLO 식탁 데이터셋 관리 (호스트, 설치 없음). 사진·라벨 파일은 ~/corecode/yolo_train/assist_table 에.

  python3 yolo_dataset.py prepare           # data.yaml·coco_map.json 만들기 + 폴더 사진 DB 등록 → 자동 라벨 명령 안내
  python3 yolo_dataset.py status            # 사진 수, 라벨 상태(none/auto/human), 클래스별 박스 수
  python3 yolo_dataset.py export [--only_labeled]   # Roboflow/CVAT 업로드·Colab 학습용 zip (train/val/test 분할 + data.yaml + 노트북)
  python3 yolo_dataset.py import <고친라벨.zip|폴더>   # Roboflow/CVAT 에서 YOLO 형식으로 내보낸 라벨을 되가져오기 (예전 라벨은 백업)

흐름: capture_table_images.py (찍기) → prepare → 컨테이너 auto_label.py (초벌 박스) → export → 웹 도구에서 고치기
      → import → export → Colab colab_yolo_train.ipynb → best.pt → assist_detection model_path
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time
import zipfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from assist_arm.common.datadb import DataDB  # noqa: E402

CFG = yaml.safe_load(open(os.path.join(HERE, '..', 'config', 'yolo_classes.yaml')))
DS = os.path.expanduser(os.environ.get('ASSIST_YOLO_DATASET', CFG['dataset_dir']))   # 테스트용으로 바꿀 수 있음
NAMES = CFG['classes']


def images():
    return sorted(glob.glob(os.path.join(DS, 'images', '*.jpg')) + glob.glob(os.path.join(DS, 'images', '*.png')))


def label_of(img):
    return os.path.join(DS, 'labels', os.path.splitext(os.path.basename(img))[0] + '.txt')


def n_boxes(path):
    if not os.path.exists(path):
        return 0
    return sum(1 for ln in open(path) if ln.strip())


def sync_db(db):
    """폴더 기준으로 DB 라벨 상태 맞추기: 라벨 없음 none / auto_labels.json 에 있고 안 고침 auto / import 한 것 human"""
    man = json.load(open(os.path.join(DS, 'auto_labels.json'))) if os.path.exists(os.path.join(DS, 'auto_labels.json')) else {}
    human = set(json.load(open(os.path.join(DS, 'human_labels.json')))) if os.path.exists(os.path.join(DS, 'human_labels.json')) else set()
    sid = None
    for img in images():
        if not db.has_path('yolo_images', 'image_path', img):
            sid = sid or db.find_or_create_session('table', 'unknown', 'yolo_dataset')
            db.add_yolo_image(sid, img, source=os.path.basename(img).split('_')[0])
        lab = label_of(img)
        stem = os.path.splitext(os.path.basename(img))[0]
        state = 'human' if stem in human else ('auto' if stem in man and os.path.exists(lab) else
                                                ('human' if os.path.exists(lab) else 'none'))
        db.set_yolo_label(img, lab, state, n_boxes(lab))


def cmd_prepare(a):
    os.makedirs(os.path.join(DS, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DS, 'labels'), exist_ok=True)
    with open(os.path.join(DS, 'data.yaml'), 'w') as f:
        yaml.safe_dump({'path': '.', 'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
                        'names': {i: n for i, n in enumerate(NAMES)}}, f, allow_unicode=True)
    json.dump({'classes': NAMES, 'coco_map': CFG['coco_map'], 'prelabel_conf': CFG.get('prelabel_conf', 0.25)},
              open(os.path.join(DS, 'coco_map.json'), 'w'), indent=1, ensure_ascii=False)
    db = DataDB()
    sync_db(db)
    print(f'준비 완료: {DS} (사진 {len(images())}장, 클래스 {NAMES})')
    print('\n다음: 컨테이너에서 초벌 라벨 (새 설치 없음)')
    print('  docker start yolo-detection')
    print(f"  docker exec yolo-detection python3 /ros2_ws/src/assist_yolo/scripts/auto_label.py "
          f"--dataset {CFG['container_dataset_dir']} --weights /yolo_train/Fruits/yolov8n.pt")
    print('그다음: python3 yolo_dataset.py status → export → 웹 도구에서 고치기')


def cmd_status(a):
    db = DataDB()
    sync_db(db)
    rows = db.query('SELECT label_state, COUNT(*) n, SUM(n_boxes) b FROM yolo_images GROUP BY label_state')
    per_cls = {n: 0 for n in NAMES}
    for img in images():
        lab = label_of(img)
        if os.path.exists(lab):
            for ln in open(lab):
                if ln.strip():
                    k = int(ln.split()[0])
                    if 0 <= k < len(NAMES):
                        per_cls[NAMES[k]] += 1
    print(f'사진 {len(images())}장 ({DS})')
    for r in rows:
        print(f"  라벨 {r['label_state']:6}: {r['n']}장, 박스 {r['b'] or 0}개")
    print('  클래스별 박스:', per_cls)
    few = [k for k, v in per_cls.items() if v < 50]
    if few:
        print('  50개 미만 (더 찍거나 라벨 추가 권장):', few)


def cmd_export(a):
    db = DataDB()
    sync_db(db)
    splits = {r['image_path']: r['split'] for r in db.query('SELECT image_path, split FROM yolo_images')}
    out_dir = os.path.expanduser(a.out)
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, f'yolo_table_{time.strftime("%Y%m%d_%H%M%S")}.zip')
    n = {'train': 0, 'val': 0, 'test': 0}
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for img in images():
            lab = label_of(img)
            if a.only_labeled and not os.path.exists(lab):
                continue
            sp = splits.get(os.path.abspath(img), 'train')
            z.write(img, f'assist_table/{sp}/images/{os.path.basename(img)}')
            if os.path.exists(lab):
                z.write(lab, f'assist_table/{sp}/labels/{os.path.basename(lab)}')
            n[sp] += 1
        z.writestr('assist_table/data.yaml', yaml.safe_dump({
            'path': '.', 'train': 'train/images', 'val': 'val/images', 'test': 'test/images',
            'names': {i: nm for i, nm in enumerate(NAMES)}}, allow_unicode=True))
        z.writestr('assist_table/classes.txt', '\n'.join(NAMES) + '\n')      # CVAT/Roboflow 가져오기용
        nb = os.path.join(HERE, 'colab_yolo_train.ipynb')
        if os.path.exists(nb):
            z.write(nb, 'assist_table/colab_yolo_train.ipynb')
    print(f'zip: {zpath} ({os.path.getsize(zpath) / 1e6:.1f} MB) 분할 {n}')
    print('  Roboflow/CVAT: 새 프로젝트 → YOLO 형식으로 이 zip 업로드 → 박스 고치기 → YOLO 형식으로 내보내기 → import')
    print('  Colab: Drive 에 올리고 colab_yolo_train.ipynb 실행')


def cmd_import(a):
    src = os.path.expanduser(a.path)
    tmp = None
    if src.endswith('.zip'):
        tmp = os.path.join(DS, f'_import_{time.strftime("%H%M%S")}')
        zipfile.ZipFile(src).extractall(tmp)
        src = tmp
    labs = [p for p in glob.glob(os.path.join(src, '**', '*.txt'), recursive=True)
            if os.path.basename(p) not in ('classes.txt', 'obj.names', 'README.roboflow.txt')]
    stems = {os.path.splitext(os.path.basename(i))[0] for i in images()}
    bak = os.path.join(DS, f'labels_bak_{time.strftime("%Y%m%d_%H%M%S")}')
    shutil.copytree(os.path.join(DS, 'labels'), bak, dirs_exist_ok=True)
    human_path = os.path.join(DS, 'human_labels.json')
    human = set(json.load(open(human_path))) if os.path.exists(human_path) else set()
    n = miss = 0
    for p in labs:
        stem = os.path.splitext(os.path.basename(p))[0]
        if stem not in stems:          # Roboflow 는 이름 뒤에 .rf.<해시> 를 붙이기도 함
            stem = stem.split('.rf.')[0].replace('_jpg', '').replace('_png', '')
        if stem not in stems:
            miss += 1
            continue
        shutil.copy2(p, os.path.join(DS, 'labels', stem + '.txt'))
        human.add(stem)
        n += 1
    json.dump(sorted(human), open(human_path, 'w'), indent=1)
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    sync_db(DataDB())
    print(f'고친 라벨 {n}개 가져옴 (사진과 이름이 안 맞아 건너뜀 {miss}개). 예전 라벨 백업: {bak}')
    if CFG['classes'] != NAMES:
        print('주의: 클래스 순서가 config 와 다르면 학습이 틀어져요')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prepare')
    sub.add_parser('status')
    e = sub.add_parser('export')
    e.add_argument('--only_labeled', action='store_true')
    e.add_argument('--out', default='~/data/assist_arm/export')
    i = sub.add_parser('import')
    i.add_argument('path')
    a = ap.parse_args()
    {'prepare': cmd_prepare, 'status': cmd_status, 'export': cmd_export, 'import': cmd_import}[a.cmd](a)


if __name__ == '__main__':
    main()
