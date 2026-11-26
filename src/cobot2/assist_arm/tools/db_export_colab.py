"""DB → Colab 용 내보내기: 테이블별 CSV + 참조된 관절 파일(npz)을 zip 하나로.

  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/db_export_colab.py [--labels clear stop] [--sources realsense]
          [--split train val] [--with_videos] [--out ~/data/assist_arm/export]
결과: ~/data/assist_arm/export/assist_arm_<시각>.zip
  assist_arm/
    gestures/<라벨>/*.npz          ← 학습 노트북이 그대로 읽는 구조 (colab_gesture_seq_gru.ipynb)
    features.py, gesture_dataset.py, colab_gesture_seq_gru.ipynb
    db/<테이블>.csv                 ← sessions, gesture_samples(zip 안 경로 rel_path 포함), clips, decision_events, voice_utterances, models
    clips/<라벨>/*.mp4             ← --with_videos 일 때만 (크기 큼)
Colab: zip 을 Drive 내 드라이브에 올리고 노트북 첫 셀에서  !unzip -q /content/drive/MyDrive/assist_arm_<시각>.zip -d /content/drive/MyDrive/
"""
import argparse
import csv
import io
import os
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from assist_arm.common.datadb import DataDB  # noqa: E402

TABLES = ['sessions', 'gesture_samples', 'clips', 'decision_events', 'voice_utterances', 'models']


def to_csv(rows):
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return buf.getvalue().encode('utf-8-sig')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='')
    ap.add_argument('--labels', nargs='*')
    ap.add_argument('--sources', nargs='*')
    ap.add_argument('--split', nargs='*', help='train / val / test (기본 전부)')
    ap.add_argument('--with_videos', action='store_true')
    ap.add_argument('--out', default='~/data/assist_arm/export')
    a = ap.parse_args()
    db = DataDB(a.db) if a.db else DataDB()
    if not db.ok:
        sys.exit('DB 를 열 수 없어요')
    where, params = [], []
    for col, vals in (('g.label', a.labels), ('s.source', a.sources), ('g.split', a.split)):
        if vals:
            where.append(f"{col} IN ({','.join('?' * len(vals))})")
            params += vals
    q = ('SELECT g.*, s.person, s.source FROM gesture_samples g LEFT JOIN sessions s ON g.session_id = s.id'
         + (' WHERE ' + ' AND '.join(where) if where else '') + ' ORDER BY g.id')
    samples = db.query(q, params)
    out_dir = os.path.expanduser(a.out)
    os.makedirs(out_dir, exist_ok=True)
    zpath = os.path.join(out_dir, f'assist_arm_{time.strftime("%Y%m%d_%H%M%S")}.zip')
    missing = 0
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for r in samples:
            if not os.path.exists(r['file_path']):
                missing += 1
                r['rel_path'] = ''
                continue
            rel = f"gestures/{r['label']}/{os.path.basename(r['file_path'])}"
            z.write(r['file_path'], 'assist_arm/' + rel)
            r['rel_path'] = rel
        for t in TABLES:
            rows = samples if t == 'gesture_samples' else db.query(f'SELECT * FROM {t} ORDER BY id')
            z.writestr(f'assist_arm/db/{t}.csv', to_csv(rows))
        src = os.path.join(HERE, '..')
        for f, arc in ((os.path.join(src, 'assist_arm', 'common', 'features.py'), 'features.py'),
                       (os.path.join(HERE, 'gesture_dataset.py'), 'gesture_dataset.py'),
                       (os.path.join(HERE, 'colab_gesture_seq_gru.ipynb'), 'colab_gesture_seq_gru.ipynb')):
            z.write(f, 'assist_arm/' + arc)
        n_vid = 0
        if a.with_videos:
            for c in db.query('SELECT * FROM clips ORDER BY id'):
                if os.path.exists(c['video_path']):
                    z.write(c['video_path'], f"assist_arm/clips/{c['label'] or 'misc'}/{os.path.basename(c['video_path'])}")
                    n_vid += 1
    by_label = {}
    for r in samples:
        by_label[r['label']] = by_label.get(r['label'], 0) + 1
    print(f'샘플 {len(samples)}개 {by_label} (파일 없음 {missing}), 영상 {n_vid}개')
    print('zip:', zpath, f'({os.path.getsize(zpath) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
