"""보관 재고(DB storage_inventory) 보기·맞추기 — 로봇 없이 실행 (v2.2 6.6: 포개진 그릇은 안 보이므로 DB 가 기준).

  python3 tools/inventory.py show                       # 지금 재고
  python3 tools/inventory.py seed                       # table_layout.yaml 의 initial_inventory 로 전부 다시 채움
  python3 tools/inventory.py set stack_B rice_bowl rice_bowl   # 한 자리를 [아래 → 위] 로 맞춤
  python3 tools/inventory.py clear [자리]               # 한 자리(또는 전부) 비우기
  python3 tools/inventory.py episodes [N]               # 최근 작업 N개와 동작 기록
DB 위치: ~/data/assist_arm/assist_arm.db (환경변수 ASSIST_ARM_DB)
"""
import argparse
import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from assist_arm.common.datadb import DataDB  # noqa: E402


def layout():
    with open(os.path.join(HERE, '..', 'config', 'table_layout.yaml')) as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['show', 'seed', 'set', 'clear', 'episodes'])
    ap.add_argument('rest', nargs='*')
    a = ap.parse_args()
    db = DataDB()
    if not db.ok:
        sys.exit('DB 를 열 수 없어요')
    if a.cmd == 'seed':
        init = layout().get('initial_inventory', {}) or {}
        for loc in list(db.inventory()):
            db.inventory_set(loc, [])
        for loc, items in init.items():
            db.inventory_set(loc, list(items))
    elif a.cmd == 'set':
        if not a.rest:
            sys.exit('자리 이름이 필요해요 (예: set stack_B rice_bowl rice_bowl)')
        db.inventory_set(a.rest[0], a.rest[1:])
    elif a.cmd == 'clear':
        for loc in (a.rest or list(db.inventory())):
            db.inventory_set(loc, [])
    elif a.cmd == 'episodes':
        n = int(a.rest[0]) if a.rest else 5
        for e in db.query('SELECT * FROM episodes ORDER BY id DESC LIMIT ?', (n,)):
            print(f"#{e['id']} {e['started_at']} {e['mode']} ({e['planner']}) → {e['status']} "
                  f"동작 {e['n_actions']} {e['reason'] or ''}")
            for t in db.query('SELECT * FROM task_log WHERE episode_id=? ORDER BY id', (e['id'],)):
                ok = '✓' if t['success'] else '✗'
                print(f"   {ok} {t['skill']:12s} {t['object_type']:10s} {t['src']} → {t['dst']} {t['error_code']}")
        return
    inv = db.inventory()
    print('보관 재고 (아래 → 위)' if inv else '보관 재고 없음')
    for loc in sorted(inv):
        print(f'  {loc:14s} {inv[loc]}')


if __name__ == '__main__':
    main()
