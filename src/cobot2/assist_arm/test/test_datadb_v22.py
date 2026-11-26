"""datadb v2.2 테이블 단위 시험: python3 -m pytest test/ -q"""
import os
import sqlite3
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from assist_arm.common.datadb import SCHEMA_VERSION, DataDB  # noqa: E402


def test_episode_and_task_log(tmp_path):
    db = DataDB(str(tmp_path / 'a.db'), quiet=True)
    assert db.ok
    ep = db.start_episode('clear', 'rule', people=2)
    assert db.log_task(ep, 'pick', 'rice_bowl', 'seat1', '', success=True)
    assert db.log_task(ep, 'place_stack', 'rice_bowl', 'seat1', 'stack_B', '', success=True)
    assert db.log_task(ep, 'pick', 'soup_bowl', 'seat1', '', success=False, error_code='GRASP_FAIL')
    db.finish_episode(ep, 'BLOCKED', reason='soup_bowl:out_of_workspace', n_actions=1, duration_s=12.5)
    e = db.query('SELECT * FROM episodes WHERE id=?', (ep,))[0]
    assert e['status'] == 'BLOCKED' and e['success'] == 0 and e['n_actions'] == 1 and e['reason']
    assert len(db.query('SELECT * FROM task_log WHERE episode_id=?', (ep,))) == 3


def test_done_counts_as_success(tmp_path):
    db = DataDB(str(tmp_path / 'b.db'), quiet=True)
    ep = db.start_episode('set_table', 'rl', people=2)
    db.finish_episode(ep, 'DONE')
    assert db.query('SELECT success FROM episodes')[0]['success'] == 1


def test_inventory_stack_order(tmp_path):
    db = DataDB(str(tmp_path / 'c.db'), quiet=True)
    assert db.inventory_push('stack_B', 'rice_bowl') == 0
    assert db.inventory_push('stack_B', 'rice_bowl') == 1
    db.inventory_push('holder_spoon', 'spoon')
    assert db.inventory('stack_B') == ['rice_bowl', 'rice_bowl']
    assert db.inventory() == {'stack_B': ['rice_bowl', 'rice_bowl'], 'holder_spoon': ['spoon']}
    assert db.inventory_pop('stack_B') == 'rice_bowl'
    assert db.inventory('stack_B') == ['rice_bowl']
    assert db.inventory_pop('empty_place') is None
    assert db.inventory_set('stack_A', ['soup_bowl'] * 3)
    assert db.inventory('stack_A') == ['soup_bowl'] * 3


def test_user_pref(tmp_path):
    db = DataDB(str(tmp_path / 'd.db'), quiet=True)
    db.set_pref('minsu', 'handedness', 'left')
    db.set_pref('minsu', 'handedness', 'right')
    db.set_pref('minsu', 'seat', 1)
    assert db.get_pref('minsu', 'handedness') == 'right'
    assert db.get_pref('minsu', 'seat') == 1
    assert db.get_pref('nobody', 'seat', 'x') == 'x'


def test_existing_v1_db_upgrades(tmp_path):
    """예전(v1) DB 파일에 새 테이블이 추가되고 기존 줄은 남는다"""
    p = str(tmp_path / 'old.db')
    c = sqlite3.connect(p)
    c.executescript("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);"
                    "INSERT INTO meta VALUES ('schema_version', '1');"
                    "CREATE TABLE sessions (id INTEGER PRIMARY KEY AUTOINCREMENT, person TEXT, source TEXT, "
                    "device TEXT, started_at TEXT, notes TEXT);"
                    "INSERT INTO sessions (person, source) VALUES ('lee', 'realsense');")
    c.commit()
    c.close()
    db = DataDB(p, quiet=True)
    assert db.ok
    assert db.query("SELECT value FROM meta WHERE key='schema_version'")[0]['value'] == str(SCHEMA_VERSION)
    assert db.counts()['sessions'] == 1 and db.counts()['episodes'] == 0
