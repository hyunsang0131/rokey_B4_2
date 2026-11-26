"""실행기(executor) 단위 시험 — 가짜 task_manager 로 관찰→결정→실행→DB 기록 흐름 확인.

robot_io 가 geometry_msgs·od_msg 를 import 하므로 ROS 환경을 불러온 터미널에서 실행:
  source ~/rokey_B4_2/install/setup.bash && cd src/cobot2/assist_arm && python3 -m pytest test/ -q
"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from assist_arm.common.datadb import DataDB  # noqa: E402
from assist_arm.execution.executor import run_plan  # noqa: E402
from assist_arm.execution.robot_io import SafetyAbort, TaskFailed  # noqa: E402
from assist_arm.execution.table_planner import TablePlanner, seat_slots  # noqa: E402

CFG = os.path.join(os.path.dirname(__file__), '..', 'config')
LAYOUT = yaml.safe_load(open(os.path.join(CFG, 'table_layout.yaml')))
WP = yaml.safe_load(open(os.path.join(CFG, 'waypoints.yaml')))
SEAT1 = [600.0, -150.0]


class FakeTM:
    """executor 가 쓰는 task_manager 기능만 흉내 (로봇·ROS 없음)"""

    def __init__(self, db, dets, fail_pick=0, abort_at=None):
        self.db, self.layout = db, LAYOUT
        self.planner = TablePlanner(LAYOUT, workspace=LAYOUT['planner_workspace'])
        self.dets = dets
        self.dry_removed, self.holding, self.at_observe = set(), None, False
        self.fail_pick, self.abort_at = fail_pick, abort_at
        self.said, self.moves = [], []
        self.params = {'seats_file': '/nonexistent/seats.yaml', 'poke_radius_mm': 120.0}

    def g(self, n):
        return self.params[n]

    def check(self):
        if self.abort_at is not None and len(self.moves) >= self.abort_at:
            raise SafetyAbort()

    def observe_table(self):
        self.at_observe = True

    def detect_all(self, classes):
        return [dict(d) for d in self.dets if d['cls'] in classes and d['id'] not in self.dry_removed]

    def pick(self, item, xyz, z_offset=0.0):
        self.check()
        if self.fail_pick:
            self.fail_pick -= 1
            raise TaskFailed('그리퍼 폭 이상')
        self.moves.append(('pick', item, z_offset))
        self.holding = item

    def place(self, item, xy, slow=False, z_offset=0.0):
        self.check()
        self.moves.append(('place', item, tuple(round(v) for v in xy), z_offset))
        self.holding = None

    def recover_holding(self):
        self.holding = None
        return True

    def say(self, t):
        self.said.append(t)

    def state_note(self, s, extra=''):
        pass


def det(i, cls, xy):
    return {'id': i, 'cls': cls, 'xyz': [xy[0], xy[1], 30.0], 'score': 0.9, 'fill': 0.0}


@pytest.fixture
def db(tmp_path):
    return DataDB(str(tmp_path / 'x.db'), quiet=True)


def test_clear_stacks_by_type_and_logs(db):
    rice = seat_slots(LAYOUT, SEAT1)
    rice_xy = next(xy for it, xy in rice if it == 'rice_bowl')
    tm = FakeTM(db, [det(1, 'bowl', rice_xy), det(2, 'rice_bowl', [550, 100]), det(3, 'spoon', [520, 200])])
    status = run_plan(tm, 'clear', seats={1: {'xy': SEAT1, 'state': 'DONE'}})
    assert status == 'DONE'
    inv = db.inventory()
    assert inv['stack_B'] == ['rice_bowl', 'rice_bowl']          # bowl 은 자리 템플릿으로 밥그릇 판정
    assert inv['holder_spoon'] == ['spoon']
    places = [m for m in tm.moves if m[0] == 'place' and m[1] == 'rice_bowl']
    assert places[0][3] == -2.0 and places[1][3] == 13.0          # 선반(-2 mm), 두 번째는 한 개 높이만큼 위
    e = db.query('SELECT * FROM episodes')[0]
    assert e['status'] == 'DONE' and e['success'] == 1 and e['n_actions'] == 3     # 옮긴 물체 3개
    assert len(db.query('SELECT 1 FROM task_log')) == 6                            # pick·place 3쌍
    assert tm.said[-1] == '다 치웠어요'


def test_pick_failures_stop_with_blocked(db):
    tm = FakeTM(db, [det(1, 'rice_bowl', [550, 0])], fail_pick=5)
    assert run_plan(tm, 'clear', seats={}) == 'BLOCKED'
    rows = db.query('SELECT success, error_code FROM task_log')
    assert len(rows) == 3 and all(r['success'] == 0 for r in rows)
    assert db.query('SELECT status FROM episodes')[0]['status'] == 'BLOCKED'


def test_one_failure_then_success(db):
    tm = FakeTM(db, [det(1, 'rice_bowl', [550, 0])], fail_pick=1)
    assert run_plan(tm, 'clear', seats={}) == 'DONE'
    assert db.inventory('stack_B') == ['rice_bowl']


def test_clear_dish_only_pointed_object(db):
    tm = FakeTM(db, [det(1, 'rice_bowl', [550, 0]), det(2, 'soup_bowl', [600, 200])])
    assert run_plan(tm, 'clear', seats={}, target_xy=[600, 210], max_actions=1) == 'DONE'
    assert db.inventory() == {'stack_A': ['soup_bowl']}


def test_clear_dish_nothing_near_point(db):
    tm = FakeTM(db, [det(1, 'rice_bowl', [550, 0])])
    assert run_plan(tm, 'clear', seats={}, target_xy=[300, 400], max_actions=1) == 'BLOCKED'
    assert '가리킨 곳' in tm.said[-1]


def test_set_table_uses_inventory(db):
    for loc, items in LAYOUT['initial_inventory'].items():
        db.inventory_set(loc, list(items))
    tm = FakeTM(db, [])
    seats = {1: {'xy': SEAT1, 'state': 'EMPTY'}}
    assert run_plan(tm, 'set_table', seats=seats) == 'DONE'
    need = [it for it, _ in seat_slots(LAYOUT, SEAT1) if it in tm.planner.storable()]   # 보관 자리가 있는 종류만
    picked = [m[1] for m in tm.moves if m[0] == 'pick']
    assert sorted(picked) == sorted(need)
    rice_pick = next(m for m in tm.moves if m[0] == 'pick' and m[1] == 'rice_bowl')
    assert rice_pick[2] == 13.0                                   # 선반(-2) 위 2개 쌓인 더미의 맨 위
    assert len(db.inventory('stack_B')) == 1


def test_safety_abort_marks_episode(db):
    tm = FakeTM(db, [det(1, 'rice_bowl', [550, 0])], abort_at=1)
    with pytest.raises(SafetyAbort):
        run_plan(tm, 'clear', seats={})
    assert db.query('SELECT status FROM episodes')[0]['status'] == 'ABORTED'
