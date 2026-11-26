"""table_planner 단위 시험 (ROS·로봇 없이): python3 -m pytest test/ -q   (assist_arm 패키지 폴더에서)"""
import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from assist_arm.execution.table_planner import (  # noqa: E402
    ACT, BLOCKED, DONE, EATING, PERCEPTION_FAIL, SEAT_DONE, History, TablePlanner, seat_slots)

LAYOUT = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'config', 'table_layout.yaml')))
WORKSPACE = {'min': [150.0, -600.0, 0.0], 'max': [800.0, 600.0, 500.0]}
SEAT1 = [600.0, -150.0]
SEAT2 = [600.0, 150.0]


def det(cls, x, y):
    return {'cls': cls, 'xyz': [x, y, 0.0], 'score': 0.9, 'fill': 0.0}


def slot(seat, item):
    return next(xy for it, xy in seat_slots(LAYOUT, seat) if it == item)


@pytest.fixture
def planner():
    return TablePlanner(LAYOUT, WORKSPACE)


def test_done_when_nothing_to_clear(planner):
    r = planner.next_action([det('cup', 500, 0)], {}, mode='clear')   # cup 은 clear_classes 에 없음
    assert r.status == DONE


def test_detect_failure(planner):
    assert planner.next_action(None, {}).status == PERCEPTION_FAIL


def test_holding_blocks_planning(planner):
    r = planner.next_action([det('rice_bowl', 500, 0)], {}, holding='rice_bowl')
    assert r.status == BLOCKED and r.reason.startswith('holding')


def test_rice_bowl_goes_to_its_stack(planner):
    r = planner.next_action([det('rice_bowl', 500, -100)], {})
    assert r.status == ACT
    a = r.action
    assert a.item == 'rice_bowl' and a.dst == 'stack_B' and a.dst_site == 'stack' and a.dst_count == 0


def test_stack_count_and_top_from_inventory(planner):
    r = planner.next_action([det('soup_bowl', 500, -100)], {'stack_A': ['soup_bowl', 'soup_bowl']})
    assert r.action.dst == 'stack_A' and r.action.dst_count == 2 and r.action.dst_top_type == 'soup_bowl'


def test_bowl_classified_by_seat_template(planner):
    seats = {1: {'xy': SEAT1, 'state': SEAT_DONE}}
    xy = slot(SEAT1, 'rice_bowl')
    r = planner.next_action([det('bowl', xy[0] + 20, xy[1])], {}, seats=seats)
    assert r.action.item == 'rice_bowl' and r.action.dst == 'stack_B'


def test_unknown_bowl_goes_to_temp(planner):
    r = planner.next_action([det('bowl', 450, 300)], {})
    assert r.status == ACT and r.action.item == 'bowl' and r.action.dst_site == 'temp'


def test_eating_seat_is_not_touched(planner):
    seats = {1: {'xy': SEAT1, 'state': EATING}, 2: {'xy': SEAT2, 'state': SEAT_DONE}}
    near1 = slot(SEAT1, 'rice_bowl')
    near2 = slot(SEAT2, 'rice_bowl')
    r = planner.next_action([det('rice_bowl', *near1), det('rice_bowl', *near2)], {}, seats=seats)
    assert r.action.src == 'seat2'
    r = planner.next_action([det('rice_bowl', *near1)], {}, seats=seats)
    assert r.status == DONE          # 먹는 중인 자리 것만 남음 → 이번 작업 대상 없음


def test_blocked_when_out_of_workspace(planner):
    r = planner.next_action([det('rice_bowl', 1500, 0)], {})
    assert r.status == BLOCKED and 'out_of_workspace' in r.reason


def test_blocked_when_storage_full(planner):
    full = {'stack_B': ['rice_bowl'] * 4, 'temp_1': ['x'], 'temp_2': ['x']}
    r = planner.next_action([det('rice_bowl', 500, 0)], full)
    assert r.status == BLOCKED and 'no_space' in r.reason


def test_full_stack_falls_back_to_temp(planner):
    r = planner.next_action([det('rice_bowl', 500, 0)], {'stack_B': ['rice_bowl'] * 4})
    assert r.action.dst_site == 'temp'


def test_vanished_without_pick_is_perception_fail(planner):
    h = History(prev_dets=[det('rice_bowl', 500, 0)])
    assert planner.next_action([], {}, history=h).status == PERCEPTION_FAIL


def test_vanished_with_pick_record_is_done(planner):
    h = History(prev_dets=[det('rice_bowl', 500, 0)], picked=[('rice_bowl', [500, 0])])
    assert planner.next_action([], {}, history=h).status == DONE


def test_rule_picks_closest_to_robot(planner):
    r = planner.next_action([det('rice_bowl', 700, 0), det('soup_bowl', 400, 0)], {})
    assert r.action.item == 'soup_bowl'


def test_policy_hook_overrides_rule(planner):
    planner.policy = lambda obs, cands: 0          # 후보는 검출 순서 → 0번 = 먼 rice_bowl (규칙이면 soup_bowl)
    r = planner.next_action([det('rice_bowl', 700, 0), det('soup_bowl', 400, 0)], {})
    assert r.action.item == 'rice_bowl'


def test_set_table_far_seat_first_then_done(planner):
    seats = {1: {'xy': [450.0, -100.0], 'state': 'EMPTY'}, 2: {'xy': [650.0, 100.0], 'state': 'EMPTY'}}
    storage = {'stack_B': ['rice_bowl'] * 4, 'stack_C': ['plate'] * 4, 'holder_spoon': ['spoon'] * 4,
               'holder_fork': ['fork'] * 4}
    LAYOUT_SET = dict(LAYOUT)
    LAYOUT_SET['seat_layout'] = [{'item': 'rice_bowl', 'offset': [80.0, -60.0], 'from': 'rack'},
                                 {'item': 'spoon', 'offset': [20.0, -150.0], 'from': 'rack'}]
    p = TablePlanner(LAYOUT_SET, WORKSPACE)
    h = History()
    order = []
    for _ in range(10):
        r = p.next_action([], storage, seats=seats, mode='set_table', history=h)
        if r.status != ACT:
            break
        k = int(r.action.dst.split(':')[0][4:])
        order.append(k)
        h.placed.add((k, r.action.item))
    assert r.status == DONE and order[:2] == [2, 2] and len(order) == 4


def test_set_table_out_of_stock_blocks(planner):
    seats = {1: {'xy': SEAT1, 'state': 'EMPTY'}}
    r = planner.next_action([], {}, seats=seats, mode='set_table')
    assert r.status == BLOCKED and 'out_of_stock' in r.reason


def test_unknown_mode_raises(planner):
    with pytest.raises(ValueError):
        planner.next_action([], {}, mode='dance')
