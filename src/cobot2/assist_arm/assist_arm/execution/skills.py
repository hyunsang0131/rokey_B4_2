"""기능별 서브태스크. 각 함수는 (tm: TaskManager, args: dict) 를 받아 동작을 끝까지 수행한다.

원칙: 로봇은 사람과 접촉하지 않는다. 사람 쪽으로 가는 물건은 '사용자 앞 / 손 옆 테이블 위'에 내려놓고 후퇴.
      손/팔이 작업 지점 20cm 안이면 safety_monitor 가 일시정지 → 벗어나면 재개 (robot_io.wait_unpaused).
"""
import json
import os
import time

import numpy as np
import yaml
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from assist_arm.execution.robot_io import TaskFailed


def dish_classes(tm):
    return list(tm.layout.get('classes', {}).get('dishes', ['bowl', 'plate']))


def nearest(dets, xy, radius):
    if not dets:
        return None
    d = min(dets, key=lambda e: np.linalg.norm(np.array(e['xyz'][:2]) - xy))
    return d if np.linalg.norm(np.array(d['xyz'][:2]) - xy) <= radius else None


def clear_one(tm, det):
    tm.state_note('CLEAR', f"{det['cls']} @ {np.round(det['xyz'][:2], 0).tolist()}")
    tm.pick(det['cls'], det['xyz'])
    tm.place(det['cls'], tm.clear_slot())
    if det.get('id') is not None:
        tm.dry_removed.add(det['id'])


# ---------------------------------------------------------------------- 1. 콕 → 사용자 앞으로 / 음식 이름 → 사용자 앞으로
def bring_dish(tm, args):
    tm.observe_table()
    if 'point' in args:
        det = nearest(tm.detect_all(dish_classes(tm)), np.array(args['point'][:2]), tm.g('poke_radius_mm'))
        if det is None:
            raise TaskFailed('가리킨 곳 근처에 그릇이 없음')
        item, xyz = det['cls'], det['xyz']
    elif 'food' in args:
        f = tm.layout['foods'].get(args['food'])
        if f is None:
            raise TaskFailed(f"모르는 음식: {args['food']}")
        item = f.get('class', 'bowl')
        if f.get('source', 'fixed') == 'detect':
            xyz = tm.detect_one(f.get('detect_class', item))
        else:
            xyz = np.array(f['pos'][:3], float)
    else:
        raise TaskFailed('bring_dish: point 또는 food 필요')
    tm.say('가져다 드릴게요')
    tm.pick(item, xyz)
    tm.place(item, tm.user_front(), slow=True)


# ---------------------------------------------------------------------- 2. 콕 + 치워 → 그 그릇 정리
def clear_dish(tm, args):
    tm.observe_table()
    det = nearest(tm.detect_all(dish_classes(tm) + ['cup']), np.array(args['point'][:2]),
                  tm.g('poke_radius_mm'))
    if det is None:
        raise TaskFailed('가리킨 곳 근처에 그릇이 없음')
    clear_one(tm, det)


# ---------------------------------------------------------------------- 3. 치워(손등 밀기) → 빈 그릇 스스로 찾아 정리
def clear_empty(tm, args):
    tm.observe_table()
    th = tm.g('empty_fill_threshold')
    dets = [d for d in tm.detect_all(dish_classes(tm)) if 0 <= d['fill'] < th]
    if not dets:
        tm.say('빈 그릇이 없어요')
        return
    tm.say(f'빈 그릇 {len(dets)}개 치울게요')
    for d in sorted(dets, key=lambda e: np.linalg.norm(e['xyz'][:2])):   # 로봇에 가까운 것부터
        clear_one(tm, d)


# ---------------------------------------------------------------------- 4. 식사 끝 → 테이블 전체 정리 (그릇 → 컵 → 수저)
def clear_all(tm, args):
    groups = tm.layout.get('classes', {})
    order = [dish_classes(tm), groups.get('cups', ['cup']), groups.get('utensils', ['spoon', 'fork', 'chopsticks'])]
    tm.say('정리할게요')
    n = 0
    for classes in order:
        for _ in range(int(tm.g('max_clear_items'))):
            tm.observe_table()
            dets = tm.detect_all(classes)
            if not dets:
                break
            clear_one(tm, min(dets, key=lambda e: np.linalg.norm(e['xyz'][:2])))
            n += 1
    tm.say(f'{n}개 정리했어요')


# ---------------------------------------------------------------------- 5. "물 줘" + 컵 잡기 → 따르기 → 손 옆 테이블에 내려놓기
def pour_water(tm, args):
    wp = tm.wp['pour']
    tm.observe_table()
    cup = tm.detect_one(tm.g('cup_class'))
    tm.say('물 따라 드릴게요')
    tm.pick('cup', cup)
    tm.place('cup', wp['cup_spot'])                 # 그리퍼가 하나 → 컵을 고정 자리에 두고 물병으로
    tm.pick('bottle', np.array(wp['bottle_pos'][:3], float))
    for q in wp['pour_waypoints']:                  # 직접교시 궤적 (물병 기울이기)
        tm.movel(q, float(wp.get('pour_vel_mms', 60.0)))
    tm.place('bottle', wp['bottle_pos'][:2])
    tm.pick('cup', np.array(list(wp['cup_spot']) + [0.0], float))
    spot = tm.near_hand_spot(tm.g('cup_near_hand_mm'))
    tm.state_note('PLACE_NEAR_HAND', f'{np.round(spot, 0).tolist()}')
    tm.place('cup', spot, slow=True)                # 손에 쥐여주지 않음: 손 옆 테이블에 두고 후퇴
    tm.say('여기 있어요')


# ---------------------------------------------------------------------- 6. "N명이야" + 자리 콕 N번 → 자리마다 세팅
def seat_frame(seat_xy):
    """자리 좌표계: u = 자리 → 로봇 베이스 방향(테이블 안쪽), v = u 를 왼쪽으로 90°"""
    s = np.array(seat_xy, float)
    u = -s / (np.linalg.norm(s) + 1e-6)
    return s, u, np.array([-u[1], u[0]])


def seat_setting(tm, args):
    seats = args.get('seats', [])
    if not seats:
        raise TaskFailed('자리 좌표 없음')
    path = os.path.expanduser(tm.g('seats_file'))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump({'seats': [list(map(float, s)) for s in seats],
                        'saved': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
    tm.get_logger().info(f'자리 {len(seats)}곳 저장: {path}')
    tm.say(f'{len(seats)}명 자리 세팅할게요')
    tm.rack_used = {}
    for i, seat in enumerate(seats):
        s, u, v = seat_frame(seat)
        for it in tm.layout['seat_layout']:
            off = it.get('offset', [0, 0])
            xy = s + off[0] * u + off[1] * v
            tm.state_note('SEAT', f"{i + 1}번 자리 {it['item']}")
            if it.get('from', 'rack') == 'rack':
                tm.pick_from_rack(it.get('rack', it['item']))
                tm.place(it.get('rack', it['item']), xy)
            else:                                    # 음식 그릇 (foods 정의)
                f = tm.layout['foods'][it['food']]
                src = tm.detect_one(f.get('detect_class', f.get('class', 'bowl'))) \
                    if f.get('source') == 'detect' else np.array(f['pos'][:3], float)
                tm.pick(f.get('class', 'bowl'), src)
                tm.place(f.get('class', 'bowl'), xy)


# ---------------------------------------------------------------------- 7. 음악 추천 + 율동
def music(tm, args):
    d = tm.wp['dance']
    tm.music_play_pub.publish(String(data=json.dumps({'mood': args.get('mood', 'any')}, ensure_ascii=False)))
    t_end = time.monotonic() + 5.0
    while tm.now_playing is None or not tm.now_playing.get('playing'):
        tm.check()
        if time.monotonic() > t_end:
            raise TaskFailed('music_node 응답 없음')
        time.sleep(0.1)
    bpm = float(tm.now_playing.get('bpm') or 100.0)
    beat = 60.0 / bpm
    tm.say(f"{tm.now_playing.get('title', '')} 틀어드릴게요")
    tm.set_safety_mode('dance')
    try:
        tm.movej(d['start_j'])
        prev = np.array(tm.posx()[:3])
        t_stop = time.monotonic() + float(d.get('max_duration_sec', 60.0))
        vmax = min(float(d.get('max_speed_mms', 120.0)), tm.speed('dance_max_speed_mms'))
        while tm.now_playing.get('playing') and time.monotonic() < t_stop:
            for q in d['waypoints']:
                tm.check()
                if not tm.now_playing.get('playing'):
                    break
                dist = float(np.linalg.norm(np.array(q[:3]) - prev))
                t = max(float(d.get('beats_per_move', 2)) * beat, dist / vmax)   # 박자에 맞추되 속도 상한
                tm.movel(q, vmax, t=t)
                prev = np.array(q[:3])
    finally:
        tm.music_stop_pub.publish(Empty())
        tm.set_safety_mode('normal')


def validate_dance(wp, logger):
    d = wp.get('dance', {})
    lo, hi = np.array(d['safe_box']['min']), np.array(d['safe_box']['max'])
    bad = [q for q in d.get('waypoints', []) if np.any(np.array(q[:3]) < lo) or np.any(np.array(q[:3]) > hi)]
    if bad:
        raise ValueError(f'율동 웨이포인트가 safe_box(사람에게서 먼 공간) 밖: {bad}')
    logger.info(f"율동 웨이포인트 {len(d.get('waypoints', []))}개 safe_box 안 확인")


# ---------------------------------------------------------------------- 8. 걸레로 테이블 얼룩 닦기
def wipe_table(tm, args):
    tm.observe_table()
    try:
        res = tm.call(tm.spill_scan_cli, Trigger.Request(), timeout=10.0)
        ok, msg = bool(res and res.success), getattr(res, 'message', '')
    except TaskFailed as e:
        ok, msg = False, str(e)
    s = json.loads(msg) if ok else {}
    if tm.dr is None and not s.get('detected') and tm.wp.get('dry_spill'):
        s = tm.wp['dry_spill']                      # dry_run 테스트용 가짜 얼룩
    elif not ok:
        raise TaskFailed(f'얼룩 스캔 실패: {msg}')
    if not s.get('detected') or not s.get('position_valid'):
        tm.say('닦을 얼룩이 안 보여요')
        return
    kind = s.get('label', 'liquid')
    tm.say(f"{'액체' if kind == 'liquid' else '가루'} 얼룩 닦을게요")
    rag = tm.wp['wipe']['rag_pos']
    tm.pick('rag', np.array(rag[:3], float))
    tm.wipe_at(np.array(s['position'][:2]), kind, max(s.get('size_mm', [0, 0]) or [0]))
    tm.place('rag', rag[:2])


# ---------------------------------------------------------------------- 9. 장면 관찰 (decision_node 가 한가할 때 요청)
def scan_scene(tm, args):
    """관찰 자세에서 식탁 스냅샷: 그릇·컵·수저(YOLO) + depth → 베이스 3D, 채움 정도·빈 그릇 여부, 얼룩.
    /table/snapshot (자세한 3D) 과 /assist/scene (decision_node 용 요약) 에 발행. 물건은 건드리지 않음."""
    groups = tm.layout.get('classes', {})
    classes = dish_classes(tm) + groups.get('cups', ['cup']) + groups.get('utensils', ['spoon', 'fork', 'chopsticks'])
    tm.observe_table()
    dets = tm.detect_all(classes)
    spill = {'detected': False}
    try:
        res = tm.call(tm.spill_scan_cli, Trigger.Request(), timeout=10.0)
        if res is not None and res.success:
            spill = json.loads(res.message)
    except TaskFailed as e:
        tm.get_logger().warn(f'얼룩 스캔 생략: {e}')
    if tm.dr is None and tm.wp.get('dry_scene_spill') is not None:
        spill = tm.wp['dry_scene_spill']
    th = float(tm.g('empty_fill_threshold'))
    scene = {'stamp': time.time(),
             'objects': [{'cls': d['cls'], 'xy': [round(float(v), 1) for v in d['xyz'][:2]],
                          'fill': round(float(d['fill']), 2)} for d in dets],
             'spill': spill}
    snapshot = {'stamp': scene['stamp'], 'frame': 'base', 'unit': 'mm', 'observe_pose': tm.posx(),
                'objects': [{'cls': d['cls'], 'xyz': [round(float(v), 1) for v in d['xyz'][:3]],
                             'score': round(float(d.get('score', 0)), 2), 'fill': round(float(d['fill']), 2),
                             'empty': bool(0 <= d['fill'] < th)} for d in dets],
                'spill': spill}
    tm.snapshot = snapshot
    tm.scene_pub.publish(String(data=json.dumps(scene, ensure_ascii=False)))
    tm.snapshot_pub.publish(String(data=json.dumps(snapshot, ensure_ascii=False)))
    tm.get_logger().info(f"scene: {len(dets)}개, spill={spill.get('detected')}")


# ---------------------------------------------------------------------- 10. 판단을 보여주는 짧은 모션 (사람에게서 먼 높이, 저속)
def express(tm, args):
    """motion: look (대상 위 높은 곳에서 내려다보기) / nod (제자리 끄덕) / retreat (뒤로 물러나기)"""
    e = tm.wp.get('express', {})
    v = float(e.get('speed_mms', 60.0))
    m = args.get('motion', 'nod')
    if m == 'look':
        xy = args.get('point') or tm.user_front()
        z = float(e.get('look_z_mm', 420.0))
        tgt = np.clip([xy[0], xy[1], z], tm.wp['workspace']['min'], tm.wp['workspace']['max'])
        tilt = list(e.get('look_orientation', [0.0, 160.0, 0.0]))
        # 대상 바로 위가 아니라 로봇 쪽으로 물러난 높은 위치에서 내려다봄 (사람 손과 멀게)
        back = np.array(tgt[:2]) * (1.0 - float(e.get('look_back_ratio', 0.25)))
        tm.movel([back[0], back[1], tgt[2]] + tilt, v)
    elif m == 'nod':
        amp = float(e.get('nod_mm', 20.0))
        for _ in range(int(e.get('nod_count', 2))):
            tm.movel([0, 0, -amp, 0, 0, 0], v, mod=tm.dr.DR_MV_MOD_REL if tm.dr else 'rel')
            tm.movel([0, 0, amp, 0, 0, 0], v, mod=tm.dr.DR_MV_MOD_REL if tm.dr else 'rel')
    elif m == 'retreat':
        cur = tm.posx()
        d = np.array(cur[:2])
        d = d / (np.linalg.norm(d) + 1e-6) * float(e.get('retreat_mm', 80.0))
        tm.movel([cur[0] - d[0], cur[1] - d[1], cur[2]] + cur[3:], v)
    tm.express_done = True


def observe_table(tm, args):
    """관찰 자세로 가서 식탁 스냅샷을 찍고, 그 자세를 유지한 채 사람 손동작을 기다림 (RealSense 주 인식)."""
    scan_scene(tm, args)
    tm.say('보고 있어요') if args.get('announce') else None


def resume(tm, args):
    """v2.2 6.2: 정지 해제 뒤 담당자 확인으로 다시 시작. 멈춘 궤적을 이어 가지 않는다:
    들고 있으면 임시 자리에 내려놓고(6.4), 끝나면 run 루프가 go_rest() 로 관찰 자세에서 다시 본다."""
    if not tm.recover_holding():
        tm.say('다시 볼게요')


TASKS = {
    'observe_table': observe_table,
    'bring_dish': bring_dish, 'clear_dish': clear_dish, 'clear_empty': clear_empty, 'clear_all': clear_all,
    'pour_water': pour_water, 'seat_setting': seat_setting, 'music': music, 'wipe_table': wipe_table,
    'scan_scene': scan_scene, 'express': express, 'resume': resume,
}
# 끝나고 홈으로 돌아가지 않는 작업 (짧은 표현 모션은 제자리 유지)
NO_HOME = {'express', 'observe_table', 'scan_scene'}   # 끝나도 그 자리 유지
