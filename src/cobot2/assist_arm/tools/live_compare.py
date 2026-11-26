"""웹캠 실시간 손 인식 비교 (한 창, 같은 프레임): MediaPipe | RTMPose(OpenVINO) | Hybrid

  TORCH_HOME=~/data/assist_arm/models/rtm python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/live_compare.py
  RealSense (위에서 내려다보기): source /opt/ros/jazzy/setup.bash 후 같은 명령 + --source realsense
  (옵션: --device 0  --rtm hand|rtmw  --ov cpu|gpu  --hand Right|Left  --test)
설치 없이: MediaPipe 는 ~/venvs/assist_mp, RTMPose 는 ~/venvs/assist_rtm 에서 자동으로 불러옴 (시스템 파이썬으로 실행).

패널
  MediaPipe : MediaPipe Hands (검출 + 추적 + '손 있음' 점수)
  RTMPose   : RTMDet-nano 손 검출기 → RTMPose-m 관절 (단독). f 키로 오검출 억제 on/off
  Hybrid    : MediaPipe 가 찾은 손 위치(21점 박스 + 25% 여유) → RTMPose 관절만. MediaPipe 가 손 없다면 없음
각 패널: 21점 뼈대, 지연 ms·FPS, 손 검출률, 흔들림(가만히 둘 때 손 크기 대비 %), 현재 제스처 (같은 규칙 분류기)
키
  t 정확도 측정 (엄지척 → 손바닥 → 검지 콕 → 손등 밀기 → 빈손 컵 시늉 → 실제 컵 쥐기 → 손 없음,
    준비 2초 + 측정 5초씩) → 표 + CSV. '실제 컵 쥐기' 는 규칙으로 판정 못 하므로 검출률·흔들림만 (위에서 본 컵 쥔 손 비교용)
  f RTMPose 오검출 억제 on/off (검출 점수 0.5, 최소 박스 40px, 관절 평균 신뢰도, 이전 손 근처만)
  e One Euro 관절 스무딩 on/off (세 패널 모두)
  r 통계 초기화, s 스크린샷, n 관절 번호, q/ESC 종료
결과: 표 출력 + ~/data/assist_arm/bench/live_compare_<시간>.csv (구간별 검출률·제스처 정답률·흔들림·오검출률·지연)
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
from assist_arm.common.features import rule_gesture  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
DETECT_ONLY = 'detect_only'        # 규칙 분류기로 판정할 수 없는 구간 → 검출률·흔들림만
TEST_STEPS = [('엄지척', 'thumbs up', 'thumbs_up'), ('손바닥 펴기', 'open palm to camera', 'stop'),
              ('검지 콕', 'index point', 'point'), ('손등 밀기', 'back of hand', 'push'),
              ('빈손 컵 시늉', 'EMPTY hand C-shape (ask water)', 'cup_grasp'),
              ('실제 컵 쥐기', 'hold the REAL cup (from above)', DETECT_ONLY),
              ('손 없음', 'NO hand in view', None)]
PREP_SEC, MEAS_SEC = 2.0, 5.0


def add_venv(path):
    for sp in sorted(glob.glob(os.path.expanduser(f'{path}/lib/python3*/site-packages'))):
        if sp not in sys.path:
            sys.path.append(sp)


# ------------------------------------------------------------------ One Euro 필터 (관절 스무딩)
class OneEuro:
    def __init__(self, min_cutoff=1.5, beta=0.02, d_cutoff=1.0):
        self.mc, self.beta, self.dc = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(1e-3, t - self.t)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.dc, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.mc + self.beta * np.abs(self.dx)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x.copy()

    def reset(self):
        self.x = None


# ------------------------------------------------------------------ 모델
class Models:
    def __init__(self, a):
        from assist_arm.common.mp_loader import HandTracker
        self.mp = HandTracker(max_hands=a.num_hands, task_model_path='~/data/assist_arm/models/hand_landmarker.task')
        add_venv('~/venvs/assist_rtm')
        os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/data/assist_arm/models/rtm'))
        from rtmlib import Hand, Wholebody
        self.kind, self.score_th = a.rtm, a.score_th
        self.rtm_filter = True
        self.prev_rtm = None
        if a.rtm == 'rtmw':
            self.whole = Wholebody(mode='lightweight', backend='openvino', device=a.ov)
            self.hand = Hand(backend='openvino', device=a.ov)        # hybrid 용 손 관절 모델
        else:
            self.whole = None
            self.hand = Hand(backend='openvino', device=a.ov)
        self.det_thr_raw = self.hand.det_model.score_thr

    def run_mp(self, bgr):
        h, w = bgr.shape[:2]
        return [(x['pts'][:, :2] * [w, h], float(x['score'])) for x in self.mp.process_all(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))]

    def run_rtm(self, bgr):
        if self.whole is not None:
            kp, sc = self.whole(bgr)
            hands = []
            for k, s in zip(kp, sc):
                for a, b in ((91, 112), (112, 133)):
                    hands.append((np.asarray(k[a:b], float), float(np.mean(s[a:b]))))
        else:
            self.hand.det_model.score_thr = 0.5 if self.rtm_filter else self.det_thr_raw
            kp, sc = self.hand(bgr)
            hands = [(np.asarray(k, float), float(np.mean(s))) for k, s in zip(kp, sc)]
        if not self.rtm_filter:
            return [h for h in hands if h[1] > 0.1]
        out = []
        for k, s in hands:
            if s <= self.score_th or np.ptp(k, axis=0).max() < 40:
                continue
            if self.prev_rtm is not None:     # 간단 추적: 이전 손 근처(손 크기 2배 안)가 아니면 2프레임 연속일 때만
                size = np.linalg.norm(self.prev_rtm[9] - self.prev_rtm[0]) + 1e-6
                if np.linalg.norm(k[0] - self.prev_rtm[0]) > 2 * size and s < self.score_th + 0.2:
                    continue
            out.append((k, s))
        self.prev_rtm = max(out, key=lambda x: x[1])[0] if out else None
        return out

    def run_hybrid(self, bgr, mp_hands):
        """MediaPipe 손 위치(21점 박스 + 25% 여유) → RTMPose 관절"""
        if not mp_hands:
            return []
        h, w = bgr.shape[:2]
        boxes = []
        for k, _ in mp_hands:
            x1, y1 = k.min(axis=0)
            x2, y2 = k.max(axis=0)
            m = 0.25 * max(x2 - x1, y2 - y1)
            boxes.append([max(0, x1 - m), max(0, y1 - m), min(w - 1, x2 + m), min(h - 1, y2 + m)])
        kp, sc = self.hand.pose_model(bgr, bboxes=boxes)
        return [(np.asarray(k, float), float(np.mean(s))) for k, s in zip(kp, sc)]


# ------------------------------------------------------------------ 통계
class Stats:
    def __init__(self):
        self.reset()

    def reset(self):
        self.lat, self.det, self.jit = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
        self.prev = None
        self.last_jit = None
        self.all_lat, self.all_det, self.all_jit = [], [], []

    def update(self, ms, hands):
        self.lat.append(ms)
        self.all_lat.append(ms)
        found = len(hands) > 0
        self.det.append(found)
        self.all_det.append(found)
        self.last_jit = None
        if found:
            k = max(hands, key=lambda x: x[1])[0]
            size = np.linalg.norm(k[9] - k[0]) + 1e-6
            if self.prev is not None and np.linalg.norm(k[0] - self.prev[0]) < 0.5 * size:
                self.last_jit = float(np.median(np.linalg.norm(k - self.prev, axis=1)) / size)
                self.jit.append(self.last_jit)
                self.all_jit.append(self.last_jit)
            self.prev = k
        else:
            self.prev = None

    def text(self):
        lat = np.mean(self.lat) if self.lat else 0
        det = np.mean(self.det) if self.det else 0
        jit = f'{np.median(self.jit) * 100:.1f}%' if self.jit else '-'
        return f'{lat:5.1f}ms {1000 / max(lat, 1e-3):4.1f}fps hand {det:4.0%} jit {jit}'

    def summary(self):
        return {'lat_ms': float(np.mean(self.all_lat)) if self.all_lat else None,
                'lat_p95_ms': float(np.percentile(self.all_lat, 95)) if self.all_lat else None,
                'detect_rate': float(np.mean(self.all_det)) if self.all_det else None,
                'jitter_pct': float(np.median(self.all_jit) * 100) if self.all_jit else None}


def classify(hands, handed):
    if not hands:
        return None
    k = max(hands, key=lambda x: x[1])[0]
    return rule_gesture(np.c_[k, np.zeros(len(k))], handed)[0]


def draw(img, hands, color, show_ids):
    for k, s in hands:
        q = k.astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(q[a]), tuple(q[b]), color, 2)
        for i, p in enumerate(q):
            cv2.circle(img, tuple(p), 3, (0, 255, 255), -1)
            if show_ids:
                cv2.putText(img, str(i), tuple(p + [4, -4]), 0, 0.33, (255, 255, 255), 1)
        cv2.putText(img, f'{s:.2f}', tuple(q[0] + [6, 16]), 0, 0.5, color, 2)


# ------------------------------------------------------------------ 정확도 측정
class TestRun:
    def __init__(self, names):
        self.i, self.t0, self.done = 0, time.monotonic(), False
        self.rows = {n: [] for n in names}

    def phase(self):
        el = time.monotonic() - self.t0
        if el > PREP_SEC + MEAS_SEC:
            self.i, self.t0 = self.i + 1, time.monotonic()
            if self.i >= len(TEST_STEPS):
                self.done = True
                return 'done', 0
            return self.phase()
        return ('prep', PREP_SEC - el) if el < PREP_SEC else ('measure', PREP_SEC + MEAS_SEC - el)

    def record(self, name, found, label, jitter, ms):
        self.rows[name].append((self.i, found, label, jitter, ms))

    def report(self, out_dir, settings):
        hdr = ['구간', '정답', '모델', '검출률', '제스처정답률', '흔들림%', '오검출률(손없음)', '지연ms']
        lines = []
        print(f'\n=== 정확도 측정 결과 ({settings}) ===')
        print(f"{'구간':<10}{'모델':<12}{'검출률':>8}{'정답률':>8}{'흔들림':>8}{'오검출':>8}{'지연ms':>8}")
        tot = {}
        for i, (ko, _, exp) in enumerate(TEST_STEPS):
            for name, r in self.rows.items():
                r = [x for x in r if x[0] == i]
                if not r:
                    continue
                det = float(np.mean([x[1] for x in r]))
                acc = float(np.mean([x[2] == exp for x in r])) if exp and exp != DETECT_ONLY else None
                jit = [x[3] for x in r if x[3] is not None]
                jit = float(np.median(jit) * 100) if jit else None
                fp = det if exp is None else None
                lat = float(np.mean([x[4] for x in r]))
                tot.setdefault(name, []).append((det if exp else None, acc, fp, lat, jit))
                print(f"{ko:<10}{name:<12}{'' if exp is None else f'{det:.0%}':>8}{'' if acc is None else f'{acc:.0%}':>8}"
                      f"{'-' if jit is None else f'{jit:.1f}%':>8}{'' if fp is None else f'{fp:.0%}':>8}{lat:>8.1f}")
                lines.append([ko, exp or 'none', name, f'{det:.3f}', '' if acc is None else f'{acc:.3f}',
                              '' if jit is None else f'{jit:.2f}', '' if fp is None else f'{fp:.3f}', f'{lat:.1f}'])
        print('-' * 62)
        for name, t in tot.items():
            m = lambda i: [x[i] for x in t if x[i] is not None]  # noqa: E731
            avg = lambda v: float(np.mean(v)) if v else 0.0  # noqa: E731
            print(f"{'평균':<10}{name:<12}{avg(m(0)):>8.0%}{avg(m(1)):>8.0%}{avg(m(4)):>7.1f}%{avg(m(2)):>8.0%}{avg(m(3)):>8.1f}")
            lines.append(['평균', '', name, f'{avg(m(0)):.3f}', f'{avg(m(1)):.3f}', f'{avg(m(4)):.2f}',
                          f'{avg(m(2)):.3f}', f'{avg(m(3)):.1f}'])
        p = os.path.join(out_dir, f'live_compare_{time.strftime("%Y%m%d_%H%M%S")}.csv')
        with open(p, 'w', encoding='utf-8-sig') as f:
            f.write(f'# {settings}\n' + ','.join(hdr) + '\n' + '\n'.join(','.join(x) for x in lines) + '\n')
        print('CSV 저장:', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--video', default='', help='웹캠 대신 녹화 영상 파일로 (반복 재생)')
    ap.add_argument('--source', choices=['webcam', 'realsense'], default='webcam',
                    help='realsense: ROS realsense2_camera 컬러 토픽 (먼저 source /opt/ros/jazzy/setup.bash)')
    ap.add_argument('--rs_color', default='/camera/color/image_raw')
    ap.add_argument('--rs_depth', default='/camera/aligned_depth_to_color/image_raw')
    ap.add_argument('--rs_info', default='/camera/color/camera_info')
    ap.add_argument('--rtm', choices=['hand', 'rtmw'], default='hand')
    ap.add_argument('--ov', default='cpu')
    ap.add_argument('--score_th', type=float, default=0.3)
    ap.add_argument('--num_hands', type=int, default=2)
    ap.add_argument('--hand', choices=['Right', 'Left'], default='Right')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--preview', default='')
    ap.add_argument('--preview_sec', type=float, default=6.0)
    a = ap.parse_args()
    print('모델 불러오는 중...')
    M = Models(a)
    names = ['MediaPipe', 'RTMPose', 'Hybrid']
    colors = {'MediaPipe': (60, 220, 60), 'RTMPose': (255, 120, 40), 'Hybrid': (200, 80, 255)}
    stats = {n: Stats() for n in names}
    filt = {n: [OneEuro() for _ in range(2)] for n in names}       # 손 2개까지
    smooth, show_ids = False, False
    rs = None
    if a.source == 'realsense':
        sys.path.insert(0, HERE)
        from live_record import RealsenseSource
        rs = RealsenseSource(a.rs_color, a.rs_depth, a.rs_info)
        cap = None
    else:
        cap = cv2.VideoCapture(os.path.expanduser(a.video) if a.video else a.device)
        if not a.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            sys.exit(f'카메라/영상을 열 수 없어요 ({a.video or a.device}) — 다른 프로그램이 웹캠을 쓰고 있으면 먼저 끄세요')
    out_dir = os.path.expanduser('~/data/assist_arm/bench')
    os.makedirs(out_dir, exist_ok=True)
    test = TestRun(names) if a.test else None
    t_start = time.monotonic()

    while True:
        if rs is not None:
            frame, _ = rs.read()
            ok = frame is not None
        else:
            ok, frame = cap.read()
        if not ok:
            if a.video:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        now = time.monotonic()
        res, ms = {}, {}
        t0 = time.perf_counter()
        res['MediaPipe'] = M.run_mp(frame)
        t1 = time.perf_counter()
        res['RTMPose'] = M.run_rtm(frame)
        t2 = time.perf_counter()
        res['Hybrid'] = M.run_hybrid(frame, res['MediaPipe'])
        t3 = time.perf_counter()
        ms = {'MediaPipe': (t1 - t0) * 1000, 'RTMPose': (t2 - t1) * 1000, 'Hybrid': ((t1 - t0) + (t3 - t2)) * 1000}
        if smooth:
            for n in names:
                hs = sorted(res[n], key=lambda x: -x[1])[:2]
                res[n] = [(filt[n][i](k, now), s) for i, (k, s) in enumerate(hs)]
                for i in range(len(hs), 2):
                    filt[n][i].reset()
        labels = {}
        for n in names:
            stats[n].update(ms[n], res[n])
            labels[n] = classify(res[n], a.hand)
        if test is not None and not test.done:
            ph, _ = test.phase()
            if ph == 'measure':
                for n in names:
                    test.record(n, bool(res[n]), labels[n], stats[n].last_jit, ms[n])
            if test.done:
                test.report(out_dir, f"source={a.source} rtm={a.rtm} ov={a.ov} filter={'on' if M.rtm_filter else 'off'} "
                                     f"smooth={'on' if smooth else 'off'} hand={a.hand}")

        panels = []
        for n in names:
            img = frame.copy()
            draw(img, res[n], colors[n], show_ids)
            title = n + {'RTMPose': f" ({'filter on' if M.rtm_filter else 'raw'})", 'Hybrid': ' (MP box+RTM)'}.get(n, '')
            cv2.rectangle(img, (0, 0), (img.shape[1], 46), (0, 0, 0), -1)
            cv2.putText(img, title, (6, 17), 0, 0.55, (255, 255, 255), 1)
            cv2.putText(img, stats[n].text(), (6, 38), 0, 0.48, (0, 255, 0), 1)
            cv2.putText(img, f'gesture: {labels[n] or "(no hand)"}', (6, img.shape[0] - 12), 0, 0.65,
                        (0, 255, 255) if labels[n] else (0, 0, 255), 2)
            panels.append(img)
        vis = np.hstack(panels)
        scale = 1600 / vis.shape[1] if vis.shape[1] > 1600 else 1.0
        if scale != 1.0:
            vis = cv2.resize(vis, None, fx=scale, fy=scale)
        H, W = vis.shape[:2]
        if test is not None and not test.done:
            ph, left = test.phase()
            _, en, _ = TEST_STEPS[min(test.i, len(TEST_STEPS) - 1)]
            col = (0, 200, 255) if ph == 'prep' else (0, 0, 255)
            cv2.rectangle(vis, (0, 48), (W, 84), (0, 0, 0), -1)
            cv2.putText(vis, f'[{test.i + 1}/{len(TEST_STEPS)}] {en} - ' + ('get ready' if ph == 'prep' else 'MEASURING')
                        + f' {left:.1f}s', (10, 76), 0, 0.85, col, 2)
        cv2.rectangle(vis, (0, H - 22), (W, H), (0, 0, 0), -1)
        cv2.putText(vis, f"t=test  f=RTM filter({'on' if M.rtm_filter else 'off'})  e=smooth({'on' if smooth else 'off'})"
                    "  r=reset  s=save  n=ids  q=quit", (8, H - 6), 0, 0.5, (255, 255, 255), 1)

        if a.preview:
            if (test is not None and test.done) or (test is None and now - t_start > a.preview_sec):
                cv2.imwrite(os.path.expanduser(a.preview), vis)
                break
            continue
        cv2.imshow('live_compare: MediaPipe | RTMPose | Hybrid', vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if k == ord('t'):
            test = TestRun(names)
            print('정확도 측정 시작: 화면 안내를 따라 주세요')
        elif k == ord('f'):
            M.rtm_filter = not M.rtm_filter
        elif k == ord('e'):
            smooth = not smooth
        elif k == ord('r'):
            for s in stats.values():
                s.reset()
        elif k == ord('s'):
            p = os.path.join(out_dir, f'live_compare_{time.strftime("%H%M%S")}.png')
            cv2.imwrite(p, vis)
            print('스크린샷:', p)
        elif k == ord('n'):
            show_ids = not show_ids
    (rs.close() if rs is not None else cap.release())
    cv2.destroyAllWindows()
    summ = {n: stats[n].summary() for n in names}
    p = os.path.join(out_dir, f'live_compare_{time.strftime("%Y%m%d_%H%M%S")}.json')
    json.dump(summ, open(p, 'w'), indent=1)
    print('\n요약 (전체 프레임)')
    for n, v in summ.items():
        j = '-' if v['jitter_pct'] is None else f"{v['jitter_pct']:.1f}%"
        print(f"  {n:10} 지연 {v['lat_ms'] or 0:6.1f}ms  p95 {v['lat_p95_ms'] or 0:6.1f}ms  "
              f"손 검출 {v['detect_rate'] or 0:4.0%}  흔들림 {j}")
    print('저장:', p)


if __name__ == '__main__':
    main()
