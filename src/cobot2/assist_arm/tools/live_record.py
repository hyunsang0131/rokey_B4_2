"""실시간 MediaPipe 관절 보기 + 손동작 녹화 (로봇 무관 — 로봇을 움직이지 않음).

  웹캠 (ROS 불필요):
    ~/venvs/assist_mp/bin/python ~/cobot2_ws/src/cobot2/assist_arm/tools/live_record.py --person lee
  RealSense D435i (ROS realsense2_camera 토픽 사용, 설치 불필요):
    터미널1: realsense      (~/.bashrc 별칭 = ros2 launch realsense2_camera rs_launch.py camera_namespace:=/ align_depth.enable:=true)
    터미널2: source /opt/ros/jazzy/setup.bash
             ~/venvs/assist_mp/bin/python ~/cobot2_ws/src/cobot2/assist_arm/tools/live_record.py --source realsense --person lee
    → 같은 화면·키·저장 형식 + 깊이 기반 손 관절 3D(카메라 좌표 mm, hand_cam) 추가 저장, source=realsense 표시

화면: 양손 21관절(번호·뼈대, 왼손 파랑/오른손 초록) + 상체 관절, FPS, 손 인식 여부, 선택 동작, 동작별 개수
키 (영상 창을 클릭한 뒤)
  1~9   동작 선택: 1 idle  2 poke  3 clear  4 meal_done  5 cup_grab(빈손 C자=물 요청)  6 stop  7 point_seat  8 v_sign
        9 hold_cup(실제 컵을 쥐고 있음 = 명령 아님)
  space 1초 카운트다운 후 2초 녹화 → 30프레임(≈15fps)으로 저장 (모델 입력과 같은 시간 간격)
  a     자동 연속 녹화 켜기/끄기
  u     방금 녹화 지우기
  n     관절 번호 표시 켜기/끄기
  q/ESC 종료
DB: 저장할 때마다 ~/data/assist_arm/assist_arm.db 에 세션·샘플·영상 경로도 기록 (--no_db 로 끄기, 실패해도 파일은 저장)
저장
  관절: ~/data/assist_arm/gestures/<동작>/<person>_<시각>.npz   (학습 스크립트 그대로)
  영상: ~/data/assist_arm/clips/<동작>/<person>_<시각>.mp4      (원본, 관절 그림 없음)
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))          # assist_arm 패키지 (ROS 빌드 없이)
from assist_arm.common.datadb import DataDB  # noqa: E402
from assist_arm.common.features import FEATURE_VERSION, RECORD_HINTS, RECORD_LABELS  # noqa: E402
from assist_arm.common.mp_loader import HandTracker, PoseTracker  # noqa: E402

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
POSE_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
POSE_IDS = (0, 11, 12, 13, 14, 15, 16, 23, 24)
MODELS = os.path.expanduser('~/data/assist_arm/models')


class WebcamSource:
    name = 'webcam'

    def __init__(self, device):
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            sys.exit(f'웹캠 {device} 를 열 수 없어요 (v4l2-ctl --list-devices 로 번호 확인)')

    def read(self):
        ok, f = self.cap.read()
        return (f, None) if ok else (None, None)

    def to_cam(self, uv, depth):
        return None

    def close(self):
        self.cap.release()


class RealsenseSource:
    """ROS realsense2_camera 토픽 (color + aligned depth + camera_info). pyrealsense2 설치 없이."""
    name = 'realsense'

    def __init__(self, color, depth, info, timeout=10.0):
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError:
            sys.exit('ROS 가 안 잡혀요: 먼저 source /opt/ros/jazzy/setup.bash')
        import threading
        rclpy.init()
        self.rclpy, self.bridge = rclpy, CvBridge()
        self.node = rclpy.create_node('live_record_rs')
        self.color = self.depth = self.intr = None
        self.seq, self.last_seq = 0, -1
        self.lock = threading.Lock()

        def on_c(m):
            with self.lock:
                self.color = self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')
                self.seq += 1

        def on_d(m):
            with self.lock:
                self.depth = self.bridge.imgmsg_to_cv2(m, desired_encoding='passthrough')

        def on_i(m):
            self.intr = (m.k[0], m.k[4], m.k[2], m.k[5])
        self.node.create_subscription(Image, color, on_c, 5)
        self.node.create_subscription(Image, depth, on_d, 5)
        self.node.create_subscription(CameraInfo, info, on_i, 5)
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        threading.Thread(target=self.ex.spin, daemon=True).start()
        t0 = time.time()
        while self.color is None and time.time() - t0 < timeout:
            time.sleep(0.1)
        if self.color is None:
            sys.exit(f'RealSense 영상이 안 들어와요 ({color}). realsense2_camera 를 켰는지, '
                     f'ros2 topic list 로 토픽 이름을 확인하세요 (--rs_color 로 변경 가능)')

    def read(self):
        while True:
            with self.lock:
                if self.seq != self.last_seq and self.color is not None:
                    self.last_seq = self.seq
                    return self.color.copy(), None if self.depth is None else self.depth.copy()
            time.sleep(0.003)

    def to_cam(self, uv, depth, win=4):
        """픽셀들 (N,2) → 카메라 좌표 (N,3) mm (object_detection 과 같은 식). 깊이 없으면 NaN."""
        out = np.full((len(uv), 3), np.nan, np.float32)
        if depth is None or self.intr is None:
            return out
        fx, fy, cx, cy = self.intr
        h, w = depth.shape[:2]
        for i, (u, v) in enumerate(uv):
            x, y = int(u), int(v)
            if not (0 <= x < w and 0 <= y < h):
                continue
            patch = depth[max(0, y - win):y + win + 1, max(0, x - win):x + win + 1]
            val = patch[patch > 0]
            if val.size:
                z = float(np.median(val))
                out[i] = ((u - cx) * z / fx, (v - cy) * z / fy, z)
        return out

    def close(self):
        self.ex.shutdown()
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def draw(img, hands, pose, show_ids):
    h, w = img.shape[:2]
    if pose is not None:
        P = (pose[:, :2] * [w, h]).astype(int)
        for a, b in POSE_EDGES:
            if pose[a, 3] > 0.5 and pose[b, 3] > 0.5:
                cv2.line(img, tuple(P[a]), tuple(P[b]), (255, 160, 0), 2)
        for i in POSE_IDS:
            if pose[i, 3] > 0.5:
                cv2.circle(img, tuple(P[i]), 5, (255, 160, 0), -1)
    for hd in hands:
        col = (60, 220, 60) if hd['handedness'] == 'Right' else (255, 120, 40)
        Q = (hd['pts'][:, :2] * [w, h]).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(Q[a]), tuple(Q[b]), col, 2)
        for i, q in enumerate(Q):
            cv2.circle(img, tuple(q), 4 if i in (4, 8, 12, 16, 20) else 3, (0, 255, 255), -1)
            if show_ids:
                cv2.putText(img, str(i), tuple(q + [4, -4]), 0, 0.35, (255, 255, 255), 1)
        cv2.putText(img, f"{hd['handedness']} {hd['score']:.2f}", tuple(Q[0] + [8, 18]), 0, 0.55, col, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--person', default='unknown')
    ap.add_argument('--source', choices=['webcam', 'realsense'], default='webcam')
    ap.add_argument('--device', type=int, default=0, help='웹캠 번호')
    ap.add_argument('--rs_color', default='/camera/color/image_raw')
    ap.add_argument('--rs_depth', default='/camera/aligned_depth_to_color/image_raw')
    ap.add_argument('--rs_info', default='/camera/color/camera_info')
    ap.add_argument('--seq_len', type=int, default=30, help='저장 프레임 수 (학습·추론 모델 입력 길이)')
    ap.add_argument('--seq_sec', type=float, default=2.0, help='녹화 시간. seq_len 프레임으로 균등 샘플링 (≈15fps, 로봇 쪽 pose_node 속도와 맞춤)')
    ap.add_argument('--num_hands', type=int, default=2)
    ap.add_argument('--det', type=float, default=0.5, help='min_hand_detection_confidence')
    ap.add_argument('--presence', type=float, default=0.5)
    ap.add_argument('--track', type=float, default=0.5)
    ap.add_argument('--no_pose', action='store_true')
    ap.add_argument('--gestures', default='~/data/assist_arm/gestures')
    ap.add_argument('--clips', default='~/data/assist_arm/clips')
    ap.add_argument('--no_db', action='store_true', help='DB(~/data/assist_arm/assist_arm.db) 기록 끄기 (파일 저장은 그대로)')
    ap.add_argument('--preview', default='', help='창 없이 N초 돌려 마지막 화면을 이 파일로 저장 (테스트용)')
    ap.add_argument('--preview_sec', type=float, default=5.0)
    a = ap.parse_args()
    gdir, cdir = os.path.expanduser(a.gestures), os.path.expanduser(a.clips)
    labels = list(RECORD_LABELS)
    for lab in labels:
        os.makedirs(os.path.join(gdir, lab), exist_ok=True)
        os.makedirs(os.path.join(cdir, lab), exist_ok=True)

    hands_t = HandTracker(max_hands=a.num_hands, min_det=a.det, min_presence=a.presence, min_track=a.track,
                          task_model_path=f'{MODELS}/hand_landmarker.task')
    pose_t = None if a.no_pose else PoseTracker(task_model_path=f'{MODELS}/pose_landmarker_lite.task')
    cap = WebcamSource(a.device) if a.source == 'webcam' else RealsenseSource(a.rs_color, a.rs_depth, a.rs_info)

    db = None if a.no_db else DataDB()
    session_id = None            # 첫 저장 때 세션 생성
    cur, state, t_state, auto, show_ids = 0, 'ready', 0.0, False, True
    buf, raw, last = [], [], None
    fps, t_prev = 0.0, time.monotonic()
    t_start = time.monotonic()

    def counts():
        return {lab: len([f for f in os.listdir(os.path.join(gdir, lab)) if f.endswith('.npz')]) for lab in labels}
    cnt = counts()

    while True:
        frame, depth = cap.read()
        if frame is None:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = hands_t.process_all(rgb)
        pose = pose_t.process(rgb) if pose_t else None
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-3, now - t_prev))
        t_prev = now

        # ---- 녹화 상태
        if state == 'countdown' and now - t_state > 1.0:
            state, buf, raw = 'record', [], []
        if state == 'rest' and now - t_state > 0.7:
            state, t_state = 'countdown', now
        if state == 'record':
            h0 = hands[0] if hands else None
            nan21 = np.full((21, 3), np.nan, np.float32)
            cam = nan21
            if h0 is not None and a.source == 'realsense':
                cam = cap.to_cam(h0['pts'][:, :2] * [frame.shape[1], frame.shape[0]], depth)
            buf.append((h0['pts'] if h0 else nan21, h0 is not None, h0['handedness'] if h0 else '',
                        h0['world'] if h0 is not None and h0['world'] is not None else nan21,
                        pose if pose is not None else np.full((33, 4), np.nan, np.float32), pose is not None,
                        cam, now))
            raw.append(frame.copy())
            if buf and now - buf[0][-1] >= a.seq_sec:
                lab = labels[cur]
                idx = np.round(np.linspace(0, len(buf) - 1, a.seq_len)).astype(int)   # 웹캠 30fps → 2초 30프레임(15fps)
                hand, hv, hd, world, ps, pv, cams, ts = zip(*[buf[i] for i in idx])
                if sum(hv) < a.seq_len * 0.6 and lab != 'idle':
                    print(f'손이 {sum(hv)}/{a.seq_len} 프레임만 보여서 버림 — 다시 녹화하세요')
                else:
                    stem = f'{a.person}_{time.strftime("%Y%m%d_%H%M%S")}_{int(time.time() * 1000) % 1000:03d}'
                    rate = (len(ts) - 1) / max(1e-3, ts[-1] - ts[0])
                    np.savez_compressed(os.path.join(gdir, lab, stem + '.npz'), hand=np.stack(hand),
                                        hand_valid=np.array(hv), handedness=np.array(hd), hand_world=np.stack(world),
                                        pose=np.stack(ps), pose_valid=np.array(pv), fps=rate, label=lab, person=a.person,
                                        source=a.source,
                                        hand_cam=np.stack(cams))   # realsense: 손 21점 카메라 3D (mm), 웹캠: NaN
                    vw = cv2.VideoWriter(os.path.join(cdir, lab, stem + '.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                         max(5.0, rate), (raw[0].shape[1], raw[0].shape[0]))
                    for f in raw:
                        vw.write(f)
                    vw.release()
                    last = (os.path.join(gdir, lab, stem + '.npz'), os.path.join(cdir, lab, stem + '.mp4'))
                    if db is not None and db.ok:          # DB 실패해도 파일 저장은 이미 끝났음
                        if session_id is None:
                            session_id = db.start_session(a.person, a.source,
                                                          device=a.device if a.source == 'webcam' else a.rs_color,
                                                          notes='live_record')
                        hds = [x for x in hd if x]
                        db.add_gesture_sample(session_id, lab, last[0], n_frames=len(hand), fps=rate,
                                              handedness=max(set(hds), key=hds.count) if hds else '',
                                              preprocess_ver=FEATURE_VERSION)
                        db.add_clip(session_id, last[1], label=lab, duration_s=ts[-1] - ts[0])
                    cnt = counts()
                    print(f'저장 {lab}: {stem} (손 {sum(hv)}/{a.seq_len}, {rate:.1f}fps)')
                buf, raw = [], []
                state, t_state = ('rest', now) if auto else ('ready', 0.0)

        # ---- 화면
        vis = frame.copy()
        draw(vis, hands, pose, show_ids)
        H, W = vis.shape[:2]
        txt = f'[{a.source}] FPS {fps:4.1f}  hands {len(hands)}' + ('' if hands else '  (NO HAND)')
        if a.source == 'realsense' and hands and depth is not None:
            tip = cap.to_cam(hands[0]['pts'][8:9, :2] * [W, H], depth)[0]
            if np.isfinite(tip).all():
                txt2 = f'tip(cam) {tip[0]:.0f},{tip[1]:.0f},{tip[2]:.0f}mm'
                cv2.putText(vis, txt2, (W - 300, 50), 0, 0.55, (0, 255, 255), 2)
        cv2.putText(vis, txt, (W - 330, 25), 0, 0.6, (0, 255, 0) if hands else (0, 0, 255), 2)
        y = 22
        for i, lab in enumerate(labels):
            sel = i == cur
            cv2.putText(vis, f'{i + 1} {lab}: {cnt[lab]}', (8, y), 0, 0.55, (0, 255, 0) if sel else (230, 230, 230),
                        2 if sel else 1)
            y += 22
        msg = {'ready': 'space=record  a=auto  u=undo  q=quit', 'countdown': 'ready...',
               'record': 'RECORDING', 'rest': 'next...'}[state] + ('  [auto]' if auto else '')
        cv2.putText(vis, RECORD_HINTS.get(labels[cur], ''), (8, H - 66), 0, 0.6, (0, 255, 255), 2)
        cv2.putText(vis, f'[{labels[cur]}] {msg}', (8, H - 40), 0, 0.65,
                    (0, 0, 255) if state == 'record' else (255, 255, 255), 2)
        if state == 'record':
            cv2.rectangle(vis, (8, H - 25), (8 + int((W - 16) * min(1.0, (now - buf[0][-1]) / a.seq_sec if buf else 0)), H - 10), (0, 0, 255), -1)

        if a.preview:
            if now - t_start > a.preview_sec:
                cv2.imwrite(os.path.expanduser(a.preview), vis)
                print(f'preview 저장: {a.preview}  FPS {fps:.1f}  hands {len(hands)}  pose {pose is not None}')
                break
            continue
        cv2.imshow('live_record (MediaPipe)', vis)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('q'), 27):
            break
        if ord('1') <= k < ord('1') + len(labels) and state in ('ready', 'rest'):
            cur = k - ord('1')
        elif k == ord(' ') and state == 'ready':
            state, t_state = 'countdown', now
        elif k == ord('a'):
            auto = not auto
            if auto and state == 'ready':
                state, t_state = 'countdown', now
            elif not auto and state == 'rest':
                state = 'ready'
        elif k == ord('u') and last:
            for f in last:
                if os.path.exists(f):
                    os.remove(f)
            if db is not None:
                db.remove_files(last[0], last[1])
            print('방금 녹화 삭제')
            last, cnt = None, counts()
        elif k == ord('n'):
            show_ids = not show_ids
    cap.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
