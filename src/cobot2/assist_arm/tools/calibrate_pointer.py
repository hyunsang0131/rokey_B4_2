"""웹캠 → 테이블 평면 호모그래피 캘리브레이션 (finger_pointer 용). 로봇을 움직이는 명령은 보내지 않는다.

방법 A: ArUco 4개 (DICT_4X4_50, id 0~3)를 테이블 네 곳에 붙임
  source /opt/ros/jazzy/setup.bash && source ~/cobot2_ws/install/setup.bash
  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/calibrate_pointer.py --ros-args -p method:=aruco
방법 B: 4점 클릭 (테이블 위 표시 4곳을 웹캠 창에서 차례로 클릭)
  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/calibrate_pointer.py --ros-args -p method:=click
각 점의 로봇 좌표 입력 (점마다 터미널에서 선택)
  - 숫자 직접 입력: "x y" (mm, 로봇 베이스)
  - 엔터만: 사람이 로봇을 직접교시(핸드 가이딩)로 그 점 위에 TCP 를 갖다 둔 상태에서
            get_current_posx 서비스로 현재 위치를 *읽기만* 함
결과: calibration_file (yaml) — H(3x3), image_points, table_points, table_z_mm, 재투영 오차
웹캠은 webcam_publisher 의 /webcam/image_raw 를 사용 (webcam_publisher 를 먼저 실행).
"""
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # assist_arm 패키지
from assist_arm.common.dsr_services import DsrServices  # noqa: E402


class Calib(Node):
    def __init__(self):
        super().__init__('calibrate_pointer')
        self.declare_parameter('method', 'click')
        self.declare_parameter('calibration_file', '~/data/assist_arm/pointer_calibration.yaml')
        self.declare_parameter('table_z_mm', 0.0)
        self.declare_parameter('robot_id', 'dsr01')
        self.declare_parameter('image_topic', '/webcam/image_raw')
        self.bridge = CvBridge()
        self.frame = None
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_img, 5)
        self.dsr = DsrServices(self, robot_id=self.get_parameter('robot_id').value)

    def on_img(self, m):
        self.frame = self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')

    def read_robot_xy(self):
        box = {}
        self.dsr.request_posx(lambda p: box.setdefault('p', p))
        t_end = time.monotonic() + 3.0
        while 'p' not in box and time.monotonic() < t_end:
            time.sleep(0.05)
        p = box.get('p')
        return None if p is None else (float(p[0]), float(p[1]))


def find_aruco(img):
    ar = cv2.aruco
    d = ar.getPredefinedDictionary(ar.DICT_4X4_50)
    if hasattr(ar, 'ArucoDetector'):
        corners, ids, _ = ar.ArucoDetector(d, ar.DetectorParameters()).detectMarkers(img)
    else:   # OpenCV 4.6 (시스템) API
        corners, ids, _ = ar.detectMarkers(img, d, parameters=ar.DetectorParameters_create())
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if 0 <= i <= 3:
                found[int(i)] = c.reshape(4, 2).mean(axis=0)
    return found


def main(args=None):
    rclpy.init(args=args)
    node = Calib()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    method = node.get_parameter('method').value
    print('웹캠 영상 기다리는 중...')
    while node.frame is None:
        time.sleep(0.1)
    img_pts = []
    if method == 'aruco':
        print('ArUco id 0~3 이 모두 보이면 자동으로 잡습니다 (q: 취소)')
        while True:
            f = node.frame.copy()
            found = find_aruco(f)
            for i, c in found.items():
                cv2.circle(f, tuple(int(v) for v in c), 6, (0, 255, 0), -1)
                cv2.putText(f, str(i), tuple(int(v) + 8 for v in c), 0, 0.8, (0, 255, 0), 2)
            cv2.imshow('calibrate_pointer', f)
            k = cv2.waitKey(30) & 0xFF
            if len(found) == 4:
                img_pts = [found[i].tolist() for i in range(4)]
                cv2.waitKey(800)
                break
            if k == ord('q'):
                return
    else:
        print('웹캠 창에서 테이블 위 표시 4곳을 차례로 클릭하세요 (r: 다시, q: 취소)')

        def on_click(ev, x, y, *_):
            if ev == cv2.EVENT_LBUTTONDOWN and len(img_pts) < 4:
                img_pts.append([float(x), float(y)])
        cv2.namedWindow('calibrate_pointer')
        cv2.setMouseCallback('calibrate_pointer', on_click)
        while len(img_pts) < 4:
            f = node.frame.copy()
            for i, (x, y) in enumerate(img_pts):
                cv2.circle(f, (int(x), int(y)), 6, (0, 0, 255), -1)
                cv2.putText(f, str(i), (int(x) + 8, int(y)), 0, 0.8, (0, 0, 255), 2)
            cv2.imshow('calibrate_pointer', f)
            k = cv2.waitKey(30) & 0xFF
            if k == ord('r'):
                img_pts.clear()
            if k == ord('q'):
                return
    cv2.destroyAllWindows()

    tbl_pts = []
    for i, uv in enumerate(img_pts):
        while True:
            s = input(f'점 {i} (이미지 {uv[0]:.0f},{uv[1]:.0f}) 로봇 좌표 "x y" mm '
                      f'[엔터 = 직접교시로 둔 현재 TCP 읽기]: ').strip()
            if not s:
                xy = node.read_robot_xy()
                if xy is None:
                    print('  get_current_posx 응답 없음 (두산 bringup 확인) — 숫자로 입력하세요')
                    continue
                print(f'  현재 TCP: {xy[0]:.1f}, {xy[1]:.1f}')
            else:
                try:
                    xy = tuple(float(v) for v in s.replace(',', ' ').split()[:2])
                except ValueError:
                    continue
            tbl_pts.append(list(xy))
            break

    src, dst = np.float64(img_pts), np.float64(tbl_pts)
    H, _ = cv2.findHomography(src, dst)
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.linalg.norm(proj - dst, axis=1).max())
    out = {'H': H.tolist(), 'image_points': img_pts, 'table_points': tbl_pts,
           'table_z_mm': float(node.get_parameter('table_z_mm').value),
           'image_size': [int(node.frame.shape[1]), int(node.frame.shape[0])],
           'method': method, 'max_reprojection_error_mm': err,
           'created': time.strftime('%Y-%m-%d %H:%M:%S')}
    path = os.path.expanduser(node.get_parameter('calibration_file').value)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.safe_dump(out, f, allow_unicode=True)
    print(f'저장: {path} (4점 재투영 최대오차 {err:.2f} mm — 4점이면 0 에 가까움, 5번째 점으로 검증 권장)')
    print('주의: 웹캠을 움직이면 다시 캘리브레이션해야 합니다.')
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
