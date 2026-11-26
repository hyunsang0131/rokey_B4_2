"""웹캠 이미지 → 테이블 평면(base_link x,y, m) 호모그래피 + 자리 영역 (config/seats.yaml).

캘리브레이션 파일 (tools/calibrate_webcam_table.py 가 만듦): H(3x3), unit('m' 또는 예전 assist_arm 의 'mm'), table_z
"""
import os

import numpy as np
import yaml


def load_calibration(path):
    """→ (H 이미지 px → base_link x,y [m], table_z [m]) 또는 (None, 0)"""
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return None, 0.0
    c = yaml.safe_load(open(path))
    H = np.array(c['H'], np.float64).reshape(3, 3)
    unit = c.get('unit', 'mm')                 # assist_arm 예전 파일은 mm
    s = 1.0 if unit == 'm' else 0.001
    if s != 1.0:
        H = np.diag([s, s, 1.0]) @ H
    tz = float(c.get('table_z_m', c.get('table_z_mm', 0.0) * 0.001))
    return H, tz


def to_table(H, uv):
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    q = np.hstack([uv, np.ones((len(uv), 1))]) @ H.T
    return q[:, :2] / q[:, 2:3]


class Zones:
    """seats.yaml: seats[{id, center[x,y], size[w,h]}] (m), robot_zone{center, size}, proximity_m"""

    def __init__(self, path):
        c = yaml.safe_load(open(os.path.expanduser(path))) or {}
        self.seats = [(int(s['id']), np.array(s['center'], float), np.array(s['size'], float)) for s in c.get('seats', [])]
        rz = c.get('robot_zone')
        self.robot_zone = (np.array(rz['center'], float), np.array(rz['size'], float)) if rz else None
        self.proximity_m = float(c.get('proximity_m', 0.20))

    @staticmethod
    def _inside(p, center, size, margin=0.0):
        return bool(np.all(np.abs(np.asarray(p) - center) <= size / 2 + margin))

    def seat_of(self, p):
        """점이 들어 있는 자리 id (없으면 None)"""
        for sid, c, s in self.seats:
            if self._inside(p, c, s):
                return sid
        return None

    def nearest_seat(self, p, max_dist=0.5):
        if not self.seats:
            return None
        d = [(np.linalg.norm(np.asarray(p) - c), sid) for sid, c, _ in self.seats]
        dist, sid = min(d)
        return sid if dist <= max_dist else None

    def near_robot(self, pts):
        """로봇 작업 영역 + proximity_m 안에 들어온 점이 있으면 True"""
        if self.robot_zone is None or pts is None or not len(pts):
            return False
        c, s = self.robot_zone
        return any(self._inside(p, c, s, self.proximity_m) for p in np.asarray(pts))
