"""컵 수위 추정 (옆에서 보는 카메라 기준). pour_level_node 와 tools/pour_level_offline.py 공용.

설정 (config/pour_level.yaml, tools/pour_level_roi.py 로 클릭해서 만듦)
  roi      [x1, y1, x2, y2]  컵 안쪽 (벽 빼고) 픽셀 사각형
  y_bottom 컵 안쪽 바닥 줄 (수위 0)
  y_full   목표로 볼 '가득' 줄 (수위 1, 보통 컵 입구보다 조금 아래)
방법 (투명 컵 대응)
  1) 빈 컵 기준 사진이 있으면(capture_empty): 지금 − 빈 컵 차이가 '아래쪽은 크고 위쪽은 작은' 경계 줄 = 수면
  2) 없으면: 가로 방향 에지(수면 선) 세기 + 위/아래 밝기·색 차이가 가장 큰 줄 = 수면
  확신도 = 그 경계가 주변보다 얼마나 뚜렷한지 (0~1). 낮으면 ratio 는 직전 값을 유지하고 confidence 를 낮춤
  시간 스무딩: 최근 값 중앙값 + 지수 평균, 수위는 따르는 동안 거의 줄지 않으므로 큰 하락은 무시(옵션)
"""
from collections import deque

import cv2
import numpy as np


class LevelEstimator:
    def __init__(self, roi, y_bottom, y_full, min_conf=0.35, window=5, alpha=0.5, monotonic=True):
        self.roi = [int(v) for v in roi]
        self.y_bottom, self.y_full = float(y_bottom), float(y_full)
        self.min_conf, self.alpha, self.monotonic = min_conf, alpha, monotonic
        self.hist = deque(maxlen=window)
        self.empty = None
        self.ratio, self.conf = 0.0, 0.0

    def _crop(self, bgr):
        x1, y1, x2, y2 = self.roi
        lab = cv2.cvtColor(cv2.GaussianBlur(bgr[y1:y2, x1:x2], (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab

    def set_empty(self, bgr):
        """빈 컵 기준 사진 (따르기 시작 전, 컵이 비어 있을 때)"""
        self.empty = self._crop(bgr)
        self.hist.clear()
        self.ratio, self.conf = 0.0, 0.5

    def raw(self, bgr):
        """→ (수면 줄 y (전체 이미지 기준) | None, 확신도 0~1)"""
        lab = self._crop(bgr)
        y1 = self.roi[1]
        lo = int(max(0, min(self.y_full, self.y_bottom) - y1 - 5))
        hi = int(min(lab.shape[0], self.y_bottom - y1 + 1))
        if hi - lo < 8:
            return None, 0.0
        if self.empty is not None and self.empty.shape == lab.shape:
            d = np.linalg.norm(lab - self.empty, axis=2).mean(axis=1)          # 줄마다 빈 컵과의 차이
            c = np.cumsum(np.r_[0, d])
            best, best_s = None, -1e9
            for r in range(lo + 2, hi - 2):
                below = (c[hi] - c[r]) / (hi - r)
                above = (c[r] - c[lo]) / (r - lo)
                s = below - above
                if s > best_s:
                    best, best_s = r, s
            scale = np.std(d[lo:hi]) + 1.0
            conf = float(np.clip(best_s / (2.5 * scale), 0, 1))
            if d[lo:hi].max() < 4.0:            # 거의 차이 없음 = 빈 컵
                return float(self.y_bottom), 0.6
            return float(best + y1), conf
        L = lab[..., 0]
        edge = np.abs(cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
        prof = lab.mean(axis=1)
        step = np.zeros(len(L))
        k = 4
        for r in range(k, len(L) - k):
            step[r] = np.linalg.norm(prof[r:r + k].mean(axis=0) - prof[r - k:r].mean(axis=0))
        sc = edge / (edge[lo:hi].max() + 1e-6) + step / (step[lo:hi].max() + 1e-6)
        seg = sc[lo:hi]
        r = int(np.argmax(seg))
        prom = (seg[r] - np.median(seg)) / (np.std(seg) + 1e-6)
        return float(r + lo + y1), float(np.clip((prom - 1.5) / 3.0, 0, 1))

    def update(self, bgr):
        """→ (ratio 0~1, confidence 0~1) 스무딩 포함"""
        y, conf = self.raw(bgr)
        if y is not None and conf >= self.min_conf:
            r = float(np.clip((self.y_bottom - y) / max(1.0, self.y_bottom - self.y_full), 0.0, 1.0))
            if self.monotonic and self.hist and r < min(self.hist) - 0.15:
                conf *= 0.5                          # 따르는 중 갑자기 크게 줄어듦 → 의심
            else:
                self.hist.append(r)
                med = float(np.median(self.hist))
                self.ratio = med if self.conf == 0 else self.alpha * med + (1 - self.alpha) * self.ratio
        self.conf = 0.7 * self.conf + 0.3 * (conf if y is not None else 0.0)
        return self.ratio, float(self.conf)

    def draw(self, bgr):
        x1, y1, x2, y2 = self.roi
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 200, 0), 1)
        for yy, col in ((self.y_bottom, (0, 0, 255)), (self.y_full, (0, 255, 0))):
            cv2.line(bgr, (x1, int(yy)), (x2, int(yy)), col, 1)
        ys = self.y_bottom - self.ratio * (self.y_bottom - self.y_full)
        cv2.line(bgr, (x1, int(ys)), (x2, int(ys)), (0, 255, 255), 2)
        cv2.putText(bgr, f'{self.ratio:.2f} ({self.conf:.2f})', (x1, max(15, y1 - 6)), 0, 0.6, (0, 255, 255), 2)
        return bgr
