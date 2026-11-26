"""테이블 얼룩 감지: 손목 RealSense 로 테이블을 내려다본 상태에서 '깨끗한 테이블' 기준 이미지와 비교.

흐름 (task_manager 의 wipe_table 서브태스크가 호출)
  1) 로봇이 table_view 자세로 이동 후 정지
  2) /spill/capture_reference (Trigger): 깨끗한 테이블 기준 이미지 저장 (같은 table_view 자세에서!)
     → 기준은 reference_path(npz) 에도 저장되어 재시작 후에도 사용
  3) /spill/scan (Trigger): scan_frames 프레임 평균과 기준의 LAB 차이 → 가장 큰 얼룩
     → 중심 픽셀 depth → 카메라 좌표 → T_gripper2camera + 현재 posx → 베이스 좌표
     → /spill/result (SpillResult) 발행, Trigger 응답 message 에 JSON 요약
  분류: model_path(.onnx, 1x3x64x64 RGB 0~1, [none, liquid, powder]) 있으면 사용,
        없으면 placeholder (어두워짐=liquid, 밝아짐=powder).
  table_roi: 기준 이미지에서 테이블만 보이는 픽셀 사각형 [x1,y1,x2,y2] (빈 리스트면 전체)
"""
import json
import os
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from assist_arm_msgs.msg import SpillResult
from assist_arm.common.heartbeat import attach_heartbeat
from assist_arm.common.rs_camera import WristCamera

LABELS = ['none', 'liquid', 'powder']


class SpillNode(Node):
    def __init__(self):
        super().__init__('spill_node')
        p = self.declare_parameter
        p('robot_id', 'dsr01')
        p('gripper2cam_path', '')
        p('table_roi', [0, 0, 0, 0])        # [x1,y1,x2,y2], 0 이면 전체 이미지
        p('reference_path', '~/data/assist_arm/table_reference.npz')
        p('scan_frames', 10)
        p('diff_threshold', 25.0)           # LAB 거리
        p('min_area_px', 300)
        p('table_z_mm', 0.0)                # depth 가 비면 이 높이로 대체
        p('model_path', '')
        p('show_debug', False)
        g = lambda n: self.get_parameter(n).value  # noqa: E731

        cbg = ReentrantCallbackGroup()
        self.cam = WristCamera(self, cbg, g('robot_id'), g('gripper2cam_path'), min_depth=100.0,
                               max_depth=2000.0)
        self.ref = None
        self.ref_path = os.path.expanduser(g('reference_path'))
        if os.path.exists(self.ref_path):
            self.ref = np.load(self.ref_path)['lab']
            self.get_logger().info(f'기준 테이블 이미지 로드: {self.ref_path}')
        self.sess = None
        mp = os.path.expanduser(g('model_path'))
        if mp and os.path.exists(mp):
            import onnxruntime as ort
            self.sess = ort.InferenceSession(mp, providers=['CPUExecutionProvider'])
        else:
            self.get_logger().warn('얼룩 분류 모델 없음 → placeholder(밝기 규칙)')
        self.pub = self.create_publisher(SpillResult, '/spill/result', 10)
        self.create_service(Trigger, '/spill/capture_reference', self.on_capture, callback_group=cbg)
        self.create_service(Trigger, '/spill/scan', self.on_scan, callback_group=cbg)
        attach_heartbeat(self)

    # ---------------------------------------------------------------- 영상
    def _avg_frames(self, n):
        labs, depths, t_end = [], [], time.monotonic() + 5.0
        last = None
        while len(labs) < n and time.monotonic() < t_end:
            bgr, depth = self.cam.snapshot()
            if bgr is not None and (last is None or not np.array_equal(bgr[::40, ::40], last)):
                last = bgr[::40, ::40]
                labs.append(cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2LAB)
                            .astype(np.float32))
                depths.append(depth)
            time.sleep(0.03)
        if not labs:
            return None, None, None
        return np.mean(labs, axis=0), depths[-1], bgr

    def _roi(self, shape):
        x1, y1, x2, y2 = [int(v) for v in self.get_parameter('table_roi').value]
        h, w = shape[:2]
        if x2 <= x1 or y2 <= y1:
            return 0, 0, w, h
        return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

    def on_capture(self, req, res):
        lab, _, _ = self._avg_frames(int(self.get_parameter('scan_frames').value))
        if lab is None:
            res.success, res.message = False, 'RealSense 영상 없음'
            return res
        self.ref = lab
        os.makedirs(os.path.dirname(self.ref_path), exist_ok=True)
        np.savez_compressed(self.ref_path, lab=lab)
        res.success, res.message = True, f'기준 저장: {self.ref_path}'
        self.get_logger().info(res.message)
        return res

    def _classify(self, bgr, lab, mask, box):
        x1, y1, x2, y2 = box
        if self.sess is not None:
            crop = cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            x = (cv2.resize(crop, (64, 64)).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            out = np.asarray(self.sess.run(None, {self.sess.get_inputs()[0].name: x})[0]).reshape(-1)
            pr = np.exp(out - out.max())
            pr /= pr.sum()
            i = int(pr.argmax())
            return LABELS[i], float(pr[i]), 'model'
        dL = float((lab[..., 0][mask > 0] - self.ref[..., 0][mask > 0]).mean())
        if dL < -8:
            return 'liquid', min(0.9, 0.5 + abs(dL) / 60), 'placeholder'
        if dL > 8:
            return 'powder', min(0.9, 0.5 + abs(dL) / 60), 'placeholder'
        return 'none', 0.5, 'placeholder'

    def on_scan(self, req, res):
        out = SpillResult()
        out.header.stamp = self.get_clock().now().to_msg()
        out.label, out.classifier = 'none', 'placeholder' if self.sess is None else 'model'
        if self.ref is None:
            res.success, res.message = False, '기준 테이블 이미지 없음 → /spill/capture_reference 먼저'
            return res
        lab, depth, bgr = self._avg_frames(int(self.get_parameter('scan_frames').value))
        if lab is None or lab.shape != self.ref.shape:
            res.success, res.message = False, '영상 없음 또는 해상도 불일치'
            return res
        x1, y1, x2, y2 = self._roi(lab.shape)
        diff = np.linalg.norm(lab - self.ref, axis=2)
        mask = np.zeros(diff.shape, np.uint8)
        mask[y1:y2, x1:x2] = (diff[y1:y2, x1:x2] > self.get_parameter('diff_threshold').value) * 255
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [c for c in cnts if cv2.contourArea(c) >= self.get_parameter('min_area_px').value]
        if not cnts:
            self.pub.publish(out)
            res.success, res.message = True, json.dumps({'detected': False})
            return res
        c = max(cnts, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(c)
        cmask = np.zeros_like(mask)
        cv2.drawContours(cmask, [c], -1, 255, -1)
        out.label, out.confidence, out.classifier = self._classify(bgr, lab, cmask, (bx, by, bx + bw, by + bh))
        out.detected = out.label != 'none'
        out.bbox = [bx, by, bx + bw, by + bh]
        out.area_ratio = float(cv2.contourArea(c) / max(1, (x2 - x1) * (y2 - y1)))
        M = cv2.moments(c)
        u, v = M['m10'] / M['m00'], M['m01'] / M['m00']
        cams = [self.cam.pixel_to_cam(u, v, depth, 7),
                self.cam.pixel_to_cam(bx, by, depth, 7), self.cam.pixel_to_cam(bx + bw, by + bh, depth, 7)]
        if cams[0] is None:
            # 얼룩(액체)은 depth 가 비기 쉬움 → 주변 depth 중앙값으로 대체
            valid = depth[max(0, by - 20):by + bh + 20, max(0, bx - 20):bx + bw + 20]
            valid = valid[valid > 0]
            if valid.size:
                from assist_arm.common.geometry import pixel_to_camera
                cams[0] = pixel_to_camera(u, v, float(np.median(valid)), self.cam.intr)
        done = {}

        def on_base(bases):
            done['b'] = bases
        self.cam.to_base_async(cams, on_base)
        t_end = time.monotonic() + 2.0
        while 'b' not in done and time.monotonic() < t_end:
            time.sleep(0.02)
        bases = done.get('b')
        if bases and bases[0] is not None:
            out.position_valid = True
            out.position.x, out.position.y, out.position.z = map(float, bases[0])
            if bases[1] is not None and bases[2] is not None:
                out.size_x_mm, out.size_y_mm = [float(abs(a - b)) for a, b in zip(bases[1][:2], bases[2][:2])]
        self.pub.publish(out)
        res.success = True
        res.message = json.dumps({
            'detected': out.detected, 'label': out.label, 'confidence': round(out.confidence, 3),
            'position_valid': out.position_valid,
            'position': [out.position.x, out.position.y, out.position.z],
            'size_mm': [out.size_x_mm, out.size_y_mm]})
        self.get_logger().info(f'scan: {res.message}')
        if self.get_parameter('show_debug').value:
            cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
            cv2.imshow('spill', bgr)
            cv2.waitKey(1)
        return res


def main(args=None):
    rclpy.init(args=args)
    node = SpillNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
