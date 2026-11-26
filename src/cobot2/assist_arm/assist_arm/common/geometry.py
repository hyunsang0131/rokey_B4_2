"""좌표 변환. 기존 코드와 *동일한 수식*을 그대로 옮겨옴(원본 파일은 수정하지 않음).

- get_robot_pose_matrix / transform_to_base : robot_control/robot_control.py (RobotController)
  (robot_control.robot_control 모듈은 import 시 rclpy.init + 그리퍼 연결을 해버려서 import 불가 → 복사)
- pixel_to_camera / median_depth            : object_detection/detection.py
단위: 카메라 depth(mm) → 로봇 베이스 mm, 자세는 두산 ZYZ euler(deg).
"""
import os

import numpy as np
from scipy.spatial.transform import Rotation


def get_robot_pose_matrix(x, y, z, rx, ry, rz):
    R = Rotation.from_euler("ZYZ", [rx, ry, rz], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def load_gripper2cam(path=''):
    if not path:
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory('robot_control'),
                            'resource', 'T_gripper2camera.npy')
    return np.load(path)


def transform_to_base(camera_coords, gripper2cam, robot_pos):
    coord = np.append(np.array(camera_coords, dtype=float), 1)
    base2gripper = get_robot_pose_matrix(*robot_pos[:6])
    base2cam = base2gripper @ gripper2cam
    return np.dot(base2cam, coord)[:3]


def pixel_to_camera(x, y, z, intr):
    return ((x - intr['ppx']) * z / intr['fx'],
            (y - intr['ppy']) * z / intr['fy'],
            z)


def median_depth(depth, x, y, win=5):
    h, w = depth.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return None
    patch = depth[max(0, y - win):min(h, y + win + 1), max(0, x - win):min(w, x + win + 1)]
    valid = patch[patch > 0]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def tool_point(posx, offset_tool):
    """TCP 자세(posx) 기준 툴 좌표계 offset(mm) 점 → 베이스 좌표. (예: 숟가락 끝)"""
    T = get_robot_pose_matrix(*posx[:6])
    return (T @ np.append(np.asarray(offset_tool, dtype=float), 1.0))[:3]


def ray_plane_z(p0, p1, plane_z):
    """p0→p1 반직선과 z=plane_z 평면의 교점 (가리키기). 평행이거나 뒤쪽이면 None."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    if abs(d[2]) < 1e-6:
        return None
    t = (plane_z - p0[2]) / d[2]
    if t <= 0:
        return None
    return p0 + t * d


def pixel_ray_to_plane(u, v, intr, gripper2cam, robot_pos, plane_z):
    """깊이가 없을 때: 카메라에서 픽셀 (u,v) 방향 광선이 베이스 z=plane_z 평면(테이블)과 만나는 점."""
    d_cam = np.array([(u - intr['ppx']) / intr['fx'], (v - intr['ppy']) / intr['fy'], 1.0])
    base2cam = get_robot_pose_matrix(*robot_pos[:6]) @ gripper2cam
    o, d = base2cam[:3, 3], base2cam[:3, :3] @ d_cam
    if abs(d[2]) < 1e-9:
        return None
    t = (plane_z - o[2]) / d[2]
    return None if t <= 0 else o + t * d
