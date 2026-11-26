"""[인식층·물체] object_perception (담당: 은상, v2.3) — 물체·장면 인식 중 ROS 노드 부분.

  spill_node   손목 RealSense 로 테이블 얼룩 → /spill/result, 서비스 /spill/scan, /spill/capture_reference
  (물체 검출·3D 위치는 YOLO 컨테이너 assist_yolo 의 /assist/get_3d_position 서비스 — 별도 실행)

노드·토픽·서비스 이름은 예전(rs_perception 안의 spill_node)과 같다.
"""
from assist_arm.common.multi import run_nodes


def main():
    from assist_arm.perception.spill import SpillNode
    run_nodes('object_perception', [('spill_node', SpillNode)])


if __name__ == '__main__':
    main()
