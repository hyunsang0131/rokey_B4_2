"""[예전 이름, v2.3부터 런치에서 안 씀 → hand_perception + object_perception] rs_perception: 손목 RealSense 하나로 — 손 21점 3D (rs_hand) + 테이블 얼룩 (spill).

  /rs/landmarks, /rs/hand_table (관찰 자세에서만), /spill/scan·/spill/capture_reference 서비스, /spill/result
식탁 스냅샷(그릇·컵 YOLO + depth → 3D, 빈 그릇)은 task_manager 의 observe_table 이 컨테이너 YOLO 서비스로 만든다.
"""
from assist_arm.common.multi import run_nodes


def main():
    from assist_arm.perception.rs_hand import RsHandNode
    from assist_arm.perception.spill import SpillNode
    run_nodes('rs_perception', [('rs_hand_node', RsHandNode), ('spill_node', SpillNode)])


if __name__ == '__main__':
    main()
