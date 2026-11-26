"""[인식층·손] hand_perception (담당: 호연, v2.3) — 손 인식 전부를 한 프로세스로.

  rs_hand_node      손목 RealSense (주): 손 21관절 + 깊이 3D. 관찰 자세(/assist/observing)에서만
                    → /rs/landmarks (손동작), /rs/hand_table (콕·안전, 베이스 mm)
  webcam_publisher  웹캠 영상 (보조) → /webcam/image_raw
  pose_node         웹캠 손 21점(2D)·상체 → /user/landmarks
                    → 로봇 이동 중 안전 감시·'멈춰', RealSense 가 손을 못 볼 때 대체

노드·토픽 이름은 예전(rs_perception + webcam_watch)과 같다. 얼룩(spill_node)은 object_perception 으로 옮김.
"""
from assist_arm.common.multi import run_nodes


def main():
    from assist_arm.perception.pose import PoseNode
    from assist_arm.perception.rs_hand import RsHandNode
    from assist_arm.perception.webcam import WebcamPublisher
    run_nodes('hand_perception', [('rs_hand_node', RsHandNode), ('webcam_publisher', WebcamPublisher),
                                  ('pose_node', PoseNode)])


if __name__ == '__main__':
    main()
