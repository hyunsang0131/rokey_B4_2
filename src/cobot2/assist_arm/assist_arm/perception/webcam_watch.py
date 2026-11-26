"""[인식층] webcam_watch: 웹캠(아무 USB 캠) — 안전 감시용 보조 인식.

  webcam_publisher (/webcam/image_raw) + pose_node (MediaPipe 손 21점·상체 → /user/landmarks)
  → safety(손·팔 근접 일시정지), intent(멈춰 손바닥, 로봇 이동 중엔 '멈춰'만, RealSense 에 손 없을 때 대신)
"""
from assist_arm.common.multi import run_nodes


def main():
    from assist_arm.perception.pose import PoseNode
    from assist_arm.perception.webcam import WebcamPublisher
    run_nodes('webcam_watch', [('webcam_publisher', WebcamPublisher), ('pose_node', PoseNode)])


if __name__ == '__main__':
    main()
