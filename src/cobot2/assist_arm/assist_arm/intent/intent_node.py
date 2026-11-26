"""[해석층] intent_node: 손동작·콕·음성·애매한 말 → 작업 요청 하나 (/assist/request).

한 프로세스 안의 노드들 (이름·토픽은 예전과 같음)
  gesture_node        제스처 (관찰 중 RealSense, 아니면 웹캠 / 이동 중엔 '멈춰'만)
  finger_pointer      콕 (RealSense 3D 검지 끝, 없으면 웹캠 호모그래피) + 안전용 웹캠 손 좌표
  request_router      제스처·음성·콕 조합 → /assist/request
  claude_planner_node 키워드로 못 알아들은 말 → 로컬 의도 분류(기본, 무료) / Claude(키 있을 때 선택)
  voice_command_node  'Hello Rokey' + STT (voice_command_node.enabled:=true 일 때만 마이크 사용)
"""
from assist_arm.common.multi import run_nodes


def main():
    from assist_arm.intent.gesture import GestureNode
    from assist_arm.intent.planner import ClaudePlanner
    from assist_arm.intent.pointer import FingerPointer
    from assist_arm.intent.router import RequestRouter
    from assist_arm.intent.voice import VoiceCommand

    def start_voice(nodes):
        for n in nodes:
            if isinstance(n, VoiceCommand) and n.get_parameter('enabled').value:
                n.start()

    run_nodes('intent_node', [('gesture_node', GestureNode), ('finger_pointer', FingerPointer),
                              ('request_router', RequestRouter), ('claude_planner_node', ClaudePlanner),
                              ('voice_command_node', VoiceCommand)], threads=8, after_start=start_voice)


if __name__ == '__main__':
    main()
