"""음성 명령 → /voice/command (std_msgs/String, STT 원문). intent_node 안에서 별도 스레드로 돈다.

기존 voice_processing 의 WakeupWord("Hello Rokey") + STT(Whisper) 를 그대로 재사용 (같은 resource/.env).
  웨이크워드 → record_sec 초 녹음 → STT → 텍스트 발행 → router 가 키워드/의도로 작업 요청, safety 는 "멈춰" 검사.
주의: 기존 get_keyword 노드와 마이크를 동시에 쓰면 충돌 → 둘 중 하나만.
"""
import os
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class VoiceCommand(Node):
    def __init__(self):
        super().__init__('voice_command_node')
        self.declare_parameter('enabled', False)        # true 면 마이크 사용 (get_keyword 와 동시 사용 금지)
        self.declare_parameter('record_sec', 3)
        self.pub = self.create_publisher(String, '/voice/command', 10)
        self.thread = None

    def start(self):
        """마이크 루프를 백그라운드 스레드로 시작. 준비 실패하면 False"""
        try:
            from ament_index_python.packages import get_package_share_directory
            from dotenv import load_dotenv
            from voice_processing.stt import STT
            from voice_processing.wakeup_word import WakeupWord
            load_dotenv(os.path.join(get_package_share_directory('voice_processing'), 'resource', '.env'))
            self.stt = STT(openai_api_key=os.getenv('OPENAI_API_KEY'))
            self.stt.duration = int(self.get_parameter('record_sec').value)
            self.ww = WakeupWord()
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f'voice_processing 재사용 실패 → 음성 꺼짐: {e}')
            return False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self.get_logger().info("'Hello Rokey' 후 명령 (예: 다 먹었어 / 김치 줘 / 물 줘 / 멈춰)")
        return True

    def _loop(self):
        while rclpy.ok():
            try:
                self.ww.open()
                try:
                    while rclpy.ok() and not self.ww.is_wakeup():
                        pass
                finally:
                    self.ww.close()
                text = self.stt.speech2text()
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'음성 처리 실패: {e}')
                continue
            self.get_logger().info(f'음성: {text}')
            self.pub.publish(String(data=text))
