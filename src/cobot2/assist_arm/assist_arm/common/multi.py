"""여러 노드를 한 프로세스에서 같이 돌리기 (hand_perception / object_perception / intent_node 공용).

노드 이름·토픽·파라미터는 그대로라 config/assist_arm.yaml 의 노드별 설정이 그대로 적용된다.
하나가 만들어지다 실패해도(예: MediaPipe 없음) 나머지는 계속 돈다 — 실패 이유는 로그로.
"""
import sys

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor


def run_nodes(process_name, factories, threads=6, after_start=None):
    """factories: [(이름, 노드를 만드는 함수)]. after_start(nodes): 시작 후 추가 작업 (예: 음성 스레드)"""
    rclpy.init()
    nodes = []
    for name, make in factories:
        try:
            nodes.append(make())
        except Exception as e:  # noqa: BLE001
            print(f'[{process_name}] {name} 시작 실패 → 이 부분만 빼고 실행: {e}', file=sys.stderr)
    if not nodes:
        print(f'[{process_name}] 실행할 노드가 없어요', file=sys.stderr)
        rclpy.try_shutdown()
        sys.exit(1)
    ex = MultiThreadedExecutor(num_threads=threads)
    for n in nodes:
        ex.add_node(n)
    if after_start:
        after_start(nodes)
    print(f"[{process_name}] 실행 중: {', '.join(n.get_name() for n in nodes)}")
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # noqa: BLE001  (종료 신호 중)
        if rclpy.ok():
            raise
    finally:
        for n in nodes:
            try:
                n.destroy_node()
            except Exception:  # noqa: BLE001
                pass
        rclpy.try_shutdown()
