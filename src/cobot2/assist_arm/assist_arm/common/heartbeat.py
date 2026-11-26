"""모든 assist 노드는 /assist/heartbeat 에 자기 이름을 주기적으로 발행 → safety_monitor 가 감시."""
from std_msgs.msg import String

HEARTBEAT_TOPIC = '/assist/heartbeat'


def attach_heartbeat(node, name=None, period=0.5):
    pub = node.create_publisher(String, HEARTBEAT_TOPIC, 10)
    msg = String(data=name or node.get_name())
    node._assist_hb_alive = True   # 인지 루프가 멈췄을 때 False 로 바꿔 heartbeat 를 끊을 수 있음

    def _tick():
        if node._assist_hb_alive:
            pub.publish(msg)
    node._assist_hb_timer = node.create_timer(period, _tick)
    return pub
