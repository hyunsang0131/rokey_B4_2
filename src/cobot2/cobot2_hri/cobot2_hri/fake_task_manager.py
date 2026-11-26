"""[테스트용] 가짜 task_manager — /cobot2/user/command 를 받아 액션으로 넘기는 흐름만 확인 (실제 담당은 상엽 권장).

  ROS_DOMAIN_ID=77 ros2 run cobot2_hri fake_task_manager
  pour        → /cobot2/pour (컵 = 자리 중심(seats.yaml), 물병 = bottle_xyz 파라미터)
  set/clear   → /cobot2/skill OBSERVE 한 번 (planner 대신 '움직이는 중' 흉내)
  stop        → (정지는 gesture/voice 가 safety/stop 을 이미 보냄) 진행 중 목표 취소
  resume      → /cobot2/safety/stop = false (재개 허용 — 이 노드만 false 를 보냄)
  진행 중이면 새 작업 거절 ("지금은 … 중이에요")
"""
import os

import rclpy
import yaml
from geometry_msgs.msg import Pose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from cobot2_interfaces.action import Pour, Skill
from cobot2_interfaces.msg import UserCommand

STOP_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def pose(x, y, z):
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.y = 1.0          # 그리퍼 수직 아래 (ry=180°)
    return p


class FakeTaskManager(Node):
    def __init__(self):
        super().__init__('task_manager')
        self.declare_parameter('bottle_xyz', [0.45, 0.30, 0.10])
        self.declare_parameter('cup_z', 0.05)
        from ament_index_python.packages import get_package_share_directory
        seats = yaml.safe_load(open(os.path.join(get_package_share_directory('cobot2_hri'), 'config', 'seats.yaml')))
        self.seat_xy = {int(s['id']): s['center'] for s in seats.get('seats', [])}
        self.pour_ac = ActionClient(self, Pour, '/cobot2/pour')
        self.skill_ac = ActionClient(self, Skill, '/cobot2/skill')
        self.stop_pub = self.create_publisher(Bool, '/cobot2/safety/stop', STOP_QOS)
        self.create_subscription(UserCommand, '/cobot2/user/command', self.on_cmd, 10)
        self.active = None              # (이름, goal_handle)
        self.get_logger().info('가짜 task_manager 준비')

    def on_cmd(self, m):
        c = m.command
        self.get_logger().info(f'명령 받음: {c} 자리 {list(m.seat_ids)} ({m.source})')
        if c == UserCommand.CMD_RESUME:
            self.stop_pub.publish(Bool(data=False))
            return
        if c in (UserCommand.CMD_STOP, UserCommand.CMD_CANCEL):
            if self.active and self.active[1] is not None:
                self.active[1].cancel_goal_async()
            return
        if self.active:
            self.get_logger().warn(f'지금은 {self.active[0]} 중이에요 → 거절')
            return
        if c == UserCommand.CMD_POUR:
            sid = m.seat_ids[0] if m.seat_ids else next(iter(self.seat_xy), None)
            xy = self.seat_xy.get(sid, [0.55, 0.0])
            g = Pour.Goal(cup_pose=pose(xy[0], xy[1], self.get_parameter('cup_z').value),
                          bottle_pose=pose(*self.get_parameter('bottle_xyz').value),
                          target_ratio=m.pour_ratio or 0.6)
            self._send('pour', self.pour_ac, g, lambda f: self.get_logger().info(
                f'  수위 {f.feedback.current_ratio:.2f}', throttle_duration_sec=1.0))
        elif c in (UserCommand.CMD_SET_TABLE, UserCommand.CMD_CLEAR_TABLE):
            self._send(c, self.skill_ac, Skill.Goal(skill=Skill.Goal.SKILL_OBSERVE), None)

    def _send(self, name, ac, goal, fb):
        if not ac.wait_for_server(timeout_sec=2.0):
            self.get_logger().error(f'{name} 액션 서버 없음')
            return
        self.active = (name, None)
        fut = ac.send_goal_async(goal, feedback_callback=fb)

        def on_goal(f):
            gh = f.result()
            self.active = (name, gh)
            gh.get_result_async().add_done_callback(
                lambda r: (self.get_logger().info(f'{name} 결과: {r.result().result}'), setattr(self, 'active', None)))
        fut.add_done_callback(on_goal)


def main(args=None):
    rclpy.init(args=args)
    node = FakeTaskManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
