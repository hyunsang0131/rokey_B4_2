"""[테스트용] 가짜 skill_server — 로봇 없이 Pour / Skill 액션, RobotState, safety/stop 흐름 확인.

  ROS_DOMAIN_ID=77 ros2 run cobot2_hri fake_skill_server [--ros-args -p sim_level:=false]
- /cobot2/pour : PourModule + FakeRobot (실제 skill_server 에 붙일 모듈과 같은 코드), 수위는 /cobot2/pour/level
                 또는 sim_level:=true 면 가짜 수위
- /cobot2/skill: 아무 스킬이나 skill_sec 초 동안 MOVING 흉내 (Pour 와 같은 잠금 → 겹치면 BUSY)
- /cobot2/safety/stop true → 진행 중 동작 STOPPED, 이후 목표 거절 / false → 재개 허용
- /cobot2/robot/state 5 Hz
"""
import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool

from cobot2_interfaces.action import Pour, Skill
from cobot2_interfaces.msg import CupLevel, RobotState
from cobot2_hri.pour_module import FakeRobot, PourModule

STOP_QOS = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class FakeSkillServer(Node):
    def __init__(self):
        super().__init__('skill_server')
        self.declare_parameter('sim_level', True)
        self.declare_parameter('skill_sec', 3.0)
        self.lock = threading.Lock()               # Skill 과 Pour 가 공유하는 동작 잠금
        self.robot = FakeRobot(self.get_logger(), speed=8.0, sim_level=self.get_parameter('sim_level').value)
        self.pour = PourModule(self.robot, self.lock, self.get_logger(), {'step_wait_s': 0.2})
        self.stopped = False
        self.state, self.skill = RobotState.IDLE, ''
        cbg = ReentrantCallbackGroup()
        self.state_pub = self.create_publisher(RobotState, '/cobot2/robot/state', 10)
        self.create_subscription(Bool, '/cobot2/safety/stop', self.on_stop, STOP_QOS, callback_group=cbg)
        self.create_subscription(CupLevel, '/cobot2/pour/level', lambda m: self.pour.on_level(m.ratio, m.confidence),
                                 10, callback_group=cbg)
        ActionServer(self, Pour, '/cobot2/pour', self.exec_pour, goal_callback=self.accept,
                     cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cbg)
        ActionServer(self, Skill, '/cobot2/skill', self.exec_skill, goal_callback=self.accept,
                     cancel_callback=lambda _: CancelResponse.ACCEPT, callback_group=cbg)
        self.create_timer(0.2, self.pub_state, callback_group=cbg)
        self.get_logger().info('가짜 skill_server 준비 (로봇 없음)')

    def accept(self, goal):
        return GoalResponse.ACCEPT         # BUSY·STOPPED 는 실행에서 결과로 돌려줌 (설계: 거절 = 오류 코드)

    def on_stop(self, m):
        self.stopped = bool(m.data)
        self.pour.request_stop(self.stopped)
        if self.stopped:
            self.robot.stopped = True
            self.get_logger().warn('safety/stop = true → 정지')
        else:
            self.robot.stopped = False
            self.get_logger().info('safety/stop = false → 재개 허용 (자동으로 다시 움직이지는 않음)')

    def pub_state(self):
        m = RobotState(state=RobotState.STOPPED if self.stopped else self.state, current_skill=self.skill,
                       holding_type='bottle' if self.robot.holding else '', gripper_width=0.0 if self.robot.holding else 0.1)
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        self.state_pub.publish(m)

    def exec_pour(self, gh):
        g = gh.request
        self.state, self.skill = RobotState.MOVING, 'POUR'
        res = self.pour.execute(g.cup_pose, g.bottle_pose, g.target_ratio,
                                feedback=lambda r: gh.publish_feedback(Pour.Feedback(current_ratio=r)),
                                cancelled=lambda: gh.is_cancel_requested)
        self.state, self.skill = RobotState.IDLE, ''
        out = Pour.Result(final_ratio=res['final_ratio'], success=res['success'], error_code=res['error_code'])
        gh.succeed() if res['success'] else gh.abort()
        return out

    def exec_skill(self, gh):
        g = gh.request
        r = Skill.Result()
        if self.stopped:
            r.success, r.error_code = False, Skill.Result.ERR_STOPPED
            gh.abort()
            return r
        if not self.lock.acquire(blocking=False):
            r.success, r.error_code = False, Skill.Result.ERR_BUSY
            gh.abort()
            return r
        try:
            self.state, self.skill = RobotState.MOVING, g.skill
            t_end = time.monotonic() + self.get_parameter('skill_sec').value
            while time.monotonic() < t_end:
                if self.stopped or gh.is_cancel_requested:
                    r.success, r.error_code = False, Skill.Result.ERR_STOPPED
                    gh.abort()
                    return r
                gh.publish_feedback(Skill.Feedback(stage='moving'))
                time.sleep(0.1)
            r.success, r.error_code, r.gripper_width = True, Skill.Result.ERR_OK, 0.1
            gh.succeed()
            return r
        finally:
            self.state, self.skill = RobotState.IDLE, ''
            self.lock.release()


def main(args=None):
    rclpy.init(args=args)
    node = FakeSkillServer()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
