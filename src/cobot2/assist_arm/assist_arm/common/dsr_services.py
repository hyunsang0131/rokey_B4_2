"""두산 컨트롤러 서비스를 DR_init/DSR_ROBOT2 없이 비동기로 부르는 얇은 클라이언트.

DSR_ROBOT2 는 import 시 전역 노드를 잡고 spin_until_future_complete 로 막기 때문에,
여러 콜백에서 쓰는 감시/인지 노드(safety_monitor, spill_node, calibrate_pointer)는 이 클라이언트를 쓴다.
서비스 이름은 dsr_controller2 규칙: /<robot_id>/dsr_controller2/<group>/<name>
(rokey_cobot2/basic/get_current_pos.py 와 같은 접두어)
"""
from dsr_msgs2.srv import GetCurrentPosx, GetToolForce, MovePause, MoveResume, MoveStop

DR_QSTOP_STO, DR_QSTOP, DR_SSTOP, DR_HSTOP = 0, 1, 2, 3


class DsrServices:
    def __init__(self, node, robot_id='dsr01', controller='dsr_controller2'):
        pre = f'/{robot_id}/{controller}/'
        self.node = node
        self.posx_cli = node.create_client(GetCurrentPosx, pre + 'aux_control/get_current_posx')
        self.force_cli = node.create_client(GetToolForce, pre + 'aux_control/get_tool_force')
        self.stop_cli = node.create_client(MoveStop, pre + 'motion/move_stop')
        self.pause_cli = node.create_client(MovePause, pre + 'motion/move_pause')
        self.resume_cli = node.create_client(MoveResume, pre + 'motion/move_resume')

    def request_posx(self, callback, ref=0):
        """callback(posx_list[6] | None)"""
        if not self.posx_cli.service_is_ready():
            callback(None)
            return
        req = GetCurrentPosx.Request()
        req.ref = ref
        fut = self.posx_cli.call_async(req)

        def _done(f):
            try:
                r = f.result()
                if r and r.success and r.task_pos_info:
                    callback(list(r.task_pos_info[0].data)[:6])
                    return
            except Exception as e:  # noqa: BLE001
                self.node.get_logger().warn(f'get_current_posx 실패: {e}')
            callback(None)
        fut.add_done_callback(_done)

    def move_stop(self, mode=DR_QSTOP):
        if not self.stop_cli.service_is_ready():
            self.node.get_logger().error('move_stop 서비스 없음 — 펜던트 비상정지 사용!')
            return False
        req = MoveStop.Request()
        req.stop_mode = int(mode)
        self.stop_cli.call_async(req)
        return True

    def move_pause(self):
        """진행 중 모션 일시정지 (사람 손이 작업 지점 근처). 서비스 없으면 False."""
        if not self.pause_cli.service_is_ready():
            self.node.get_logger().warn('move_pause 서비스 없음', throttle_duration_sec=5.0)
            return False
        self.pause_cli.call_async(MovePause.Request())
        return True

    def move_resume(self):
        if not self.resume_cli.service_is_ready():
            return False
        self.resume_cli.call_async(MoveResume.Request())
        return True
