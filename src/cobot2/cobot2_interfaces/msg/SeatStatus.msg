# /cobot2/hri/seat_status  (gesture_node → planner_node, 바뀔 때 + 1 Hz 반복)
uint8 EMPTY = 0        # 비어 있음
uint8 EATING = 1       # 먹는 중
uint8 FINISHED = 2     # 다 먹음

std_msgs/Header header
int32 seat_id
uint8 status           # EMPTY / EATING / FINISHED
