# /cobot2/robot/state  (skill_server → 모든 노드, 5 Hz)
uint8 IDLE = 0         # 대기
uint8 MOVING = 1       # 동작
uint8 STOPPED = 2      # 정지 (safety/stop)
uint8 ERROR = 3        # 오류

std_msgs/Header header
uint8 state
string current_skill   # 실행 중인 스킬 이름 (Skill 상수 문자열, 없으면 "")
string holding_type    # 들고 있는 물체 종류 (ObjectTypes, 빈손이면 "")
float32 gripper_width  # 그리퍼 폭 (m)
