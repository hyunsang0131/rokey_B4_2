# /cobot2/user/command  (voice_node, gesture_node → task_manager, 이벤트)
string SOURCE_VOICE = "voice"
string SOURCE_GESTURE = "gesture"

string CMD_SET_TABLE = "set_table"
string CMD_CLEAR_TABLE = "clear_table"
string CMD_POUR = "pour"
string CMD_STOP = "stop"
string CMD_RESUME = "resume"
string CMD_CANCEL = "cancel"

std_msgs/Header header
string source          # SOURCE_*
string command         # CMD_*
int32[] seat_ids       # 대상 자리 번호 (비면 전체/해당 없음)
float32 pour_ratio     # 따르기 목표 비율 0~1 (command == pour 일 때)
