# /cobot2/log/task  (planner_node → db_logger_node, 동작마다)
string PLANNER_RL = "rl"
string PLANNER_RULE = "rule"

std_msgs/Header header
int32 episode                 # 회차
string mode                   # set_table / clear_table
string planner_type           # rl / rule
string object_type            # ObjectTypes
geometry_msgs/Pose from_pose  # 출발 위치 (base_link, m·rad)
geometry_msgs/Pose to_pose    # 도착 위치
string from_place             # 출발 장소 종류 (Skill.Goal.PLACE_*)
string to_place               # 도착 장소 종류
bool success
string error_code             # Skill.Result.ERR_* (성공이면 "OK")
float32 duration              # 소요 시간 (s)
