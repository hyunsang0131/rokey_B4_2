# safety_monitor 출력 (/safety/status). 정지 여부만 필요하면 /safety/stop (std_msgs/Bool)
std_msgs/Header header
bool stop
string reason                # gesture_stop / voice_stop / heartbeat:<node> / speed / manual
string mode                  # normal / dance
bool paused                  # 사람 손/팔이 작업 지점 근처 → 일시정지 중
float32 hand_distance_mm     # 손/팔 ↔ 작업 지점 최소 거리, -1 이면 측정 안 됨
