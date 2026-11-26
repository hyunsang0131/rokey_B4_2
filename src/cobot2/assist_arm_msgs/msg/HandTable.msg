# finger_pointer 출력 (/user/hand_table): 웹캠 손·팔 점들을 테이블 평면(호모그래피)으로 옮긴 로봇 베이스 좌표 (mm)
# safety_monitor 가 로봇 작업 지점 20cm 근접 감시에 사용
std_msgs/Header header
bool valid
geometry_msgs/Point fingertip          # 검지 끝 (landmark 8)
geometry_msgs/Point palm               # 손바닥 중심
geometry_msgs/Point[] points           # 손 21점 + (보이면) 팔꿈치·손목(Pose 13/14/15/16)
string gesture_hint                    # pointing / open / other
