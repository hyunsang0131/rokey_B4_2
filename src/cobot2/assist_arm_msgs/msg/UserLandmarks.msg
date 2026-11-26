# pose_node(웹캠) 출력.
#   hand / hand2 : MediaPipe 정규화 이미지 좌표 (x,y: 0~1 이미지 비율, z: 손목 기준 상대 깊이)
#   hand_world   : MediaPipe hand world landmarks — 미터 단위 실제 3D (손 중심 원점). 카메라 거리·화면 위치에 덜 흔들림
std_msgs/Header header
int32 image_width
int32 image_height

bool hand_valid              # 가장 확실한 손 (score 높은 쪽)
string handedness            # "Left" / "Right" (MediaPipe 기준, 거울상 주의)
float32 hand_score
float32[] hand               # 21점 x (x, y, z) = 63개, 평탄화
bool hand_world_valid
float32[] hand_world         # 21점 x (x, y, z) 미터 = 63개

bool hand2_valid             # 두 번째 손 (num_hands=2 일 때)
string handedness2
float32 hand2_score
float32[] hand2
float32[] hand2_world

bool pose_valid
float32[] pose               # 33점 x (x, y, z, visibility) = 132개, 평탄화
