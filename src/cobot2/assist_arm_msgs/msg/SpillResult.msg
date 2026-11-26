# spill_node 출력 (/spill/result) — 테이블 위 얼룩 (손목 RealSense 로 테이블을 내려다본 상태에서 측정)
std_msgs/Header header
bool detected
string label                     # none / liquid / powder
float32 confidence
string classifier                # "model" / "placeholder"
float32 area_ratio               # 테이블 ROI 대비 면적 비율
int32[] bbox                     # [x1, y1, x2, y2] (RealSense color 픽셀)
bool position_valid
geometry_msgs/Point position     # 얼룩 중심, 로봇 베이스 좌표 (mm)
float32 size_x_mm                # 베이스 X/Y 방향 대략 크기 (mm)
float32 size_y_mm
