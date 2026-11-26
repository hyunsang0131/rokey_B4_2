# gesture_node 출력
std_msgs/Header header
string label                 # cup_request / stop / idle / (kNN 등록 라벨)
float32 confidence
bool confirmed               # 연속 N프레임 동일 → true
string source                # "onnx" / "sklearn" / "knn" / "rule"
