# GetScene 응답의 물체 하나 (perception_node)
string type                   # ObjectTypes
geometry_msgs/Pose pose       # base_link 기준 위치·방향 (m, 쿼터니언)
geometry_msgs/Vector3 size    # 크기 (m): x, y, z
float32 confidence            # 0~1
