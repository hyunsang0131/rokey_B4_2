# request_router → task_manager (/assist/request)
std_msgs/Header header
string task        # bring_dish / clear_dish / clear_empty / clear_all / pour_water / seat_setting /
                   # music / wipe_table / cancel /
                   # resume  (v2.2: 정지 해제(/safety/reset) 뒤 담당자 확인 → 들고 있으면 복구 → 다시 관찰)
string args        # JSON (예: {"food": "kimchi"}, {"point": [x,y,z]}, {"seats": 2}, {"mood": "신나는"})
string source      # gesture / voice / poke / combo / manual
string raw         # 원문 (음성 텍스트 또는 제스처 라벨)
