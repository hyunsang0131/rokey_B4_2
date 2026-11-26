# request_router → task_manager (/assist/request)
std_msgs/Header header
string task        # bring_dish / clear_dish / clear_empty / clear_all / pour_water / seat_setting /
                   # music / wipe_table / cancel
string args        # JSON (예: {"food": "kimchi"}, {"point": [x,y,z]}, {"seats": 2}, {"mood": "신나는"})
string source      # gesture / voice / poke / combo / manual
string raw         # 원문 (음성 텍스트 또는 제스처 라벨)
