# assist_arm 명령어 모음 (4층 구조)

모든 명령은 노트북 터미널에서. `<이름>` 은 직접 바꿔 넣기.
**로봇이 실제로 움직이는 건 `bringup_all.launch.py` (dry_run 아닐 때) 와 `task_manager` 뿐**이에요. 펜던트 비상정지 담당자를 꼭 세워 두세요.

## 구조 한눈에

| 층 | 실행 파일 (`ros2 run assist_arm …`) | 하는 일 |
|---|---|---|
| 인식 (손, 호연) | `hand_perception` | 손목 RealSense(주): 손 21점 → 로봇 좌표 3D (관찰 자세에서만) + 웹캠(보조): 안전 감시용 손·상체 (멈춰 손바닥, 손·팔 근접) |
| 인식 (물체, 은상) | `object_perception` | 테이블 얼룩 (물체 검출은 YOLO 컨테이너) |
| 해석 | `intent_node` | 제스처 + 콕 + 음성 + 애매한 말 해석 → 작업 요청 하나 (`/assist/request`) |
| 결정·실행 | `task_manager` | 상태 머신 + 동작(skills). 작업 뒤 관찰 자세로 돌아와 식탁 스냅샷 |
| 결정·실행 | `decision_node` | (선택) 먼저 제안 "물 더 드릴까요?" + 반응으로 학습 |
| 결정·실행 | `music_node` | (선택) 음악 추천·재생 |
| 안전 | `safety_monitor` | 멈춰(손바닥·음성) 정지, 손 20cm 일시정지/재개, 신호 끊김 정지 |

YOLO(그릇·컵)는 도커 컨테이너의 `assist_detection`. 녹화·비교·학습·DB 스크립트는 `tools/`.

기능 우선순위: **P0** 치우기(clear)·가져오기(bring) / **P1** 물(water)·자리 세팅(seat) / **P2** 제안(decision)·음악(music)·닦기(wipe)

---

## ① 처음 한 번 / 터미널마다

```bash
# 터미널마다
source /opt/ros/jazzy/setup.bash
source ~/ws_cobot_pjt/ws_dsr/install/setup.bash
source ~/cobot2_ws/install/setup.bash
# 빌드 (코드 고친 뒤에도 이것만)
cd ~/cobot2_ws && colcon build --packages-select assist_arm_msgs assist_arm
```

이미 설치돼 있음: `~/venvs/assist_mp` (mediapipe 1.2.3), `~/venvs/assist_rtm` (openvino·rtmlib, 비교용),
모델 `~/data/assist_arm/models/` (hand_landmarker.task, pose_landmarker_lite.task, rtm/), Arc GPU 런타임.

카메라·로봇·YOLO 켜기 (각각 다른 터미널):
```bash
sod && sodreal            # 두산 로봇 (실제)
realsense                 # RealSense (~/.bashrc 별칭 = ros2 launch realsense2_camera rs_launch.py camera_namespace:=/ align_depth.enable:=true)
docker start yolo-detection && docker exec -it yolo-detection bash
#   (컨테이너 안, 처음 한 번) cd /ros2_ws && colcon build --packages-select assist_yolo
#   (컨테이너 안) source install/setup.bash && ros2 run assist_yolo assist_detection --ros-args -p model_path:=yolov8n.pt
```

준비 작업:
- **관찰 자세 교시**: 직접교시로 테이블 위 50~70cm 에 두고 관절값을 `config/waypoints.yaml` 의 `table_view_j` 에
- **웹캠 → 테이블 좌표 캘리브레이션** (웹캠 근접 감시용, 웹캠 옮기면 다시):
  ```bash
  python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/calibrate_pointer.py --ros-args -p method:=click
  ```
- **얼룩 기준 사진** (깨끗한 테이블, 관찰 자세에서): `ros2 service call /spill/capture_reference std_srvs/srv/Trigger`

---

## ② 실행 (launch 3개)

| 명령 | 켜지는 것 | 로봇 |
|---|---|---|
| `ros2 launch assist_arm perception.launch.py` | hand_perception, object_perception, intent_node (인식·해석 확인) | 안 움직임 |
| `ROS_DOMAIN_ID=77 ros2 launch assist_arm dry_run.launch.py` | 전부, task_manager 는 로그만 | 안 움직임 |
| `ros2 launch assist_arm bringup_all.launch.py` | 전부 | **움직임** |

`bringup_all` / `dry_run` 인자:
- `tasks:=clear,bring,water,seat` (기본 P0+P1) — `clear` `bring` `water` `seat` `music` `wipe` 또는 `all`
- `decision:=true` 먼저 제안 켜기 (기본 꺼짐), `voice:=true` 음성 명령 (get_keyword 와 동시 사용 금지)
- `show_debug:=true` 인식 화면 창

정지 해제 (사람이 확인한 뒤): `ros2 service call /safety/reset std_srvs/srv/Trigger`
키보드로 요청 (테스트): `ros2 topic pub --once /assist/request_text std_msgs/msg/String "{data: '다 먹었어'}"`
제스처 즉석 등록: `ros2 service call /gesture/register assist_arm_msgs/srv/RegisterGesture "{label: clear, duration_sec: 5.0, clear: true}"`

---

## ③ 녹화 (tools/, 로봇 무관)

관절 실시간 화면 + 손동작 녹화:
```bash
# 웹캠 (아무 USB 캠, ROS 불필요)
~/venvs/assist_mp/bin/python ~/cobot2_ws/src/cobot2/assist_arm/tools/live_record.py --person <이름> [--device 0]
# RealSense (먼저 터미널 하나에서 realsense)
source /opt/ros/jazzy/setup.bash
~/venvs/assist_mp/bin/python ~/cobot2_ws/src/cobot2/assist_arm/tools/live_record.py --source realsense --person <이름>
```
- 키: `1~9` 동작 (1 idle, 2 poke, 3 clear, 4 meal_done, 5 cup_grab, 6 stop, 7 point_seat, 8 v_sign, 9 hold_cup),
  `space` 2초 녹화, `a` 자동, `u` 방금 것 삭제, `n` 관절 번호, `q` 종료
  - 5 cup_grab = 빈손 C자 = "물 줘" 요청 / 9 hold_cup = 실제 컵을 쥐고 있음 (명령 아님, '마시는 중')
- 저장: 관절 `~/data/assist_arm/gestures/<동작>/*.npz`, 영상 `~/data/assist_arm/clips/<동작>/*.mp4`, DB 에도 자동 기록
- 권장: 동작마다 30~50개+, 각도·거리·왼손/오른손·사람 바꿔가며, idle 은 2배 (컵 옆에 손 두기 등 '아닌' 동작 포함)

## ④ 인식 모델 비교 (tools/)

```bash
TORCH_HOME=~/data/assist_arm/models/rtm python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/live_compare.py [--source realsense] [--ov gpu]
```
- 한 창에 MediaPipe | RTMPose | Hybrid(MediaPipe 위치 + RTMPose 관절)
- 키: `t` 정확도 측정(7구간 안내 → 표 + `~/data/assist_arm/bench/live_compare_<시각>.csv`), `f` RTMPose 오검출 억제,
  `e` 관절 스무딩, `r` 초기화, `s` 스크린샷, `q` 종료. `--video 파일.mp4` 로 녹화 영상도 가능

## ⑤ 학습 (tools/)

```bash
source ~/cobot2_ws/install/setup.bash
# 손 모양 (노트북, 수십 초) → ~/data/assist_arm/models/gesture_static.joblib (intent_node 가 자동 사용)
python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/train_gesture_sklearn.py [--sources realsense webcam]
# 카메라별 교차 평가 (웹캠으로 학습 → RealSense 로 테스트 등)
python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/eval_cross_source.py [--fake]
```
모션(치워처럼 움직이는 동작) — Colab GRU:
1. `python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/db_export_colab.py [--sources realsense] [--labels clear hold_cup]`
   → `~/data/assist_arm/export/assist_arm_<시각>.zip`
2. zip 을 Drive **내 드라이브**에 올리고 Colab 새 셀: `!unzip -q /content/drive/MyDrive/assist_arm_<시각>.zip -d /content/drive/MyDrive/`
3. `MyDrive/assist_arm/colab_gesture_seq_gru.ipynb` 열고 [모두 실행]
4. 나온 `gesture_seq.onnx`, `gesture_seq_labels.json` → 노트북 `~/data/assist_arm/models/` → intent_node 재시작

## ⑤-2 객체인식(YOLO) 학습 — 그릇·컵·물병·수저·젓가락·휴지·행주

클래스 (`config/yolo_classes.yaml`): rice_bowl, side_dish, cup, water_bottle, spoon, chopsticks, tissue, cloth, soup_bowl, plate, fork (v2.2에서 뒤 3개 추가, 재학습 필요)
지금 쓰는 모델: 컨테이너 `assist_detection` 기본 = COCO yolov8n (bowl·cup·bottle·spoon 만, 밥그릇/반찬 구분·젓가락·휴지·행주 없음)
데이터 위치: `~/corecode/yolo_train/assist_table/` (컨테이너에서 `/yolo_train/assist_table` 로 보임)

1. **식탁 사진 찍기** (관찰 자세 그대로, 로봇은 안 움직임; 먼저 `realsense`)
   ```bash
   source /opt/ros/jazzy/setup.bash
   ~/venvs/assist_mp/bin/python ~/cobot2_ws/src/cobot2/assist_arm/tools/capture_table_images.py --source realsense
   ```
   키: space 한 장, a 자동(2초마다), u 삭제, q 종료. 물건 배치·조명·손 가림을 바꿔 가며 200~500장
2. **준비 + 초벌 라벨** (COCO 모델이 박스를 미리 쳐 줌, 컨테이너에 설치 없음)
   ```bash
   python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/yolo_dataset.py prepare
   docker start yolo-detection
   docker exec yolo-detection python3 /ros2_ws/src/assist_yolo/scripts/auto_label.py --dataset /yolo_train/assist_table --weights /yolo_train/Fruits/yolov8n.pt
   python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/yolo_dataset.py status
   ```
3. **사람이 고치기** (설치 없는 웹 도구: Roboflow 또는 CVAT)
   ```bash
   python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/yolo_dataset.py export     # → ~/data/assist_arm/export/yolo_table_<시각>.zip
   ```
   웹 도구에 YOLO 형식으로 올림 → 박스 고치기 (밥그릇/반찬 구분, 젓가락·휴지·행주 새로) → YOLO 형식으로 내보내기 → 받은 zip:
   ```bash
   python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/yolo_dataset.py import ~/Downloads/<내보낸 파일>.zip   # 예전 라벨은 자동 백업
   ```
4. **Colab 학습** (YOLO11n/s, T4/L4): `yolo_dataset.py export --only_labeled` → zip 을 Drive 에 → `tools/colab_yolo_train.ipynb` 실행
5. **새 모델 쓰기**: 받은 `best.pt` → `~/corecode/yolo_train/assist_table/best.pt`, 컨테이너에서
   ```bash
   ros2 run assist_yolo assist_detection --ros-args -p model_path:=/yolo_train/assist_table/best.pt
   ```
   (처음 한 번 또는 assist_yolo 코드가 바뀐 뒤: 컨테이너 안 `cd /ros2_ws && colcon build --packages-select assist_yolo`)
DB `yolo_images` 표에 사진마다 라벨 상태(none/auto/human)·박스 수·분할이 기록돼요.

## 강화학습 / 스스로 학습 현황
- **decision_node (LinUCB)**: 실제로 동작 — 제안에 대한 수락/거절로 사람별로 바로 학습, DB `decision_events` 에 기록
- **그릇 밀기 강화학습**: `tools/colab_push_dish_rl.ipynb` (MuJoCo + PPO, Colab) — 뼈대만, 아직 실행 안 함
- **제스처**: 녹화 → sklearn / Colab GRU (위 ⑤), **객체인식**: 위 ⑤-2

## ⑥ 데이터 DB (sqlite3, `~/data/assist_arm/assist_arm.db`)

메타데이터와 파일 경로만 (파일은 원래 자리). 녹화·제안 반응·음성 요청·학습 모델이 자동 기록.
```bash
python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/db_import_existing.py [--dry]     # 있던 파일 등록 (중복 없음)
python3 ~/cobot2_ws/src/cobot2/assist_arm/tools/db_export_colab.py --sources realsense   # Colab zip
python3 -c "import sqlite3;c=sqlite3.connect('/home/lee/data/assist_arm/assist_arm.db');print(c.execute('select label,count(*) from gesture_samples group by label').fetchall())"
```
끄기: `live_record.py --no_db`, 노드 `-p use_db:=false`.

## ⑦ 설정 파일 (`config/`, 고친 뒤 ① 빌드)

| 파일 | 고치는 것 |
|---|---|
| `assist_arm.yaml` | 노드별 설정 (웹캠 번호, 손 인식 신뢰도, 제스처 모델 경로, 콕 시간, 일시정지 20cm, 속도·힘, 컵 거리 25cm, 판단 방식) |
| `waypoints.yaml` | 로봇 자세 좌표 (직접교시): 홈, **관찰 자세 table_view_j**, 물병·따르기, 걸레, 율동, 제안 모션 |
| `table_layout.yaml` | 테이블 높이, 그릇·수저 집는 높이, 거치대, 자리 세팅 배치, 음식 목록, 정리 위치 |
| `intents.yaml` | 애매한 말 예시 ("배불러" → 전체 정리 …) |
| `music_library.yaml` | 음악 파일·bpm·분위기 |

---

## 이번 재구성에서 지운 것 / 합친 것 (백업: `~/data/assist_arm_backup_20261126_1425_v10`)

| 예전 | 지금 |
|---|---|
| `rs_hand_node`, `webcam_publisher`, `pose_node` | `hand_perception` (perception/rs_hand.py, webcam.py, pose.py) |
| `spill_node` | `object_perception` (perception/spill.py) |
| `gesture_node`, `finger_pointer`, `request_router`, `claude_planner_node`, `voice_command_node` | `intent_node` (intent/*.py) — 노드 이름·토픽은 그대로 |
| `task_manager`, `subtasks.py`, `robot_io.py` | execution/task_manager.py, execution/skills.py, execution/robot_io.py |
| `decision_node`, `music_node`, `safety_monitor` | execution/, safety/ 로 이동 (그대로) |
| `calibrate_pointer` (ros2 run) | `tools/calibrate_pointer.py` (python3 로 실행) |
| `collect_gestures` (ROS 녹화기) | **삭제** → `tools/live_record.py` |
| `benchmark/record_clips.py`, `bench_hand_pose.py` | **삭제** → `tools/live_compare.py` (`--video` 로 녹화 영상도) |
| `training/prepare_colab_upload.sh` | **삭제** → `tools/db_export_colab.py` |
| `benchmark/`, `training/` 폴더 | `tools/` 하나로 합침 |
| launch `assist_all`, `vision`, `table`, `water`, `music`, `wipe` | `bringup_all` (`tasks:=…`), `perception`, `dry_run` |
| 메시지 | 바꾼 것 없음 (모두 사용 중 — 기존 데이터·모델과 호환) |
