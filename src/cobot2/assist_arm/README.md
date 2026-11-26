# assist_arm — 사람 손과 상호작용하는 가정용 로봇 (무접촉)

**원칙**: 로봇은 사람과 접촉하지 않는다. 사람 쪽 물건은 사용자 앞이나 손 옆 테이블에 두고 후퇴.
손·팔이 로봇 작업 지점 20cm 안이면 일시정지, 벗어나면 재개. 손바닥/“멈춰”는 즉시 정지.

**모든 실행 명령·녹화·학습·DB 사용법은 [COMMANDS.md](COMMANDS.md)** 에 있어요.

## 구조 (4층)

```
[인식]   rs_perception (손목 RealSense: 손 21점 3D, 얼룩)      webcam_watch (웹캠: 안전 감시용 손·상체)
            │ /rs/landmarks /rs/hand_table                         │ /user/landmarks
            ▼                                                       ▼
[해석]   intent_node (gesture · pointer(콕) · router · planner(로컬/Claude) · voice) ──▶ /assist/request
            ▼
[결정·실행] task_manager (상태 머신 + skills, 작업 뒤 관찰 자세 → /table/snapshot)   decision_node(선택)   music_node(선택)
            │ DSR(amovel…), RG2                    YOLO: 컨테이너 assist_detection (/assist/get_3d_position)
[안전]   safety_monitor ── /safety/stop, /safety/pause → 두산 MoveStop / MovePause / MoveResume
```

흐름: 작업 끝 → 관찰 자세(테이블 위 50~70cm) → 식탁 스냅샷(그릇·컵 3D, 빈 그릇) → RealSense 로 손동작 대기
→ 요청(치워 / 콕 / 식사 끝 / 물 줘 + 컵 잡기 / N명이야 + 콕 …) → 작업 → 다시 관찰 자세.
로봇이 움직이는 동안 웹캠은 '멈춰'만 받는다.

## 기능

| 우선순위 | 작업 (`tasks:=`) | 요청 |
|---|---|---|
| P0 | clear (clear_dish / clear_empty / clear_all) | 콕+치워, 손등 밀기, 엄지척·"다 먹었어" |
| P0 | bring (bring_dish) | 콕(3초 대기), "김치 줘" |
| P1 | water (pour_water) | "물 줘" + 빈손 컵 시늉(cup_grab) → 손 옆 25cm 에 둠 |
| P1 | seat (seat_setting) | "N명이야" + 자리 콕 N번 |
| P2 | decision_node | 상황 보고 먼저 제안 → 엄지척/손등 밀기로 LinUCB 학습 |
| P2 | music / wipe | 음악·율동 / 걸레로 테이블 얼룩 닦기 |

## 폴더

```
assist_arm/
  perception/  rs_perception.py (rs_hand, spill), webcam_watch.py (webcam, pose)
  intent/      intent_node.py (gesture, pointer, router, planner, voice)
  execution/   task_manager.py, skills.py, robot_io.py, decision_node.py, music_node.py
  safety/      safety_monitor.py
  common/      features(손 특징 v2), mp_loader, rs_camera, geometry, dsr_services, datadb, local_intent, multi, heartbeat
tools/         live_record, live_compare, train_gesture_sklearn, eval_cross_source, gesture_dataset,
               db_import_existing, db_export_colab, calibrate_pointer, Colab 노트북들, train_spill_classifier
config/        assist_arm.yaml, waypoints.yaml, table_layout.yaml, intents.yaml, music_library.yaml
launch/        bringup_all, perception, dry_run
```

## 재사용한 기존 인터페이스 (기존 패키지는 수정하지 않음)

- 로봇: robot_control 과 같은 `DR_init` + `DSR_ROBOT2` 방식, `robot_control.onrobot.RG` 그리퍼, `T_gripper2camera.npy`
- RealSense 토픽·수식: object_detection 의 ImgNode (`/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`)
- YOLO: `od_msg/SrvDepthPosition` + object_detection 코드 상속 (`yolo_container/assist_yolo`)
- 음성: voice_processing 의 WakeupWord + STT

## 아직 임시

waypoints·table_layout 좌표 전부, 빈 그릇 판단 규칙, 얼룩 분류(밝기 규칙), 접시·젓가락 YOLO 클래스,
학습된 제스처 모델(녹화 후 학습 필요), 율동·따르기 궤적, TTS.
