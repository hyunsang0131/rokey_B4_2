# cobot2_interfaces — 팀 공용 인터페이스 (설계 v1.1)

공통 규칙: 좌표 `base_link` 기준, 단위 **m·rad** (두산 mm·deg, 그리퍼 1/10mm 변환은 `skill_server` 안에서만),
우리 네임스페이스 `/cobot2/...`. 물체 종류 문자열은 `ObjectTypes` 상수 (YOLO 클래스 이름과 동일):
`rice_bowl`, `soup_bowl`, `main_plate`, `spoon`, `fork`, `cup`, `bottle`.

빌드: `cd ~/cobot2_ws && colcon build --packages-select cobot2_interfaces && source install/setup.bash`

| 종류 | 이름 | 타입 | 보내는 → 받는 | 주기·특징 |
|---|---|---|---|---|
| 토픽 | `/cobot2/user/command` | `msg/UserCommand` | voice_node, gesture_node → task_manager | 이벤트 |
| 토픽 | `/cobot2/hri/seat_status` | `msg/SeatStatus` | gesture_node → planner_node | 바뀔 때 + 1 Hz |
| 토픽 | `/cobot2/pour/level` | `msg/CupLevel` | pour_level_node → skill_server(pour) | 10 Hz 이상 |
| 토픽 | `/cobot2/safety/stop` | `std_msgs/Bool` | voice, gesture, task_manager → skill_server | true 정지 / false 재개 허용 |
| 토픽 | `/cobot2/robot/state` | `msg/RobotState` | skill_server → 모든 노드 | 5 Hz |
| 토픽 | `/cobot2/log/task` | `msg/TaskLog` | planner_node → db_logger_node | 동작마다 |
| 서비스 | `/cobot2/perception/get_scene` | `srv/GetScene` (+ `msg/DetectedObject`) | planner ↔ perception_node | 관측 자세 정지 + 빈손일 때만, 1 s 이하 |
| 액션 | `/cobot2/table_task` | `action/TableTask` | task_manager → planner_node | 취소 시 현재 동작 마치고 멈춤 |
| 액션 | `/cobot2/pour` | `action/Pour` | task_manager → skill_server | 수위 폐루프, Skill 과 잠금 공유 |
| 액션 | `/cobot2/skill` | `action/Skill` | planner_node → skill_server | 한 동작 단위, Pour 와 잠금 공유 |

상수 (문자열/숫자로 정의, 코드에서 `UserCommand.CMD_POUR` 처럼 사용)
- `UserCommand`: `SOURCE_VOICE/GESTURE`, `CMD_SET_TABLE/CLEAR_TABLE/POUR/STOP/RESUME/CANCEL`
- `SeatStatus`: `EMPTY=0, EATING=1, FINISHED=2`
- `RobotState`: `IDLE=0, MOVING=1, STOPPED=2, ERROR=3`
- `TableTask`: `TASK_SET_TABLE/CLEAR_TABLE`, `PLANNER_RL/RULE`
- `Skill`: `SKILL_OBSERVE/PICK/PLACE/HOME/OPEN/CLOSE`, `PLACE_TABLE/STACK/UTENSIL_BIN/TEMP`, (목표 쪽 `Skill.Goal.SKILL_*`, `Skill.Goal.PLACE_*`), 결과 쪽 `Skill.Result.ERR_OK/BUSY/STOPPED/FAILED …`

## 문서(v1.1)에 없는데 추가한 것 — 팀 확인 필요
- `msg/ObjectTypes` : 물체 종류 문자열 상수 모음 (토픽으로 보내지 않음, 오타 방지용)
- `TaskLog.header`, 그리고 출발·도착을 `from_pose/to_pose`(Pose) **와** `from_place/to_place`(장소 종류 문자열) 둘 다
  (문서에 "출발·도착"만 있어서 위치와 장소 종류를 모두 담음)
- `Skill` 오류 코드 `ERR_NO_OBJECT`(집기 실패), `ERR_UNREACHABLE`(도달 불가), `ERR_INVALID_GOAL`(잘못된 목표)
- 피드백 `stage` 의 예시 값, `TaskLog.duration` 단위 s, `RobotState.current_skill` 은 Skill 상수 문자열
