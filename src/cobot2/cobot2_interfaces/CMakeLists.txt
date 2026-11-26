cmake_minimum_required(VERSION 3.8)
project(cobot2_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/ObjectTypes.msg"
  "msg/UserCommand.msg"
  "msg/SeatStatus.msg"
  "msg/CupLevel.msg"
  "msg/RobotState.msg"
  "msg/TaskLog.msg"
  "msg/DetectedObject.msg"
  "srv/GetScene.srv"
  "action/TableTask.action"
  "action/Pour.action"
  "action/Skill.action"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
