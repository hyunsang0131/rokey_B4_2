cmake_minimum_required(VERSION 3.8)
project(assist_arm_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/UserLandmarks.msg"
  "msg/Gesture.msg"
  "msg/SpillResult.msg"
  "msg/SafetyStatus.msg"
  "msg/TaskRequest.msg"
  "msg/HandTable.msg"
  "srv/RegisterGesture.srv"
  "srv/SetSafetyMode.srv"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
