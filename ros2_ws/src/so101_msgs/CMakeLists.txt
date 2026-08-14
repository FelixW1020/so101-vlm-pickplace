cmake_minimum_required(VERSION 3.8)
project(so101_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GroundTarget.srv"
  "action/PickAndPlace.action"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_package()
