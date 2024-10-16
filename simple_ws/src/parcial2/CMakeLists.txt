cmake_minimum_required(VERSION 3.8)
project(parcial2)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)

install(
  DIRECTORY
  src
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  src/network.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
