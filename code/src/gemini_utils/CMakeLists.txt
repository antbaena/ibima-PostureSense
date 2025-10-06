cmake_minimum_required(VERSION 3.5)
project(gemini_utils)

# Habilita advertencias de compilador para GNU/Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Encuentra dependencias
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(uncompress_node src/uncompress_node.cpp)
add_executable(depth_decompressor_node src/uncompress_depth_node.cpp)

target_include_directories(uncompress_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(uncompress_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
)

target_link_libraries(uncompress_node
  cv_bridge::cv_bridge
  ${OpenCV_LIBRARIES}
)

# Instalación del ejecutable y del launch
install(
  TARGETS uncompress_node
  DESTINATION lib/${PROJECT_NAME}
)


target_include_directories(depth_decompressor_node
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(depth_decompressor_node
  rclcpp
  sensor_msgs
  cv_bridge
  image_transport
  OpenCV
)

target_link_libraries(depth_decompressor_node
  cv_bridge::cv_bridge
  ${OpenCV_LIBRARIES}
)

# Instalación del ejecutable y del launch
install(
  TARGETS depth_decompressor_node
  DESTINATION lib/${PROJECT_NAME}
)



install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/launch
)

# Genera los artefactos de paquete
ament_package()
