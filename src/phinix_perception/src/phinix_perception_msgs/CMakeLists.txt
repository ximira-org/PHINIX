# All rights reserved.
cmake_minimum_required(VERSION 3.5)

### Export headers
project(phinix_perception_msgs)
# Default to C99

if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()


# Generate messages
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BBoxMsg.idl"
  DEPENDENCIES
    "builtin_interfaces"
    "std_msgs"
    "geometry_msgs"
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()