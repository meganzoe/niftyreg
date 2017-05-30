#-----------------------------------------------------------------------------
# Eigen version 3.3.3 - 67e894c6cd8f
if(NOT EXISTS ${CMAKE_BINARY_DIR}/third-party/eigen3)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar x ${CMAKE_SOURCE_DIR}/third-party/eigen_3.3.3.tar.gz
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/third-party
  )
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E rename
    ${CMAKE_BINARY_DIR}/third-party/eigen-eigen-67e894c6cd8f
    ${CMAKE_BINARY_DIR}/third-party/eigen3
  )
  message(STATUS "Eigen3 files are copied in ${CMAKE_BINARY_DIR}/third-party/eigen3")
endif(NOT EXISTS ${CMAKE_BINARY_DIR}/third-party/eigen3)
#-----------------------------------------------------------------------------
if(USE_CERES)
  message(STATUS "Ceres-solver will be downloaded and built")
  include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)
  ExternalProject_Add(Ceres
    GIT_REPOSITORY "https://github.com/ceres-solver/ceres-solver"
    GIT_TAG "1.12.0"
    UPDATE_COMMAND ""
    BUILD_COMMAND ""
    PREFIX "${CMAKE_BINARY_DIR}/third-party/Ceres"
    SOURCE_DIR "${CMAKE_BINARY_DIR}/third-party/Ceres/source"
    BINARY_DIR "${CMAKE_BINARY_DIR}/third-party/Ceres/build"
    INSTALL_DIR "${CMAKE_BINARY_DIR}/third-party/Ceres/install"
    STAMP_DIR "${CMAKE_BINARY_DIR}/third-party/Ceres/stamp"
    TMP_DIR "${CMAKE_BINARY_DIR}/third-party/Ceres/tmp"
    CMAKE_ARGS -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    CMAKE_ARGS -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}/third-party/Ceres/install
    CMAKE_ARGS -DOPENMP=${USE_OPENMP}
    CMAKE_ARGS -DEIGEN_INCLUDE_DIR=${CMAKE_BINARY_DIR}/third-party/eigen3/
    CMAKE_ARGS -DBUILD_EXAMPLES=OFF
    CMAKE_ARGS -DBUILD_TESTING=OFF
    CMAKE_ARGS -DEIGENSPARSE=OFF
    CMAKE_ARGS -DSCHUR_SPECIALIZATIONS=OFF
    CMAKE_ARGS -DGFLAGS=OFF
    CMAKE_ARGS -DMINIGLOG=ON
  )
endif(USE_CERES)
#-----------------------------------------------------------------------------
