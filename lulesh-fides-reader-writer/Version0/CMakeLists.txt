cmake_minimum_required(VERSION 3.10)
add_compile_options(-Wno-narrowing)

project(LULESH LANGUAGES C CXX)

option(WITH_MPI    "Build LULESH with MPI"          TRUE)
option(WITH_OPENMP "Build LULESH with OpenMP"       FALSE)
option(WITH_SILO   "Build LULESH with silo support" FALSE)
option(WITH_ADIOS  "Build LULESH with ADIOS support" TRUE)

if (WITH_MPI)
  find_package(MPI REQUIRED COMPONENTS C CXX)
endif()

if (WITH_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

if (WITH_ADIOS)
  find_package(ADIOS2 COMPONENTS CXX MPI REQUIRED)
  find_package(catalyst REQUIRED)
endif()

if (WITH_SILO)
  find_path(SILO_INCLUDE_DIR silo.h
    HINTS ${SILO_DIR}/include)
  find_library(SILO_LIBRARY
    NAMES siloh5
    HINTS ${SILO_DIR}/lib)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SILO DEFAULT_MSG
    SILO_LIBRARY
    SILO_INCLUDE_DIR)

  if (SILO_FOUND)
    add_definitions("-DVIZ_MESH")
    include_directories(${SILO_INCLUDE_DIR})
    #   Note: silo needs to be built as a dynamic lib, otherwise
    # there are additional dependencies (hdf5) which we don't know.
    # This would be fixed by silo providing a CMake package.
    list(APPEND LULESH_EXTERNAL_LIBS ${SILO_LIBRARY})
  endif()
endif()

set(LULESH_SOURCES
  lulesh-comm.cc
  lulesh-init.cc
  lulesh-util.cc
  lulesh-viz.cc
  writer.cpp
  CatalystAdaptor.h
  lulesh.cc)

set(LULESH_EXEC lulesh2.0)

add_executable(${LULESH_EXEC} ${LULESH_SOURCES})
if(WITH_MPI)
 target_link_libraries(${LULESH_EXEC}
   PRIVATE
     MPI::MPI_C MPI::MPI_CXX)

 target_compile_definitions(${LULESH_EXEC}
   PRIVATE
    "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
     USE_MPI=1)
else()
 target_compile_definitions(${LULESH_EXEC}
   PRIVATE
     USE_MPI=0)
endif()

if(WITH_OPENMP)
 target_link_libraries(${LULESH_EXEC}
   PRIVATE
     OpenMP::OpenMP_CXX)
endif()

if (WITH_ADIOS)
  target_link_libraries(${LULESH_EXEC}
   PRIVATE
     adios2::cxx_mpi)
  target_link_libraries(${LULESH_EXEC} PRIVATE catalyst::catalyst)
endif()