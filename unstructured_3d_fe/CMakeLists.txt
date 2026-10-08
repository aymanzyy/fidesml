cmake_minimum_required(VERSION 3.13)
project(UnstructuredGridML LANGUAGES C CXX)

include (GNUInstallDirs)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}")

#------------------------------------------------------------------------------
# since we use C++11 in this example.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Since this example uses MPI, find and link against it.
find_package(MPI COMPONENTS C CXX)
if (NOT MPI_FOUND)
  message(STATUS
    "Skipping example: ${PROJECT_NAME} requires MPI.")
  return ()
endif ()

#------------------------------------------------------------------------------
add_executable(UnstructuredGridML
  FEDataStructures.cxx
  FEDataStructures.h
  FEDriver.cxx)
target_link_libraries(UnstructuredGridML
  PRIVATE
    MPI::MPI_C
    MPI::MPI_CXX)

#------------------------------------------------------------------------------
option(USE_CATALYST "Build example with Catalyst enabled" ON)
if (USE_CATALYST)
  find_package(catalyst REQUIRED)
  find_path(ParaView_CATALYST_DIR lib*-catalyst.* REQUIRED DOC "Location to ParaViewCatalyst Library")

  target_compile_definitions(UnstructuredGridML
    PRIVATE
      "PARAVIEW_IMPL_DIR=\"${ParaView_CATALYST_DIR}\""
      USE_CATALYST=1)
  target_link_libraries(UnstructuredGridML
    PRIVATE
      catalyst::catalyst)

  include(CTest)
  if (BUILD_TESTING)
    add_test(
      NAME UnstructuredGridML::SimplePipeline
      COMMAND UnstructuredGridML
              ${CMAKE_CURRENT_SOURCE_DIR}/catalyst_pipeline.py)

    add_test(
      NAME UnstructuredGridML::IOPipeline::Prepare
      COMMAND "${CMAKE_COMMAND}" -E remove_directory  "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/result")
    set_tests_properties(
      "UnstructuredGridML::IOPipeline::Prepare"
      PROPERTIES
        FIXTURES_SETUP "UnstructuredGridML::IOPipeline::Prepare")

    add_test(
      NAME UnstructuredGridML::IOPipeline
      COMMAND UnstructuredGridML
              --output "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/result/dataset-%04ts.vtpd"
              --exists "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/result/dataset-0093.vtpd"
              --exists "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/result/dataset-0093/dataset-0093_0.vti")
    set_tests_properties(
      "UnstructuredGridML::IOPipeline"
      PROPERTIES
        FIXTURES_REQUIRED "UnstructuredGridML::IOPipeline::Prepare")

    add_test(
      NAME UnstructuredGridML::IOPipelineRelativePaths::Prepare
      COMMAND "${CMAKE_COMMAND}" -E rm -rf  "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/relative_paths/*")

    set_tests_properties(
      "UnstructuredGridML::IOPipelineRelativePaths::Prepare"
      PROPERTIES
        FIXTURES_SETUP "UnstructuredGridML::IOPipelineRelativePaths::Prepare")

    make_directory("${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/relative_paths")
    add_test(
      NAME UnstructuredGridML::IOPipelineRelativePaths
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/relative_paths"
      COMMAND UnstructuredGridML
              --output "dataset-%04ts.vtpd"
              --exists "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/relative_paths/dataset-0093.vtpd"
              --exists "${CMAKE_CURRENT_BINARY_DIR}/Testing/Temporary/relative_paths/dataset-0093/dataset-0093_0.vti")

    set_tests_properties(
      "UnstructuredGridML::IOPipelineRelativePaths"
      PROPERTIES
        FIXTURES_REQUIRED "UnstructuredGridML::IOPipelineRelativePaths::Prepare")

    set(_vtk_fail_regex
      # vtkLogger
      "(\n|^)ERROR: "
      "ERR\\|"
      # Python errors / exceptions
      "Error"
      # vtkDebugLeaks
      "instance(s)? still around")

    set_tests_properties(
        "UnstructuredGridML::SimplePipeline"
        "UnstructuredGridML::IOPipeline"
        "UnstructuredGridML::IOPipelineRelativePaths"
      PROPERTIES
        FAIL_REGULAR_EXPRESSION "${_vtk_fail_regex}"
        SKIP_RETURN_CODE 125)

    # this ensures that ParaView Catalyst was indeed loaded.
    set_tests_properties(
        "UnstructuredGridML::SimplePipeline"
      PROPERTIES
        PASS_REGULAR_EXPRESSION "executing catalyst_pipeline"
        SKIP_REGULAR_EXPRESSION "Python support not enabled")
  endif()
endif()
