SC_BEGIN_PACKAGE(SCDLRTPICKER)

# Lets plugins/picker/dl reach the in-tree (not yet installed) SeisComP
# core headers, same pattern as src/extras/scrtdd/CMakeLists.txt.
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../../base/common/libs)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/../../base/common/libs)

# Ships its own FindONNXRuntime.cmake (plugins/picker/dl's dependency),
# same pattern as src/extras/SED-EEW-SeisComP-contributions/CMakeLists.txt.
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

SUBDIRS(plugins)
