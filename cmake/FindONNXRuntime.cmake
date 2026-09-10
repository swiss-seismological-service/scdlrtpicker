# - Find ONNX Runtime
# Find the ONNX Runtime C++ API includes and library.
# This module defines
#  ONNXRuntime_INCLUDE_DIR, where to find onnxruntime_cxx_api.h
#  ONNXRuntime_LIBRARIES, the libraries needed to use ONNX Runtime
#  ONNXRuntime_FOUND, if false, ONNX Runtime support is disabled
#
# The search can be guided with ONNXRUNTIME_ROOT, either as a CMake
# variable or environment variable, pointing at an ONNX Runtime
# install prefix (containing include/ and lib/), e.g. an extracted
# onnxruntime-linux-x64-<version>.tgz release.
set(ONNXRUNTIME_ROOT "" CACHE PATH
	"Root directory of an ONNX Runtime install (contains include/ and lib/), e.g. an extracted onnxruntime-linux-x64-<version>.tgz. Only needed if ONNX Runtime is not found automatically.")

# find_path()/find_library() below cache ONNXRuntime_INCLUDE_DIR/
# ONNXRuntime_LIBRARIES once found, and -- like any cached variable --
# silently keep that value on a later cmake run even if ONNXRUNTIME_ROOT
# is repointed at a different install: the "already found" check right
# after this short-circuits before the search ever re-runs. Re-pointing
# ONNXRUNTIME_ROOT would then still link/compile against the old
# install, with no warning, only a build error deep in a consumer (e.g.
# onnxmodel.cpp failing to find onnxruntime_cxx_api.h once the old
# path is removed). Detect a changed ONNXRUNTIME_ROOT here and force a
# fresh search.
if(NOT "${ONNXRUNTIME_ROOT}" STREQUAL "${_ONNXRuntime_LAST_ROOT}")
	unset(ONNXRuntime_INCLUDE_DIR CACHE)
	unset(ONNXRuntime_LIBRARIES CACHE)
endif()
set(_ONNXRuntime_LAST_ROOT "${ONNXRUNTIME_ROOT}" CACHE INTERNAL "")

if(ONNXRuntime_INCLUDE_DIR AND ONNXRuntime_LIBRARIES)
	set(ONNXRuntime_FOUND TRUE)

else()
	find_path(ONNXRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
		HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
		PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
	)

	find_library(ONNXRuntime_LIBRARIES NAMES onnxruntime
		HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
		PATH_SUFFIXES lib lib64
	)

	if(ONNXRuntime_INCLUDE_DIR AND ONNXRuntime_LIBRARIES)
		set(ONNXRuntime_FOUND TRUE)
		if(NOT ONNXRuntime_FIND_QUIETLY)
			message(STATUS "Found ONNX Runtime: ${ONNXRuntime_INCLUDE_DIR}, ${ONNXRuntime_LIBRARIES}")
		endif()
	else()
		set(ONNXRuntime_FOUND FALSE)
		if(ONNXRuntime_FIND_REQUIRED)
			message(FATAL_ERROR "ONNX Runtime not found.")
		else()
			if(NOT ONNXRuntime_FIND_QUIETLY)
				message(STATUS "ONNX Runtime not found, deep-learning pickers (DL1C/DL3C/SDL1C/SDL3C) "
				               "will be disabled. Set ONNXRUNTIME_ROOT to enable.")
			endif()
		endif()
	endif()

	mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARIES)
endif()
