SET(PACKAGE_NAME DLRTPICKER)

# Off by default, same convention as the core iLoc locator plugin
# (SC_TRUNK_LOCATOR_ILOC). When on, ONNX Runtime must be present or
# configuration fails outright -- no silent fallback.
OPTION(SC_DLRTPICKER "Build the ONNX Runtime based deep-learning picker plugin (DL1C/DL3C/SDL1C/SDL3C); requires ONNX Runtime" OFF)

IF(SC_DLRTPICKER)
	FIND_PACKAGE(ONNXRuntime REQUIRED MODULE)
	INCLUDE_DIRECTORIES(${ONNXRuntime_INCLUDE_DIR})

	SET(${PACKAGE_NAME}_SOURCES
		plugin.cpp
		onnxmodel.cpp
		dl.cpp
		S_dl.cpp
	)

	# Global plugin (third arg ""), installed to share/plugins/ like
	# lociloc.so -- enabled per application via "plugins = ..., dlrtpicker"
	# in that application's .cfg, then "picker = DL3C" / "spicker = SDL3C".
	SC_ADD_PLUGIN_LIBRARY(${PACKAGE_NAME} dlrtpicker "")
	SC_LINK_LIBRARIES_INTERNAL(dlrtpicker client core)
	SC_LINK_LIBRARIES(dlrtpicker ${ONNXRuntime_LIBRARIES})

	# Registers picker.DL1C.*/picker.DL3C.*/spicker.SDL1C.*/spicker.SDL3C.*
	# as scconfig-visible global bindings, the same way AIC/BK (global.xml)
	# and DFX (processing/fx/descriptions/global_dfx.xml) do -- without
	# this, the plugin works but "doesn't have bindings like the other
	# pickers in seiscomp".
	FILE(GLOB descs "${CMAKE_CURRENT_SOURCE_DIR}/descriptions/*.xml")
	INSTALL(FILES ${descs} DESTINATION ${SC3_PACKAGE_APP_DESC_DIR})

	# No RPATH is set here, consistent with every other plugin in this
	# tree (lociloc/LAPACK, dbmysql/dbpostgresql): libonnxruntime.so.1
	# must be resolvable through the normal dynamic linker search path
	# (a standard lib directory, /etc/ld.so.conf.d/ + ldconfig, or
	# LD_LIBRARY_PATH) on any host that runs an application with
	# "plugins = ..., dlrtpicker" enabled -- only those processes need it.
ENDIF()
