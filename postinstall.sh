#!/bin/bash
echo "libtool running postinstall.sh"
install_name_tool -change @rpath/libcurand.10.dylib /usr/local/cuda/lib/libcurand.10.dylib /usr/local/lib/libneurongpu.so
