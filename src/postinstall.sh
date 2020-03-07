#!/bin/bash
echo "libtool running postinstall.sh"
install_name_tool -change @rpath/usr/local/lib/ /usr/local/lib/libcurand.10.dylib /usr/local/lib/libneurongpu.so
