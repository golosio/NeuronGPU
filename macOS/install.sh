#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "No source directory supplied"
    exit
fi
srcdir=$1/NESTGPU

if [ ! -f /usr/local/cuda/lib/libcurand.10.dylib ]; then
    echo "File not found /usr/local/cuda/libcurand.10.dylib"
    echo "It seems that CUDA toolkit is not properly installed"
    echo "Please install it following the instructions in:"
    echo "https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html"
    exit 1
fi

# create installation directory if it doesn't exist and clean it
mkdir -p "/usr/local/nestgpu"
rm -fr /usr/local/nestgpu/*
mkdir -p "/usr/local/lib"

# copy subdirectories
cp -r $srcdir/src /usr/local/nestgpu
cp -r $srcdir/python /usr/local/nestgpu
cp -r $srcdir/c++ /usr/local/nestgpu
cp -r $srcdir/macOS/pythonlib /usr/local/nestgpu
cp -r $srcdir/macOS/lib /usr/local/nestgpu

#create include directory and copy header file
mkdir /usr/local/nestgpu/include
cp $srcdir/src/nestgpu.h /usr/local/nestgpu/include/

# find python package directory
SITEDIR=$(python -m site --user-site)
SITEDIR3=$(python3 -m site --user-site)

# create if it doesn't exist
mkdir -p "$SITEDIR"
mkdir -p "$SITEDIR3"

# create new .pth file with path to nestgpu python module
echo "/usr/local/nestgpu/pythonlib/" > "$SITEDIR/nestgpu.pth"
echo "/usr/local/nestgpu/pythonlib/" > "$SITEDIR3/nestgpu.pth"

# create a symbolic link in /usr/local/lib to the dynamic-link library
ln -s /usr/local/nestgpu/lib/libnestgpu.so /usr/local/lib/libnestgpu.so
