#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "No source directory supplied"
fi
srcdir=$1/NeuronGPU

if [ ! -f /usr/local/cuda/lib/libcurand.10.dylib ]; then
    echo "File not found /usr/local/cuda/libcurand.10.dylib"
    echo "It seems that CUDA toolkit is not properly installed"
    echo "Please install it following the instructions in:"
    echo "https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html"
    exit 1
fi

# create installation directory if it doesn't exist and clean it
mkdir -p "/usr/local/neurongpu"
rm -fr /usr/local/neurongpu/*

# copy subdirectories
echo $srcdir
ls $srcdir
cp -r $srcdir/src /usr/local/neurongpu
cp -r $srcdir/python /usr/local/neurongpu
cp -r $srcdir/c++ /usr/local/neurongpu
cp -r $srcdir/macOS/pythonlib /usr/local/neurongpu
cp -r $srcdir/macOS/lib /usr/local/neurongpu

#create include directory and copy header file
mkdir /usr/local/neurongpu/include
cp $srcdir/src/neurongpu.h /usr/local/neurongpu/include/

# find python package directory
SITEDIR=$(python -m site --user-site)

# create if it doesn't exist
mkdir -p "$SITEDIR"

# create new .pth file with path to neurongpu python module
echo "/usr/local/NeuronGPU/pythonlib/" > "$SITEDIR/neurongpu.pth"

# create a symbolic link in /usr/local/lib to the dynamic-link library
ln -s /usr/local/neurongpu/lib/libneurongpu.so /usr/local/lib/libneurongpu.so
