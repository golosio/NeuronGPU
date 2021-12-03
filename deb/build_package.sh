#!/bin/bash
if [ $# -ne 1 ]; then
   echo "Usage: $0 distro"
   echo "where distro is ubuntu1 for bionic(18.04), ubuntu2 for eoan(19.10), debian1, ..."
   exit
fi

version=1.4.6~$1

#define source and target directories
srcdir=..
tgdir=nestgpu_$version

# create installation directory if it doesn't exist and clean it
mkdir -p $tgdir/usr/local/nestgpu
rm -fr $tgdir/usr/local/nestgpu/*
mkdir -p $tgdir/usr/local/lib

# copy subdirectories
cp -r $srcdir/src $tgdir/usr/local/nestgpu
cp -r $srcdir/python $tgdir/usr/local/nestgpu
cp -r $srcdir/c++ $tgdir/usr/local/nestgpu
cp -r $srcdir/deb/lib $tgdir/usr/local/nestgpu

#create include directory and copy header file
mkdir $tgdir/usr/local/nestgpu/include
cp $srcdir/src/nestgpu.h $tgdir/usr/local/nestgpu/include/

# create python package directory
mkdir -p $tgdir/usr/lib/python2.7/dist-packages/
mkdir -p $tgdir/usr/lib/python3/dist-packages/

# copy the nestgpu python module
cp $srcdir/pythonlib/nestgpu.py $tgdir/usr/lib/python2.7/dist-packages/
cp $srcdir/pythonlib/nestgpu.py $tgdir/usr/lib/python3/dist-packages/

# create a symbolic link in /usr/local/lib to the dynamic-link library
ln -s /usr/local/nestgpu/lib/libnestgpu.so $tgdir/usr/local/lib/libnestgpu.so

# create dependency list
depends=$(./depends.sh)

# create metadata file and control file
mkdir $tgdir/DEBIAN
cat control.templ | sed "s/__version__/$version/;s/__depends__/$depends/" > $tgdir/DEBIAN/control
dpkg-deb --build $tgdir
