version=1.0.8-1

#define source and target directories
srcdir=..
tgdir=neurongpu_$version

# create installation directory if it doesn't exist and clean it
mkdir -p $tgdir/usr/local/neurongpu
rm -fr $tgdir/usr/local/neurongpu/*
mkdir -p $tgdir/usr/local/lib

# copy subdirectories
cp -r $srcdir/src $tgdir/usr/local/neurongpu
cp -r $srcdir/python $tgdir/usr/local/neurongpu
cp -r $srcdir/c++ $tgdir/usr/local/neurongpu
cp -r $srcdir/deb/lib $tgdir/usr/local/neurongpu

#create include directory and copy header file
mkdir $tgdir/usr/local/neurongpu/include
cp $srcdir/src/neurongpu.h $tgdir/usr/local/neurongpu/include/

# create python package directory
mkdir -p $tgdir/usr/lib/python2.7/dist-packages/

# copy the neurongpu python module
cp $srcdir/pythonlib/neurongpu.py $tgdir/usr/lib/python2.7/dist-packages/

# create a symbolic link in /usr/local/lib to the dynamic-link library
ln -s /usr/local/neurongpu/lib/libneurongpu.so $tgdir/usr/local/lib/libneurongpu.so

# create dependency list
depends=$(./depends.sh)

# create metadata file and control file
mkdir $tgdir/DEBIAN
cat control.templ | sed "s/__version__/$version/;s/__depends__/$depends/" > $tgdir/DEBIAN/control
dpkg-deb --build $tgdir
