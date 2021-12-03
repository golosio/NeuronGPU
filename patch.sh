echo "Makefile patch for renaming libnestgpu.la to libnestgpu.so"
cat Makefile | sed 's/libnestgpu.la/libnestgpu.so/'> tmpfile
mv tmpfile Makefile
