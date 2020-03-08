echo "Makefile patch for renaming libneurongpu.la to libneurongpu.so"
cat Makefile | sed 's/libneurongpu.la/libneurongpu.so/'> tmpfile
mv tmpfile Makefile
