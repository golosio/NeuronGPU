file_list=$(echo $(ls src/*.cu src/*.cpp | grep -v dummyfile.cpp | xargs -n 1 basename))
escaped_rhs=$(echo "$file_list" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
cat src/Makefile.in | sed 's/$(AM_V_CXXLD)$(/$(NVCC) -ccbin=mpic++ --compiler-options  "${COMPILER_FLAGS}" ${CUDA_FLAGS} ${CUDA_LDFLAGS} -o libneurongpu.so $(all_SOURCES) $(CUDA_LIBS)\n# $(AM_V_CXXLD)$(/' | sed "/libneurongpu_la_SOURCES =/ a all_SOURCES = $escaped_rhs" | sed 's/libneurongpu.la/libneurongpu.so/'> tmpfile
mv tmpfile src/Makefile.in
