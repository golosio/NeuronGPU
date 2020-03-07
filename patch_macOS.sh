cd src
echo "Patch to Makefile.in link command for compiling and linking in one step"
file_list=$(echo $(ls ../$ac_top_srcdir/src/*.cu ../$ac_top_srcdir/src/*.cpp | grep -v dummyfile.cpp)) #| xargs -n 1 basename))
escaped_rhs=$(echo "$file_list" | sed 's:[\/&]:\\&:g;$!s/$/\\/')
cat ../$ac_top_srcdir/src/Makefile.in | sed 's/$(AM_V_CXXLD)$(/$(NVCC) -ccbin=$(MPICXX) --compiler-options  "${COMPILER_FLAGS}" -I .. ${CUDA_FLAGS} ${CUDA_LDFLAGS} -o libneurongpu.so $(all_SOURCES) $(CUDA_LIBS) \
# $(AM_V_CXXLD)$(/' | sed "/libneurongpu_la_SOURCES =/ a\\
all_SOURCES = $escaped_rhs \\
" | sed 's/libneurongpu.la/libneurongpu.so/'> tmpfile
mv tmpfile ../$ac_top_srcdir/src/Makefile.in
cd ..
#Becho "libtool postinstall_cmds patch for correcting @rpath in libneurongpu.so"
#sed 's|postinstall_cmds=""|postinstall_cmds="./postinstall.sh"|g' libtool > libtool.fixed
#Bcp -f libtool.fixed libtool
#chmod +x libtool
