AC_PREREQ([2.65])
AC_INIT([neurongpu], [1.0.6], [])

# Config initialization
AC_CONFIG_SRCDIR([m4/ct_check_cuda.m4])
AC_CONFIG_SRCDIR([m4/pypath.m4])
#, [m4/ct_check_cuda_sdk.m4])
AC_CONFIG_AUX_DIR(m4)
AC_CONFIG_HEADER(config.h)
AC_CONFIG_MACRO_DIRS([m4])

AC_ENABLE_SHARED
AC_DISABLE_STATIC
LT_INIT

# automake initialization
AM_INIT_AUTOMAKE

# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AM_PROG_CC_C_O

# Search nvcc compiler
AC_PATH_PROG(NVCC, nvcc, "no")
AS_IF([test "x$NVCC" = "xno"],[AC_MSG_ERROR([NVCC compiler not found!])])
# Make variable $NVCC available in Makefile.am
AC_SUBST(NVCC)

# check for OpenMP
AC_OPENMP

#check for Python
AM_PATH_PYTHON([2.4])
#adl_CHECK_PYTHON
#AC_MSG_ERROR([$PYTHONINC])


# Checks for libraries.
AC_CHECK_LIB([m], [pow])
#AC_CHECK_LIB([just_a_test], [pow], [], [AC_MSG_ERROR([pow was not found in libjust_a_test])])

# Checks for header files.
AC_CHECK_HEADERS([limits.h stdlib.h string.h])

# check for CUDA
CT_CHECK_CUDA_API

# check for CUDA SDK
#CT_CHECK_CUDA_SDK

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.
AC_CHECK_FUNCS([pow])

AC_CONFIG_FILES([Makefile
src/Makefile
pythonlib/Makefile])


AC_OUTPUT

#$ac_top_srcdir/patch.sh
./patch.sh
