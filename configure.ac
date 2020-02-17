AC_PREREQ([2.65])
AC_INIT([neurongpu], [1.0.6], [])

# Config initialization
AC_CONFIG_SRCDIR([m4/ct_check_cuda.m4])
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

# Checks for

# Checks for libraries.
AC_CHECK_LIB([m], [pow])

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
src/Makefile])


AC_OUTPUT

#$ac_top_srcdir/patch.sh
./patch.sh
