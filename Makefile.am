AUTOMAKE_OPTIONS = 1.4 no-dependencies foreign

## ------------------- ##
#
## ------------------- ##
ACLOCAL_AMFLAGS = -I m4

SUBDIRS=src
#dist_data_DATA=autogen.sh
CCLD = $(CC)
LINK = $(CCLD) $(AM_CFLAGS) $(CFLAGS) $(AM_LDFLAGS) $(LDFLAGS) -o $@
