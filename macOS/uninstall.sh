# remove installation directory
rm -fr /usr/local/neurongpu

# find python package directory
SITEDIR=$(python -m site --user-site)
SITEDIR3=$(python3 -m site --user-site)

# remove .pth file
rm -f "$SITEDIR/neurongpu.pth"
rm -f "$SITEDIR3/neurongpu.pth"

# remove symbolic link to the dynamic-link library from /usr/local/lib
rm -f /usr/local/lib/libneurongpu.so
