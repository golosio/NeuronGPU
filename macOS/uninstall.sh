# remove installation directory
rm -fr /usr/local/nestgpu

# find python package directory
SITEDIR=$(python -m site --user-site)
SITEDIR3=$(python3 -m site --user-site)

# remove .pth file
rm -f "$SITEDIR/nestgpu.pth"
rm -f "$SITEDIR3/nestgpu.pth"

# remove symbolic link to the dynamic-link library from /usr/local/lib
rm -f /usr/local/lib/libnestgpu.so
