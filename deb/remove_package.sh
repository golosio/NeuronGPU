#!/bin/bash
if [ $# -ne 1 ]; then
   echo "Usage: $0 distro"
   exit
fi
sudo reprepro -Vb /srv/deb/ubuntu remove $1 nestgpu
