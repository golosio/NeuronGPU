#!/bin/bash
if [ $# -ne 3 ]; then
   echo "Usage: $0 folder distro package-name"
   echo "where folder is /srv/dev/ubuntu or /srv/deb/debian and distro is bionic, eoan, ..."
   exit
fi
reprepro -b $1 includedeb $2 $3
