#!/bin/bash
if [ $# -ne 1 ]; then
   echo "Usage: $0 package-name"
   exit
fi

dpkg-sig --sign builder $1
