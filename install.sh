#!/bin/bash
if [ "$#" -ne 0 ] && ([ "$#" -ne 2 ] || [ "$1" != "--dir" ]); then
    echo
    echo "Usage: ./install.sh"
    echo "The default installation folder is $HOME/neuralgpu"
    echo "If you want to install the library in a different folder, do:"
    echo "./install.sh [--dir installation-folder]"
    echo
    exit
fi

if [ "$#" -eq 2 ]; then
    instdir=$2
else
    instdir=$HOME/neuralgpu
fi

sudo apt install nvidia-cuda-toolkit
sudo apt install -y openmpi-bin libopenmpi-dev libomp-dev g++ python python-mpi4py python-matplotlib

ver=$(head -1 VERSION | tr -d '\040\011\012\015')
cat /home/$USER/.bashrc | grep -v 'neuralgpu_env.sh' > tmp1
echo ". $instdir/neuralgpu-$ver/bin/neuralgpu_env.sh" > tmp2
cat tmp1 tmp2 > /home/$USER/.bashrc
rm -f tmp1 tmp2

. make.sh --dir $instdir

