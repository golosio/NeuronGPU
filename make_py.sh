if [ "$#" -ne 1 ]; then
    echo
    echo "Please provide the path for installing the library."
    echo "Example:"
    echo "./$0 $HOME/lib/"
    exit
fi

if [ ! -d "$1" ]; then
    mkdir $1

    if [ ! -d "$1" ]; then
        echo
        echo "Error:"
        echo "Cannot create the directory $1 for installing the library."
        exit
    fi
fi

g++ -Wall -fPIC -shared -L ./lib -I ./src -o lib/libpyneuralgpu.so src/pyneuralgpu.cpp -lneuralgpu

cp lib/libpyneuralgpu.so $1

echo
echo "Done"
echo "Please be sure to include the directory $1 in your shared library path"
echo "For instance in Linux you can add to your .bashrc file"
echo "the following line:"
echo "export LD_LIBRARY_PATH=$1:\$LD_LIBRARY_PATH"
