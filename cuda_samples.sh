if [ "$CUDA_SAMPLES" = "" ]; then
    echo "Error:"
    echo "CUDA_SAMPLES environmental variable not defined."
    echo "Be sure that you have installed NVIDIA CUDA SAMPLES"
    echo "and set CUDA_SAMPLES to the path where they are installed."
    echo "For instance, in Linux you can add to your .bashrc file:"
    echo "export CUDA_SAMPLES=/path-to-samples/NVIDIA_CUDA-7.5_Samples/"
    exit
fi
if [ ! -f $CUDA_SAMPLES/6_Advanced/scan/scan.cu ]; then
    echo "Error:"
    echo "File $CUDA_SAMPLES/6_Advanced/scan/scan.cu not found"
    exit
fi
if [ ! -f $CUDA_SAMPLES/6_Advanced/scan/scan_common.h ]; then
    echo "Error:"
    echo "File $CUDA_SAMPLES/6_Advanced/scan/scan_common.h not found"
    exit
fi
if [ ! -f $CUDA_SAMPLES/common/inc/helper_cuda.h ]; then
    echo "Error:"
    echo "File $CUDA_SAMPLES/common/inc/helper_cuda.h not found"
    exit
fi
if [ ! -f $CUDA_SAMPLES/common/inc/helper_string.h ]; then
    echo "Error:"
    echo "File $CUDA_SAMPLES/common/inc/helper_string.h not found"
    exit
fi
cat $CUDA_SAMPLES/6_Advanced/scan/scan.cu | sed 's/THREADBLOCK_SIZE 256/THREADBLOCK_SIZE 1024/' > src/scan_tmp.cu
