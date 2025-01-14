#!/bin/bash

cd -- "$(dirname -- "BASH_SOURCE")"

CLEAN=false
while (( $# > 0 ))
do
    case $1 in
	clean)
	    CLEAN=true
	    ;;
	*)
	    ;;
    esac
    shift
done

if [ ! -d executorch ]; then
    echo "Not found executorch repository." >&2
    echo "Please run \"download.sh\" and \"setup.sh\" ahead." >&2
    exit 1
fi

cd executorch

if "$CLEAN"; then
    rm -rf build-* *-out* buck2-bin $HOME/.buck
    if [ -e backends/qualcomm/aot/ir/qcir_generated.h ]; then
        rm backends/qualcomm/aot/ir/qcir_generated.h
    fi
    exit 0
fi

if [ "$ANDROID_SDK" == "" ]; then
    echo "Please run \"setup.sh\" ahead." >&2
    exit 1
fi

if "$USE_QNN"; then
    ./install_requirements.sh
else
    ./install_requirements.sh --pybind xnnpack
fi
if [ $? -ne 0 ]; then
    echo "Failed to \"Executorch setup\"."
    exit 1
fi

if "$USE_QNN"; then
    ./backends/qualcomm/scripts/build.sh --release
    if [ $? -ne 0 ]; then
        echo "Failed to \"Build QNN backend with ExecuTorch (step 1)\"."
        exit 1
    fi

    cmake -DPYTHON_EXECUTABLE=python -DCMAKE_INSTALL_PREFIX=cmake-out -DEXECUTORCH_ENABLE_LOGGING=1 -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_QNN=ON -DQNN_SDK_ROOT=${QNN_SDK_ROOT} -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON -Bcmake-out .
    if [ $? -ne 0 ]; then
        echo "Failed to \"Build QNN backend with ExecuTorch (step 2)\"."
        exit 1
    fi

    cmake --build cmake-out -j16 --target install --config Release
    if [ $? -ne 0 ]; then
        echo "Failed to \"Build QNN backend with ExecuTorch (step 3)\"."
        exit 1
    fi
fi

sh examples/models/llama/install_requirements.sh
if [ $? -ne 0 ]; then
    echo "Failed to \"Setup Llama Runner (step 1)\"."
    exit 1
fi

if "$USE_QNN"; then
    cmake -DPYTHON_EXECUTABLE=python -DCMAKE_INSTALL_PREFIX=cmake-out -DCMAKE_BUILD_TYPE=Release -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON -DEXECUTORCH_BUILD_QNN=ON -Bcmake-out/examples/models/llama examples/models/llama 
    if [ $? -ne 0 ]; then
        echo "Failed to \"Setup Llama Runner (step 2)\"."
        exit 1
    fi

    cmake --build cmake-out/examples/models/llama -j16 --config Release
    if [ $? -ne 0 ]; then
        echo "Failed to \"Setup Llama Runner (step 3)\"."
        exit 1
    fi
fi

echo ""
echo "Success!"
