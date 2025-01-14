#!/bin/bash

cd -- "$(dirname -- "$BASH_SOURCE")"

if [ ! -d executorch ]; then
    git clone https://github.com/nefrock/executorch.git
fi

cd executorch
git submodule sync
git submodule update --init
# cp ../diff.patch .
# git apply diff.patch
# rm diff.patch
if "$USE_QNN"; then
    cp schema/program.fbs exir/_serialize/program.fbs
    cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
fi
cd ..


if [ ! -d qairt ] && "$USE_QNN"; then
    wget https://softwarecenter.qualcomm.com/api/download/software/qualcomm_neural_processing_sdk/v2.26.0.240828.zip
    unzip v2.26.0.240828.zip
fi

if [ ! -d llama ]; then
    mkdir llama
    echo ""
    echo "Copy model directory in \"$(pwd)/llama\"."
fi
