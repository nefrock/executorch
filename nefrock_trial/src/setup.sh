#!/bin/bash

cd -- "$(dirname -- "$BASH_SOURCE")"
cur=$(pwd)

QNN=false
while (( $# > 0 ))
do
    case $1 in
	-q | --qnn)
	    QNN=true
	    ;;
	*)
	    ;;
    esac
    shift
done

export USE_QNN=$QNN

if [ "${PYTHON}" != "" ] && [ "$(which $PYTHON)" != "" ]; then
    echo "Use $(which $PYTHON)"
else
    echo "Not set or found python command." >&2
    echo "Please set env variable \"PYTHON\" to specify python command." >&2
    exit 1
fi

if [ "$ANDROID_HOME" == "" ]; then
    echo "Not set android home path." >&2
    echo "Please set env variable \"ANDROID_HOME\"." >&2
    exit 1
fi

if [ ! -d $HOME/venv ]; then
    $PYTHON -m venv $HOME/venv
    source $HOME/venv/bin/activate
    pip install --upgrade pip
else
    source $HOME/venv/bin/activate
fi

export PATH="$ANDROID_HOME/platform-tools:$JAVA_HOME/bin:$PATH"

export ANDROID_SDK="$ANDROID_HOME"
export ANDROID_NDK_ROOT="$ANDROID_HOME/ndk/27.1.12297006"
export ANDROID_NDK="$ANDROID_HOME/ndk/27.1.12297006"
export ANDROID_ABI=arm64-v8a

if "$USE_QNN"; then
    if [ -d $cur/qairt/2.26.0.240828 ]; then
	export QNN_SDK_ROOT="$cur/qairt/2.26.0.240828"
	export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang/:$LD_LIBRARY_PATH
    else
	echo "Not found \"qnn sdk v2.26\" in \"$cur\"." >&2
	echo "Please run \"download.sh\" ahead." >&2
    fi
else
    echo "Info: Not use qnn sdk." >&2
fi

if [ -d $cur/executorch ]; then
    export EXECUTORCH_ROOT="$cur/executorch"
    export PYTHONPATH=$EXECUTORCH_ROOT/..
else
    echo "Not found \"executorch\" in \"$cur\"." >&2
    echo "Please run \"download.sh\" ahead." >&2
fi

