FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV TZ Asia/Tokyo
ENV NVIDIA_DRIVER_CAPABILITIES graphics

RUN apt-get update \
    && apt-get install -y \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    sudo \
    vim \
    jq \
    vim \
    unzip \
    gdb \
    rsync \
    libssl-dev \
    zip \
    zstd \
    --no-install-recommends


RUN apt-get install -y \
    clang-12 \
    llvm-12 \
    lld \
    libomp-12-dev \
    libc++-12-dev \
    make \
    openjdk-17-jdk \
    python3.10-dev \
    python3.10-venv \
    --no-install-recommends

# Clean Up
RUN	apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-12 50 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-12 50 \
    && update-alternatives --install /usr/bin/cc cc /usr/bin/clang 50 \
    && update-alternatives --install /usr/bin/c++ c++ /usr/bin/clang++ 50

RUN wget -q "https://github.com/facebook/buck2/releases/download/2024-05-15/buck2-x86_64-unknown-linux-gnu.zst" \
    && zstd -d buck2-x86_64-unknown-linux-gnu.zst -o buck2 \
    && chmod +x buck2 \
    && mv buck2 /usr/bin/ \
    && rm buck2-x86_64-unknown-linux-gnu.zst

ENV PYTHON /usr/bin/python3.10
ENV JAVA_HOME /usr/lib/jvm/java-17-openjdk-amd64

ENV ANDROID_HOME /home/android_sdk
RUN mkdir -p $ANDROID_HOME \
    && curl https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip?hl=ja -o $ANDROID_HOME/cmdline-tools.zip \
    && unzip $ANDROID_HOME/cmdline-tools.zip -d $ANDROID_HOME \
    && echo "y" | $ANDROID_HOME/cmdline-tools/bin/sdkmanager --sdk_root="${ANDROID_HOME}" --install "build-tools;33.0.1" "platform-tools" "platforms;android-34" "sources;android-34" \
    && echo "y" | $ANDROID_HOME/cmdline-tools/bin/sdkmanager --sdk_root="${ANDROID_HOME}" --install "ndk;27.1.12297006" \
    && chmod 777 `find $ANDROID_HOME -type d`


# If you want to add vscode user to image, you should comment out the following code
# ARG USERNAME=vscode
# ARG USER_UID=1000
# ARG USER_GID=$USER_UID
# RUN	groupadd --gid $USER_GID $USERNAME \
#     && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME \
#     && su $USERNAME \
#     && cd $HOME

# WORKDIR /workspaces
# USER vscode
