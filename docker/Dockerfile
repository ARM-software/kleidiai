#
# SPDX-FileCopyrightText: Copyright 2024-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#

# Ubuntu base version
ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION}

# These arguments are required to support multi-arch images
ARG BUILDARCH
ARG BUILDPLATFORM
ARG TARGETARCH
ARG TARGETOS
ARG TARGETPLATFORM

# Component versions
ARG FVP_UBUNTU_VERSION=24.04.5
ARG BAZEL_BUILD_TOOLS_VERSION=v8.5.1
ARG BAZELISK_VERSION=v1.29.0
ARG FVP_VERSION=11.32_19
ARG GLAB_VERSION=1.117.0

# Component checksums precalculated for linux/arm64 platform, provide new values
# if linux/amd64 is need to be built.
ARG FVP_MD5=c767449a92dacadfe23e561ebe53e949
ARG FVP_UBUNTU_SHA256=a91d5a93010193712d346d761372b7c9db6dfcf093893161c64ca107f05914f2
ARG BAZELISK_SHA256=e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010
ARG BUILDIFIER_SHA256=947bf6700d708026b2057b09bea09abbc3cafc15d9ecea35bb3885c4b09ccd04
ARG BUILDOZER_SHA256=87ee1d2d81d08ccae8f9147fc58503967c85878279e892f2990912412feef1a1
ARG UNUSED_DEPS_SHA256=7948f63fb2dcb5b9868dc67cf8b53ebefa11d09bfdb94cc8162449810c551998
ARG GLAB_SHA256=ce533c314b69014664020567d006b5c86435f0fa16561aba18929bea26506828

ENV XDG_CACHE_HOME=/cache BUILD_CACHE=/build_cache

RUN mkdir -p /opt/devtools ${XDG_CACHE_HOME}

# Software from base image required during image usage
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    build-essential \
    ca-certificates \
    clang \
    clang-tidy \
    cmake \
    curl \
    e2tools \
    gcovr \
    git \
    jq \
    llvm \
    libclang-rt-dev \
    ninja-build \
    pre-commit \
    python3 \
    python3-pip \
    reuse \
    unzip \
    wget

# Software from base image required during image build
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
    autoconf \
    automake \
    device-tree-compiler \
    flex \
    bison \
    bc \
    libssl-dev \
    genext2fs \
    ccache

WORKDIR /opt/devtools

# =============================================================================
# Linux virtual machine on FVP.
# =============================================================================

# Downloads the latest Fixed Virtual Platform.
# The links could be found at https://developer.arm.com/Tools%20and%20Software/Fixed%20Virtual%20Platforms/Arm%20Architecture%20FVPs
RUN --mount=type=cache,target=${BUILD_CACHE} <<EOF
    set -e
    FVP_NEW_ARCH=$(test "${TARGETARCH}" = "arm64" && echo -n "armv8" || echo -n "x86")
    FVP_OLD_ARCH=$(test "${TARGETARCH}" = "arm64" && echo -n "_armv8l" || true)
    FVP_ARCHIVE=
    # The five latest releases use two archive-name schemes and two release-directory schemes.
    for FVP_BASE_VERSION in $(echo -n ${FVP_VERSION} | cut -d '_' -f 1) $(echo -n ${FVP_VERSION} | tr '_' '-') ; do
        for FVP_ARCHIVE_CANDIDATE in \
            "FVP_Base_RevC_AEMvA_${FVP_VERSION}_Linux_${FVP_NEW_ARCH}.tar.gz" \
            "FVP_Base_RevC-2xAEMvA_${FVP_VERSION}_Linux64${FVP_OLD_ARCH}.tgz" ; do
            if wget -cO "${BUILD_CACHE}/${FVP_ARCHIVE_CANDIDATE}" \
                "https://developer.arm.com/-/cdn-downloads/permalink/FVPs-Architecture/FM-${FVP_BASE_VERSION}/${FVP_ARCHIVE_CANDIDATE}" ; then
                FVP_ARCHIVE=${FVP_ARCHIVE_CANDIDATE}
                break 2
            fi
        done
    done
    test -n "${FVP_ARCHIVE}"
    FVP_NAME=${FVP_ARCHIVE%.tar.gz}
    FVP_NAME=${FVP_NAME%.tgz}
    echo "${FVP_MD5}  ${BUILD_CACHE}/${FVP_ARCHIVE}" | md5sum --check --strict -
    mkdir -vp ${FVP_NAME}
    tar xzf ${BUILD_CACHE}/${FVP_ARCHIVE} -C ${FVP_NAME}
    # Check that we got FVP installer. If so do the installation
    if [ -f ${FVP_NAME}/${FVP_NAME}.sh ]; then
        cd ${FVP_NAME}
        # Install into old location
        ./${FVP_NAME}.sh --i-agree-to-the-contained-eula --skip-menu-install --no-interactive -d Base_RevC_AEMvA_pkg/
        # Remove installer once it finished
        rm ${FVP_NAME}.sh
        cd Base_RevC_AEMvA_pkg/
        # Reconstruct old structure for backward compatibility
        mkdir -p models
        ln -sr $(pwd)/bin models/Linux64${FVP_OLD_ARCH}GCC-9.3
    fi
    ln -s /opt/devtools/${FVP_NAME}/Base_RevC_AEMvA_pkg /opt/devtools/fvp_base_aemva
    readlink -e /opt/devtools/fvp_base_aemva/models/Linux64${FVP_OLD_ARCH}GCC-9.3/FVP_Base_RevC-2xAEMvA
EOF

# Creates the root filesystem.
RUN --mount=type=cache,target=${BUILD_CACHE},sharing=locked \
    export UBUNTU_DISTRO_VERSION=$(echo -n "$FVP_UBUNTU_VERSION" | cut -d '.' -f 1,2) && \
    wget -cO ${BUILD_CACHE}/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz "https://cdimage.ubuntu.com/ubuntu-base/releases/${UBUNTU_DISTRO_VERSION}/release/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz" && \
    echo "${FVP_UBUNTU_SHA256}  ${BUILD_CACHE}/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz" | sha256sum --check --strict - && \
    genext2fs -a ${BUILD_CACHE}/ubuntu-base-${FVP_UBUNTU_VERSION}-base-arm64.tar.gz -B 4096 -b 51200 linux-rootfs.img

# Compiles the latest Linux kernel, bootloader and root file system for FVP usage
RUN --mount=type=cache,target=${BUILD_CACHE},sharing=locked \
    --mount=type=bind,source=build_linux_bootloader.sh,target=build_linux_bootloader.sh \
    ./build_linux_bootloader.sh

WORKDIR /

# Download and install Bazel tools
RUN wget "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-${TARGETOS}-${TARGETARCH}" -O /usr/local/bin/bazelisk && \
    echo "${BAZELISK_SHA256}  /usr/local/bin/bazelisk" | sha256sum --check --strict - && \
    chmod a+x /usr/local/bin/bazelisk && \
    wget "https://github.com/bazel-contrib/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VERSION}/buildifier-${TARGETOS}-${TARGETARCH}" -O /usr/local/bin/buildifier && \
    echo "${BUILDIFIER_SHA256}  /usr/local/bin/buildifier" | sha256sum --check --strict - && \
    wget "https://github.com/bazel-contrib/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VERSION}/buildozer-${TARGETOS}-${TARGETARCH}" -O /usr/local/bin/buildozer && \
    echo "${BUILDOZER_SHA256}  /usr/local/bin/buildozer" | sha256sum --check --strict - && \
    wget "https://github.com/bazel-contrib/buildtools/releases/download/${BAZEL_BUILD_TOOLS_VERSION}/unused-deps-${TARGETOS}-${TARGETARCH}" -O /usr/local/bin/unused_deps && \
    echo "${UNUSED_DEPS_SHA256}  /usr/local/bin/unused_deps" | sha256sum --check --strict - && \
    chmod a+x /usr/local/bin/buildifier /usr/local/bin/buildozer /usr/local/bin/unused_deps

# Download and install GitLab CLI.
RUN --mount=type=cache,target=${BUILD_CACHE},sharing=locked \
    wget -cO "${BUILD_CACHE}/glab_${GLAB_VERSION}_linux_${TARGETARCH}.tar.gz" "https://gitlab.com/gitlab-org/cli/-/releases/v${GLAB_VERSION}/downloads/glab_${GLAB_VERSION}_linux_${TARGETARCH}.tar.gz" && \
    echo "${GLAB_SHA256}  ${BUILD_CACHE}/glab_${GLAB_VERSION}_linux_${TARGETARCH}.tar.gz" | sha256sum --check --strict - && \
    tar xzf "${BUILD_CACHE}/glab_${GLAB_VERSION}_linux_${TARGETARCH}.tar.gz" --strip-components=1 -C /usr/local/bin bin/glab
