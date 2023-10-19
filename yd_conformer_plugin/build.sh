#!/bin/bash
set -e

export WORKSPACE=${PWD}
export PATH="/usr/local/ccache:$PATH"

###############################################################################
# Environment Variables
###############################################################################
# BUILD_MODE: release/debug
# BUILD_DIR: build(default)
# BUILD_JOBS: Number of CPU used for project building
# TARGET_MLU_ARCH: CNFATBIN/MLU220/MLU270/MLU370/CE3226
# TARGET_CPU_ARCH: x86_64-linux-gnu/i386-linux-gnu/aarch64-linux-gnu/arm-aarch64-gnu/arm-linux-gnueabihf
# TARGET_C_COMPILER: C compiler full-path
# TARGET_CXX_COMPILER: CXX compiler full-path
# TOOLCHAIN_ROOT: /path/to/your/cross-compile-toolchain
# MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP: The list of pluginop names

BUILD_MODE=${BUILD_MODE:-release}
BUILD_DIR=${BUILD_DIR:-build}
BUILD_JOBS=${BUILD_JOBS:-32}
TARGET_MLU_ARCH=${TARGET_MLU_ARCH:-CNFATBIN}
TARGET_CPU_ARCH=${TARGET_CPU_ARCH:-$(uname -m)-linux-gnu}
TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT}
TARGET_C_COMPILER=${TARGET_C_COMPILER}
TARGET_CXX_COMPILER=${TARGET_CXX_COMPILER}
MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP="${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP}"
MAGICMIND_PLUGIN_BUILD_TEST="${MAGICMIND_PLUGIN_BUILD_TEST:-ON}"
if [ ! $ABI_MODE]; then
  ABI_MODE=1
fi 
###############################################################################

export BUILD_DIR

###############################################################################
# Common Funcs
###############################################################################
check_deb_package() {
  if [ -z "$(dpkg -l | grep ${1})" ]; then
    echo "-- Please sudo yum install ${1}"
    exit -1
  fi
}

check_rpm_package() {
  if [ -z "$(rpm -qa | grep ${1})" ]; then
    echo "-- Please sudo yum install ${1}"
    exit -1
  fi
}

usage() {
  echo "Usage: ./build.sh <options>"
  echo 
  echo "       If need specify neuware path, please:"
  echo "         First, export NEUWARE_HOME=/path/to/where/neuware/installed"
  echo "         Second, export TOOLCHAIN_ROOT=/path/to/cross_toolchain if cross-compile for aarch64-linux-gnu"
  echo 
  echo "Options:"
  echo "      <null>                           If no option presented, defualt arch is set to cnfatbin and x86_64-linux-gnu."
  echo "      -h, --help                       Print usage."
  echo "      --mlu220                         Build for target MLU arch MLU220, where __BANG_ARCH__ = 220."
  echo "      --mlu270                         Build for target MLU arch MLU270, where __BANG_ARCH__ = 270."
  echo "      --mlu290                         Build for target MLU arch MLU290, where __BANG_ARCH__ = 290."
  echo "      --ce3226                         Build for target MLU arch CE3226, where __BANG_ARCH__ = 322."
  echo "      --mlu370                         Build for target MLU arch MLU370, where __BANG_ARCH__ = 370."
  echo "      --aarch64                        Build for target CPU arch aarch64-linux-gnu."
  echo "      --d, --debug                     Build with debug symbols."
  echo "      -v, --verbose                    Build with verbose output."
  echo "      -j, --jobs=*                     Build with parallel jobs."
  echo "      --filter=*                       Build with specific operation only. op sperated with ;."
  echo
}


###############################################################################
# Building MagicMind Plugin
###############################################################################
# 1. Check dependency
if [ -f "/etc/os-release" ]; then
  source /etc/os-release
  if [[ "${NAME}" == Ubuntu* ]] || [[ "${NAME}" == Debian* ]]; then
    check_deb_package cmake
    CMAKE=cmake

    # Helpful but not necessary
    # check_deb_package ccache	
    # check_deb_package valgrind
    # check_deb_package doxygen
    # check_deb_package libopenblas-dev
    # check_deb_package libopencv-dev
  elif [[ "${NAME}" == CentOS* ]]; then
    check_rpm_package cmake
    CMAKE=cmake

    # Helpful but not necessary
    # check_deb_package ccache	
    # check_deb_package valgrind
    # check_deb_package doxygen
    # check_deb_package openblas
    # check_deb_package opencv-devel
  else
    echo "-- Not support build on this os!"
    exit -1
  fi
else
  echo "-- Not support build on this os!"
  exit -1
fi

# 2. Create build dir
if [ ! -d "${BUILD_DIR}" ]; then
  mkdir "${BUILD_DIR}"
fi

# 3. Handle build-options
cmdline_args=$(getopt -o h,d,v,j:,t: --long help,debug,verbose,jobs:,mlu220,mlu270,ce3226,mlu370,aarch64,arm,filter: -n 'build.sh' -- "$@")
eval set -- "$cmdline_args"
if [ $? != 0 ]; then
  echo "Unknown options, use -h or --help" >&2
  exit -1
fi

if [ $# != 0 ]; then
  while true; do
    case "$1" in
      --mlu220)
          TARGET_MLU_ARCH="mtp_220"
          shift
          ;;
      --mlu270)
          TARGET_MLU_ARCH="mtp_270"
          shift
          ;;
      --ce3226)
          TARGET_MLU_ARCH="mtp_322"
          shift
          ;;
      --mlu370)
          TARGET_MLU_ARCH="mtp_370"
          shift
          ;;
      --aarch64)
          TARGET_CPU_ARCH="aarch64-linux-gnu"
          shift
          ;;
      --arm)
          TARGET_CPU_ARCH="arm-linux-gnu"
          shift
          ;;
      -h | --help)
          usage
          exit 0
          ;;
      -d | --debug)
          BUILD_MODE="debug"
	  echo "-- Using debug mode."
          shift
          ;;
      -v | --verbose)
          BUILD_VERBOSE="VERBOSE=1"
	  shift
          ;;
      -j | --jobs)
	  shift
	  BUILD_JOBS=$1
	  shift
          ;;
      --filter)
	  shift
	  MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP=$1
	  echo "-- Build mm-plugin eith Operator: ${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP} only."
	  shift
          ;;
      -t | --type)
	  shift
	  RELEASE_TYPE=$1
	  shift
          ;;
      --)
          shift
	  break
	  ;;
      *)
          echo "-- Unknown options ${1}, use -h or --help."
	  usage
	  exit -1
	  ;;
    esac
  done
fi

# 4. Check NEUWARE_HOME and cncc
if [ ! -z "${NEUWARE_HOME}" ]; then
  echo "-- using NEUWARE_HOME = ${NEUWARE_HOME}"
else
  echo "-- NEUWARE_HOME is not set. Refer to README.md to prepare NEUWARE_HOME environment."
  exit -1
fi

if [ -z $(which cncc) ]; then
  # cncc is not in PATH, try to search cncc in ${NEUWARE_HOME}/bin.
  export PATH="${NEUWARE_HOME}/bin":$PATH
  if [ -z $(which cncc) ]; then
    echo "-- CNCC cannot be found."
    exit -1
  fi
  cncc --version || ( echo "-- CNCC cannot be used for current CPU target" && exit -1)
else
  # cncc is already in PATH, but cncc in ${NEUWARE_HOME}/bin is prefered.
  if [ -f "${NEUWARE_HOME}/bin/cncc" ]; then
    ${NEUWARE_HOME}/bin/cncc --version || cannot_use_neuware_cncc=1
    if [ "x${cannot_use_neuware_cncc}" = "x1" ]; then
      echo "-- cncc in ${NEUWARE_HOME}/bin cannot be used for compiling, use default cncc in PATH."
      echo "-- rename cncc libraries in NEUWARE_HOME."
      mv -vf ${NEUWARE_HOME}/lib/clang{,_${TARGET_CPU_ARCH}} || echo "mv clang lib failed."
      mv -vf ${NEUWARE_HOME}/bin{,_${TARGET_CPU_ARCH}} || echo "mv bin failed."
    else
      export PATH="${NEUWARE_HOME}/bin":$PATH
    fi
  fi
fi

echo "-- cncc: $(which cncc)"
export LD_LIBRARY_PATH="${NEUWARE_HOME}/lib64":$LD_LIBRARY_PATH

# 5. Check compiler and target
[[ ! ${TARGET_CPU_ARCH} =~ $(uname -m) ]] && is_cross_compile=1
IFS='-' read -a target_cpu <<< ${TARGET_CPU_ARCH}
# NOTE variable TARGET_C_COMPILER and TARGET_CXX_COMPILER have higher priority than TOOLCHAIN_ROOT
if [ -z ${TOOLCHAIN_ROOT} ]; then
  if [ -z ${is_cross_compile} ]; then
    # Native build and toolchain not set
    [ -z ${TARGET_C_COMPILER} ] && TARGET_C_COMPILER=$(which cc)
    [ -z ${TARGET_CXX_COMPILER} ] && TARGET_CXX_COMPILER=$(which c++)
    TOOLCHAIN_ROOT=$(dirname $(dirname ${TARGET_CXX_COMPILER}))
  else
    # cross-compiling but toolchain not set, use default path, i.e., MagicMind-devel environment
    case "$(uname -m)-${target_cpu}" in
      x86_64-aarch64)
        TOOLCHAIN_ROOT="/usr/local/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"
        ;;
      x86_64-arm)
        TOOLCHAIN_ROOT=""
        ;;
      *)
        echo "Please export TOOLCHAIN_ROOT=/path/to/your/cross-compiler."
        exit -1
        ;;
    esac
  fi
fi

[ -z ${TARGET_C_COMPILER} ] && TARGET_C_COMPILER=${TOOLCHAIN_ROOT}/bin/${TARGET_CPU_ARCH}-gcc
[ -z ${TARGET_CXX_COMPILER} ] && TARGET_CXX_COMPILER=${TOOLCHAIN_ROOT}/bin/${TARGET_CPU_ARCH}-g++

echo "-- TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT}"
echo "-- TARGET_C_COMPILER=${TARGET_C_COMPILER}"
echo "-- TARGET_CXX_COMPILER=${TARGET_CXX_COMPILER}"
export PATH=$(dirname ${TARGET_CXX_COMPILER}):${TOOLCHAIN_ROOT}/bin:$PATH
export TOOLCHAIN_ROOT
export CC=$(basename ${TARGET_C_COMPILER})
export CXX=$(basename ${TARGET_CXX_COMPILER})

# 6. Build Project
SOURCE_DIR=$PWD
pushd ${BUILD_DIR}
  ${CMAKE} -DCMAKE_BUILD_TYPE="${BUILD_MODE}" \
           -DNEUWARE_HOME="${NEUWARE_HOME}" \
           -DTOOLCHAIN_ROOT="${TOOLCHAIN_ROOT}" \
           -DTARGET_MLU_ARCH="${TARGET_MLU_ARCH}" \
           -DTARGET_CPU_ARCH="${TARGET_CPU_ARCH}" \
	   -DCMAKE_C_COMPILER="$(basename ${TARGET_C_COMPILER})" \
	   -DCMAKE_CXX_COMPILER="$(basename ${TARGET_CXX_COMPILER})" \
           -DMAGICMIND_PLUGIN_BUILD_SPECIFIC_OP="${MAGICMIND_PLUGIN_BUILD_SPECIFIC_OP}" \
           -DMAGICMIND_PLUGIN_BUILD_TEST="${MAGICMIND_PLUGIN_BUILD_TEST}" \
	   -DMAGICMIND_INCLUDE="${MAGICMIND_INCLUDE}" \
	   -DMAGICMIND_UTILS="${MAGICMIND_UTILS}" \
	   -DABI=${ABI_MODE} \
           ..
popd
${CMAKE} --build ${BUILD_DIR} -- ${BUILD_VERBOSE} -j${BUILD_JOBS}

