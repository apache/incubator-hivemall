#!/bin/bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

# xgboost requires g++-4.6 or higher (https://github.com/dmlc/xgboost/blob/master/doc/build.md),
# so we need to first check if the requirement is satisfied.
# g++-4.7 or higher is required when building on Ubuntu 14.04 on Docker.
COMPILER_REQUIRED_VERSION=4.7

# See simple version normalization: http://stackoverflow.com/questions/16989598/bash-comparing-version-numbers
function compiler_version { echo "$@" | awk -F. '{ printf("%03d%03d%03d\n", $1,$2,$3); }'; }

arch=$(uname -s)
if [ $arch = 'Darwin' ]; then
  if [ -z $CXX ]; then
    if type "g++-5" > /dev/null 2>&1; then
       export CXX=`which g++-5`
    elif type "g++-6" > /dev/null 2>&1; then
       export CXX=`which gcc-6`
    else
       echo 'export CXX=`which g++-X`; is required.'
       echo 'Run `brew install g++-5; export CXX=g++-5;`'
       exit 1
    fi
  fi
else
   # linux defaults
   if [ -z $CXX ]; then
     if type "g++" > /dev/null 2>&1; then
       export CXX=`which g++`
       COMPILER_VERSION_NUMBER=`${CXX} --version 2> /dev/null | grep ^g++ | \
         awk 'match($0, /[0-9]+\.[0-9]+\.[0-9]+/) {print substr($0, RSTART, RLENGTH)}'`
       if [ $(compiler_version $COMPILER_VERSION_NUMBER) -lt $COMPILER_REQUIRED_VERSION ]; then
         echo "You must compile xgboost with GNU g++-$COMPILER_REQUIRED_VERSION or higher," \
              "but the detected compiler was g++-$COMPILER_VERSION_NUMBER"
         exit 1
       fi
     else
       echo 'g++ does not find. export CXX=`which g++-X`; is required.'
       exit 1
     fi
  fi
fi

# Move to a top directory
if [ "$HIVEMALL_HOME" = "" ]; then
  if [ -e ../bin/${0##*/} ]; then
    HIVEMALL_HOME=`pwd`/..
  elif [ -e ./bin/${0##*/} ]; then
    HIVEMALL_HOME=`pwd`
  else
    echo "env HIVEMALL_HOME not defined"
    exit 1
  fi
fi

cd $HIVEMALL_HOME

# Final output dir for a custom-compiled xgboost binary
HIVEMALL_LIB_DIR="$HIVEMALL_HOME/xgboost/src/main/resources/lib/"
rm -rf $HIVEMALL_LIB_DIR >> /dev/null
mkdir -p $HIVEMALL_LIB_DIR

# Target commit hash value
XGBOOST_HASHVAL='2471e70f2436fbb6a76a0ca0121b96c07d994c4a'

# Move to an output directory
XGBOOST_OUT="$HIVEMALL_HOME/target/xgboost-$XGBOOST_HASHVAL"
rm -rf $XGBOOST_OUT >> /dev/null
mkdir -p $XGBOOST_OUT
cd $XGBOOST_OUT

# Fetch xgboost sources
git clone --progress \
  --depth 5 --branch xgboost_v0.60_with_portable_binaries --single-branch \
  https://github.com/myui/xgboost.git
cd xgboost
git checkout $XGBOOST_HASHVAL

# Resolve dependent sources
git submodule init
git submodule update

# Copy a built binary to the output
cd jvm-packages
ENABLE_STATIC_LINKS=1 CC=${CC} CXX=${CXX} ./create_jni.sh
cp ./lib/libxgboost4j.* "$HIVEMALL_LIB_DIR"

