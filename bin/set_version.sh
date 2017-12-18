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

if [ "$HIVEMALL_HOME" = "" ]; then
  if [ -e ../bin/${0##*/} ]; then
    HIVEMALL_HOME=".."
  elif [ -e ./bin/${0##*/} ]; then
    HIVEMALL_HOME="."
  else
    echo "env HIVEMALL_HOME not defined"
    exit 1
  fi
fi

cd $HIVEMALL_HOME

function yes_or_no() {
  while true; do
    echo "Type [Y/N]"
    echo -n ">>"
    read answer

    case $answer in
      [yY])
        return 0
        ;;
      [nN])
        return 1        
        ;;
    esac
  done
}

old_version=`cat VERSION`
echo "Current version number is ${old_version}"
echo

echo "This script will update the version string of Hivemall."
echo
echo "Please input a version string (e.g., 0.4.3-rc.2)"
echo -n ">>"
read new_version

echo
echo "--------------------------------------------------------------------------"
echo "[Here are the list of files to update]"
echo
find . -type f \( -name 'VERSION' -o -name 'pom.xml' -o -name 'HivemallConstants.java' -o -name 'HivemallOpsSuite.scala' -o -name 'HiveUdfSuite.scala' \)  | xargs grep ${old_version}
echo "--------------------------------------------------------------------------"
echo

echo "Do you really want to update Hivemall version string from ${old_version} to ${new_version}?"
echo

yes_or_no

if [ "$?" -eq 1 ]; then
  echo "aborted!"
  exit 1
fi
echo

echo -n "Updating ..."
find . -type f \( -name 'VERSION' -o -name 'pom.xml' -o -name 'HivemallConstants.java' -o -name 'HivemallOpsSuite.scala' -o -name 'HiveUdfSuite.scala' \) | xargs sed -i '' -e "s/${old_version}/${new_version}/g"
echo "Done!"
