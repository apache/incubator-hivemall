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

usage() {
    echo "./set_version.sh [--pom --version <ARG>]"
    echo
    echo "Option:"
    echo "  -h, --help       | show usage"
    echo "  --version <ARG>  | set version"
    echo "  --pom            | "
    echo
}

update_pom=1
for opt in "$@"; do
  case "${opt}" in
    '--pom' )
	update_pom=0
	shift
	;;
    '--version' )
	if [[ -z "$2" ]] || [[ "$2" =~ ^-+ ]]; then
	    echo "$0: $1 option MUST have a version string as the argument" 1>&2
	    exit 1
	fi
	new_version="$2"
	shift 2
	;;
    '-h'|'--help' )
	usage
	exit 1
	;;
  esac
done

old_version=`cat VERSION`
echo "Current version number is ${old_version}"

if [ -z "$new_version" ]; then
  echo
  echo "This script will update the version string of Hivemall."
  echo
  echo "Please input a version string (e.g., 0.4.3-rc.2)"
  echo -n ">>"
  read new_version
  echo
else
  echo "New version number is ${new_version}"
  echo
fi


#if [ $update_pom -eq 1 ]; then
#  echo "Do you want update pom.xml as well?"
#  yes_or_no
#  update_pom="$?"
#fi

echo "--------------------------------------------------------------------------"
echo "[Here are the list of files to update]"
echo 
if [ $update_pom -eq 0 ]; then
  find . -type f \( -name 'VERSION' -o -name 'pom.xml' -o -name 'HivemallConstants.java' -o -name 'HivemallOpsSuite.scala' -o -name 'HiveUdfSuite.scala' \)  | xargs grep ${old_version}
else
  find . -type f \( -name 'VERSION' -o -name 'HivemallConstants.java' -o -name 'HivemallOpsSuite.scala' -o -name 'HiveUdfSuite.scala' \)  | xargs grep ${old_version}
fi
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
if [ $update_pom -eq 0 ]; then
  find . -type f \( -name 'VERSION' -o -name 'pom.xml' -o -name 'HivemallConstants.java' -o -name 'HivemallOpsSuite.scala' -o -name 'HiveUdfSuite.scala' \) | xargs sed -i '' -e "s/${old_version}/${new_version}/g"
else
  find . -type f \( -name 'VERSION' -o -name 'HivemallConstants.java' -o -name 'HivemallOpsSuite.scala' -o -name 'HiveUdfSuite.scala' \) | xargs sed -i '' -e "s/${old_version}/${new_version}/g"
fi

echo "Done!"
