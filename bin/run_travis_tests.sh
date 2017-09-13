#!/bin/sh
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

set -ev

cd $HIVEMALL_HOME

mvn -q scalastyle:check test -Pspark-2.1

# Tests the spark-2.2/spark-2.0 modules only in the following runs
if [[ ! -z "$(java -version 2>&1 | grep 1.8)" ]]; then
  mvn -q scalastyle:check clean -Djava.source.version=1.8 -Djava.target.version=1.8 \
    -Pspark-2.2 -pl spark/spark-2.2 -am test -Dtest=none
fi

mvn -q scalastyle:check clean -Pspark-2.0 -pl spark/spark-2.0 -am test -Dtest=none

exit 0

