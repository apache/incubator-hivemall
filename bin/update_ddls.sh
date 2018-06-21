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

cd $HIVEMALL_HOME

define_all() {
  echo "\ndrop temporary function if exists $function_name;\ncreate temporary function $function_name as '$class_path';" >> resources/ddl/define-all.hive
  echo "Added to resources/ddl/define-all.hive";

  echo "\nsqlContext.sql(\"DROP TEMPORARY FUNCTION IF EXISTS $function_name\")\nsqlContext.sql(\"CREATE TEMPORARY FUNCTION $function_name AS '$class_path'\")" >> resources/ddl/define-all.spark
  echo "Added to resources/ddl/define-all.spark";
}

define_all_as_permanent() {
  echo "\nDROP FUNCTION IF EXISTS $function_name;\nCREATE FUNCTION $function_name as '$class_path' USING JAR '\${hivemall_jar}';" >> resources/ddl/define-all-as-permanent.hive
  echo "Added to resources/ddl/define-all-as-permanent.hive";
}

define_additional() {
  echo "\ndrop temporary function if exists $function_name;\ncreate temporary function $function_name as '$class_path';" >> resources/ddl/define-additional.hive
  echo "Added to resources/ddl/define-additional.hive";
}

read -p "Function name (e.g., 'hivemall_version'): " function_name
read -p "Class path (e.g., 'hivemall.HivemallVersionUDF'): " class_path

prefix="$(echo "$class_path" | cut -d'.' -f1,2)"
if [[ $prefix == 'hivemall.xgboost' ]]; then
  define_all_as_permanent
  define_additional
elif [[ $prefix == 'hivemall.nlp' ]]; then
  define_additional
else
  define_all
  define_all_as_permanent
fi
