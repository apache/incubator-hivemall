#!/bin/sh -eux
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

DATA_DIR='/root/data'
HDFS_DATA_DIR='/dataset/iris/raw'
DATA='iris.data'
mkdir -p $DATA_DIR
[ -f $DATA_DIR/$DATA ] || wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data -O $DATA_DIR/$DATA
hadoop fs -mkdir -p $HDFS_DATA_DIR
awk -F',' 'NF >0 {OFS="|"; print NR,$5,$1","$2","$3","$4}' $DATA_DIR/$DATA \
  | hadoop fs -put - $HDFS_DATA_DIR/$DATA
hive -e " \
  create database if not exists iris; \
  use iris; \
  create external table iris_raw (rowid int, label string, features array<float>) \
    row format delimited fields terminated by '|' \
    collection items terminated by ',' \
    stored as textfile location \"$HDFS_DATA_DIR\";"
