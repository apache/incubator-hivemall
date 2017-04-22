#!/bin/sh -eux

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
