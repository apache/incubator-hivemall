#!/bin/sh -eux

/etc/init.d/ssh start
$HADOOP_HOME/sbin/start-dfs.sh
$HADOOP_HOME/sbin/start-yarn.sh
$HADOOP_HOME/sbin/mr-jobhistory-daemon.sh start historyserver
[ -d ~/metastore_db ] || schematool -initSchema -dbType derby
