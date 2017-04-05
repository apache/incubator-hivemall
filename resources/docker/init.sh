#!/bin/sh -eux

for x in `cd /etc/init.d; ls hadoop-hdfs-*`; do service $x start; done
/usr/lib/hadoop/libexec/init-hdfs.sh
for x in `cd /etc/init.d; ls hadoop-yarn-* hadoop-mapreduce-*`; do service $x start; done
