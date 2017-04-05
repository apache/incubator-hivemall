FROM openjdk:7

WORKDIR /root/

ARG PREBUILD=true

COPY . /hivemall/

RUN set -eux && \
    wget https://archive.cloudera.com/cdh5/debian/jessie/amd64/cdh/archive.key -O - \
      | apt-key add - && \
    wget https://archive.cloudera.com/cdh5/debian/jessie/amd64/cdh/cloudera.list \
      -O /etc/apt/sources.list.d/cloudera.list && \
    sed -i \
      -e 's!${baseUrl}!http://archive.cloudera.com!g' \
      -e 's!${category}!cdh5!g' \
      /etc/apt/sources.list.d/cloudera.list && \
    apt update && \
    apt install -y --no-install-recommends maven hadoop-conf-pseudo hive && \
    su hdfs -c 'hdfs namenode -format' && \
    rm /usr/lib/hadoop/lib/slf4j-log4j12.jar && \
    ln -s /usr/lib/parquet/lib/slf4j-log4j12-*.jar \
      /usr/lib/hadoop/lib/slf4j-log4j12.jar && \
    rm /usr/lib/hive/lib/hbase-annotations.jar && \
    rm /usr/lib/hive/lib/zookeeper.jar && \
    \
    cd /hivemall && \
    HIVEMALL_VERSION=`cat VERSION` && \
    find /hivemall/resources/docker -mindepth 1 -maxdepth 1 \
      -exec sh -c 'f={} && ln -s $f /root/${f##*/}' \; && \
    ln -s /hivemall/resources/ddl/define-all.hive /root/define-all.hive && \
    ln -s /hivemall/target/hivemall-core-${HIVEMALL_VERSION}-with-dependencies.jar \
      /root/hivemall-core-with-dependencies.jar && \
    \
    (if $PREBUILD; then \
      mvn package -Dmaven.test.skip=true -pl core; \
    fi) && \
    \
    rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*

VOLUME /hivemall/ /root/data/
EXPOSE 19888 50070

CMD ["sh", "-c", "./init.sh && bash"]
