FROM openjdk:7

WORKDIR /root/

ARG PREBUILD=true
ARG HADOOP_VERSION=2.7.3
ARG HIVE_VERSION=2.1.1

ENV BASE_URL='https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename='
ENV HADOOP_HOME='/usr/local/hadoop'
ENV HIVE_HOME='/usr/local/hive'
ENV HIVEMALL_PATH='/opt/hivemall'
ENV HADOOP_OPTS=' \
      -Dsystem:java.io.tmpdir=/tmp \
      -Dsystem:user.name=root \
      -Dderby.stream.error.file=/root/derby.log'
ENV PATH="${HADOOP_HOME}/bin:${HIVE_HOME}/bin:${PATH}"

COPY . ${HIVEMALL_PATH}/

RUN set -eux && \
    apt update && \
    apt install -y --no-install-recommends openssh-server maven g++ make ruby npm && \
    ln -s /usr/bin/nodejs /usr/bin/node && \
    npm install -g gitbook-cli && \
    \
    wget ${BASE_URL}hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz -O - \
      | tar xz && \
    mv hadoop-${HADOOP_VERSION} ${HADOOP_HOME} && \
    sed -i -e 's!${JAVA_HOME}!'"${JAVA_HOME}!" ${HADOOP_HOME}/etc/hadoop/hadoop-env.sh && \
    ssh-keygen -q -P '' -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    echo 'host *\n  StrictHostKeyChecking no' > ~/.ssh/config && \
    mv ${HIVEMALL_PATH}/resources/docker/etc/hadoop/*.xml ${HADOOP_HOME}/etc/hadoop && \
    hdfs namenode -format && \
    \
    wget ${BASE_URL}hive/hive-${HIVE_VERSION}/apache-hive-${HIVE_VERSION}-bin.tar.gz -O - \
      | tar xz && \
    mv apache-hive-${HIVE_VERSION}-bin ${HIVE_HOME} && \
    cat ${HIVE_HOME}/conf/hive-default.xml.template \
      | sed -e 's!databaseName=metastore_db!databaseName=/root/metastore_db!' \
      > ${HIVE_HOME}/conf/hive-site.xml && \
    \
    cd ${HIVEMALL_PATH} && \
    HIVEMALL_VERSION=`cat VERSION` && \
    \
    (if ${PREBUILD}; then \
      cd ${HIVEMALL_PATH} && bin/build.sh; \
    fi) && \
    \
    mkdir -p /root/bin /root/hivemall && \
    find ${HIVEMALL_PATH}/resources/docker/home/bin -mindepth 1 -maxdepth 1 \
      -exec sh -c 'f={} && ln -s $f /root/bin/${f##*/}' \; && \
    ln -s ${HIVEMALL_PATH}/resources/docker/home/.hiverc /root && \
    ln -s ${HIVEMALL_PATH}/resources/ddl/define-all.hive /root/hivemall/define-all.hive && \
    ln -s ${HIVEMALL_PATH}/target/hivemall-core-${HIVEMALL_VERSION}-with-dependencies.jar \
      /root/hivemall/hivemall-core-with-dependencies.jar && \
    rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* /root/.m2/* /root/.npm/*

VOLUME ["/opt/hivemall/", "/root/data/"]
EXPOSE 8088 19888 50070

CMD ["sh", "-c", "./bin/init.sh && bash"]
