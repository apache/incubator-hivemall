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

DOCKCROSS_SCRIPT := 'dockcross.bash'
XGBOOST_REPO := 'https://github.com/maropu/xgboost.git'
XGBOOST_BRANCH := 'xgboost_v0.60_with_portable_binaries'
HIVEMALL_HOME := "$(shell pwd)"
HIVEMALL_OUT := "${HIVEMALL_HOME}/target"
XGBOOST_OUT := "${HIVEMALL_OUT}/xgboost"
HIVEMALL_LIB_DIR := "${HIVEMALL_HOME}/xgboost/src/main/resources/lib"
CANDIDATES := 'linux-arm64' 'linux-armv6' 'linux-armv7' 'linux-ppc64le' 'linux-x64' 'linux-x86' 'windows-x64' 'windows-x86'


.PHONY: phony
phony: ;

.PHONY: clean-xgboost
clean-xgboost:
	rm -rf ${XGBOOST_OUT} ${HIVEMALL_LIB_DIR}

.PHONY: fetch-xgboost
fetch-xgboost: clean-xgboost
	set -eux && \
	mkdir -p ${XGBOOST_OUT} && \
	git clone --depth 1 --single-branch -b ${XGBOOST_BRANCH} ${XGBOOST_REPO} ${XGBOOST_OUT} && \
	cd ${XGBOOST_OUT} && \
	git submodule init && \
	git submodule update

.PHONY: xgboost-native-local
xgboost-native-local: fetch-xgboost
	set -eux && \
	: $${JAVA_HOME} && \
	cd target/xgboost/jvm-packages && \
	export ENABLE_STATIC_LINKS=1 && \
	./create_jni.sh && \
	mkdir -p ${HIVEMALL_LIB_DIR} && \
	cp ${XGBOOST_OUT}/jvm-packages/lib/libxgboost4j.so ${HIVEMALL_LIB_DIR} # TODO: output dir

.PHONY: xgboost-native
xgboost-native: fetch-xgboost
	set -eux && \
	for os_arch in ${CANDIDATES}; do make xgboost-$${os_arch}; done

.PHONY: xgboost-native-%
xgboost-native-%:
	set -eux && \
	OS_ARCH=$(subst xgboost-native-,,$@) && \
	echo ${CANDIDATES} | grep -q $${OS_ARCH} && \
	cd ${HIVEMALL_HOME} && \
	docker run --rm dockcross/$${OS_ARCH} > ${DOCKCROSS_SCRIPT} && \
	chmod +x ${DOCKCROSS_SCRIPT} && \
	./${DOCKCROSS_SCRIPT} sh -c ' \
		sudo apt-get update && \
		sudo apt-get install -y --no-install-recommends openjdk-7-jdk && \
		cd target/xgboost/jvm-packages && \
		export ENABLE_STATIC_LINKS=1 && \
		export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64 && \
		./create_jni.sh' && \
	mkdir -p ${HIVEMALL_LIB_DIR}/$${OS_ARCH} && \
	cp ${XGBOOST_OUT}/jvm-packages/lib/libxgboost4j.so ${HIVEMALL_LIB_DIR}/$${OS_ARCH} && \
	rm -rf ${DOCKCROSS_SCRIPT} \
	|| echo Candidates: ${CANDIDATES}
