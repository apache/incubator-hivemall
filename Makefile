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
XGBOOST_HOME := "${HIVEMALL_OUT}/xgboost"
XGBOOST_OUT := "${XGBOOST_HOME}/target"
XGBOOST_CLASSES := "${XGBOOST_OUT}/classes"
HIVEMALL_LIB_DIR := "${HIVEMALL_HOME}/xgboost/src/main/resources/lib"
CANDIDATES := 'linux-arm64' 'linux-ppc64le' 'linux-x64' 'windows-x64'


.PHONY: phony
phony: ;

.PHONY: clean-xgboost
clean-xgboost:
	rm -rf ${XGBOOST_HOME}

.PHONY: clean-all-xgboost
clean-all-xgboost: clean-xgboost
	rm -rf ${HIVEMALL_LIB_DIR}

.PHONY: fetch-xgboost
fetch-xgboost: clean-xgboost
	set -eux && \
	mkdir -p ${XGBOOST_HOME} && \
	git clone --depth 1 --single-branch -b ${XGBOOST_BRANCH} --recursive ${XGBOOST_REPO} ${XGBOOST_HOME}

.PHONY: xgboost-native-local
xgboost-native-local: fetch-xgboost
	set -eux && \
	: $${JAVA_HOME} && \
	mkdir -p ${XGBOOST_CLASSES} && \
	$${JAVA_HOME}/bin/javac -d ${XGBOOST_CLASSES} xgboost/src/main/java/hivemall/xgboost/OSInfo.java && \
	OS=`$${JAVA_HOME}/bin/java -cp ${XGBOOST_CLASSES} hivemall.xgboost.OSInfo --os` && \
	ARCH=`$${JAVA_HOME}/bin/java -cp ${XGBOOST_CLASSES} hivemall.xgboost.OSInfo --arch` && \
	OS_ARCH=$${OS}-$${ARCH} && \
	cd ${XGBOOST_HOME}/jvm-packages && \
	export ENABLE_STATIC_LINKS=1 && \
	./create_jni.sh && \
	mkdir -p ${HIVEMALL_LIB_DIR}/$${OS_ARCH} && \
	cp ${XGBOOST_HOME}/jvm-packages/lib/libxgboost4j.so ${HIVEMALL_LIB_DIR}/$${OS_ARCH}

.PHONY: xgboost-native
xgboost-native:
	set -eux && \
	for os_arch in ${CANDIDATES}; do make xgboost-native-$${os_arch}; done

.PHONY: xgboost-native-%
xgboost-native-%: fetch-xgboost
	set -eux && \
	OS_ARCH=$(subst xgboost-native-,,$@) && \
	echo ${CANDIDATES} | grep -q $${OS_ARCH} && \
	docker run --rm dockcross/$${OS_ARCH} > ${DOCKCROSS_SCRIPT} && \
	chmod +x ${DOCKCROSS_SCRIPT} && \
	./${DOCKCROSS_SCRIPT} sh -c " \
		sudo apt-get update && \
		sudo apt-get install -y --no-install-recommends openjdk-7-jdk && \
		cd target/xgboost/jvm-packages && \
		sed -i -e 's/CXX=g++//' create_jni.sh && \
		[ ! $${OS_ARCH##*-} = 'x64' ] && \
			for f in '../Makefile' '../dmlc-core/Makefile' '../rabit/Makefile'; do sed -i -e 's/-m64//g' -e 's/-msse2//g' \$${f}; done || \
		export ENABLE_STATIC_LINKS=1 && \
		export JAVA_HOME=/usr/lib/jvm/java-7-openjdk-amd64 && \
		./create_jni.sh" && \
	mkdir -p ${HIVEMALL_LIB_DIR}/$${OS_ARCH} && \
	cp ${XGBOOST_HOME}/jvm-packages/lib/libxgboost4j.so ${HIVEMALL_LIB_DIR}/$${OS_ARCH} && \
	rm -rf ${DOCKCROSS_SCRIPT} \
	|| echo Candidates: ${CANDIDATES}
