<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

<!-- toc -->

# 1. Preparation 

## Install required softwares 

GPG and Maven, JDK 8 is mandatory for verification.

```sh
brew install gpg gpg-agent pinentry-mac
brew install maven
brew install md5sha1sum
```

## Import GPG KEYS

```sh
# Download GPG KEYS
wget https://dist.apache.org/repos/dist/dev/incubator/hivemall/KEYS

# import KEYS for GPG verification
gpg --import KEYS
```

# 2. Download all release artifacts

```sh
VERSION=0.5.0
RC_NUMBER=3

wget -e robots=off --no-check-certificate \
 -r -np --reject=html,txt,tmp -nH --cut-dirs=5 \
 https://dist.apache.org/repos/dist/dev/incubator/hivemall/${VERSION}-incubating-rc${RC_NUMBER}/
```

# 3. Verify SHA512, and GPG signatures.

```sh
cd ${VERSION}-incubating-rc${RC_NUMBER}/

for f in `find . -type f -iname '*.sha512'`; do
  echo -n "Verifying ${f%.*} ... "
  shasum -a 512 ${f%.*} | cut -f1 -d' ' | diff -Bw - ${f}
  if [ $? -eq 0 ]; then
    echo 'Valid'
  else 
    echo "SHA512 is Invalid: ${f}" >&2
    exit 1
  fi  
done
echo
for f in `find . -type f -iname '*.asc'`; do
  gpg --verify ${f}
  if [ $? -eq 0 ]; then
    echo "GPG signature is correct: ${f%.*}"
  else
    echo "GPG signature is Invalid: ${f%.*}" >&2
	exit 1
  fi
  echo
done
```

# 4. Build, Test, and Verify source 

```sh
unzip hivemall-${VERSION}-incubating-source-release.zip
cd hivemall-${VERSION}-incubating

# workaround for Maven sign-release-artifacts plugin
export GPG_TTY=$(tty)

# JDK 8 is required for packaging
export JAVA_HOME=`/usr/libexec/java_home -v 1.8`

# (Optional) TO avoid JVM errors in unit tests
export MAVEN_OPTS=-XX:MaxMetaspaceSize=256m

# (Optional) Workaround for SSL error `Received fatal alert: protocol_version`
export MAVEN_OPTS="$MAVEN_OPTS -Dhttps.protocols=TLSv1,TLSv1.1,TLSv1.2"

# (Optional) Workaround for Surefire error:
# Could not find or load main class org.apache.maven.surefire.booter.ForkedBooter
export _JAVA_OPTIONS="-Djdk.net.URLClassPath.disableClassPathURLCheck=true"

# Try to create artifacts
# RAT license check and unit tests will be issued
mvn -Papache-release clean install

# Verify Signatures of Release Artifacts
cd target/
for file in `find . -type f -iname '*.asc'`; do
  gpg --verify ${file}
done
```