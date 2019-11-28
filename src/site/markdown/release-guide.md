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

# Apache Hivemall Release Guide

This document describes the release process of Apache Hivemall particularly for [Release Managers](http://incubator.apache.org/guides/releasemanagement.html#glossary-release-manager).

For general information of the Apache Incubator release process, please refer [Incubator Release Management](http://incubator.apache.org/guides/releasemanagement.html) and [ASF Release Poloicy](http://www.apache.org/dev/#releases) page.

## Prerequisites

If it is your first time doing an Apache release, then there is some initial setup involved. Follow [this guide](./release-setup.html) for the initial setup.

1. Notifying the community the overall plan for the release
2. Ensure JIRA Issues are appropriately tagged for the Release 
    - [Check this link](https://issues.apache.org/jira/browse/HIVEMALL-162?jql=project%20%3D%20HIVEMALL%20AND%20status%20in%20\(Resolved%2C%20Closed\)%20AND%20fixVersion%20%3D%20EMPTY) for `CLOSED/RESOLVED` issues that does not have `FixVersion`.
    - Then, ensure that all JIRA issues that are addressed in this release are marked with the release version in the `FixVersion` field of the issue. [Here](https://issues.apache.org/jira/browse/HIVEMALL-157?jql=project%20%3D%20HIVEMALL%20AND%20status%20in%20\(Resolved%2C%20Closed\)%20AND%20fixVersion%20%3D%200.5.0) is an example for v0.5.0 release.

## UPDATE CHANGELOG

1. Create a release note in JIRA following [this instructions](https://confluence.atlassian.com/adminjiraserver071/creating-release-notes-802592502.html).
2. Download a release note as `ChangeLog.html` using [JIRA](https://issues.apache.org/jira/secure/ConfigureReleaseNote.jspa?projectId=12320630).

## TEST YOUR SETTINGS

Try installing locally artifacts with activation apache-release profile. The following command will build artifacts, sources and sign. 

**Note:** _Apache Hivemall keeps class file compatibility to Java 7 but requires Java 8 for packaging._

```sh
# JDK 8 is required for packaging
export JAVA_HOME=`/usr/libexec/java_home -v 1.8`

# Try to create artifacts
export MAVEN_OPTS=-XX:MaxMetaspaceSize=256m

# (Optional) Workaround for SSL error `Received fatal alert: protocol_version`
export MAVEN_OPTS="$MAVEN_OPTS -Dhttps.protocols=TLSv1,TLSv1.1,TLSv1.2"

mvn -Papache-release clean install
```

Ensure that all unit tests passes. License check by Apache RAT (`mvn apache-rat:check`) will be ran by the above commands.

## SNAPSHOT RELEASE

### PREPARE YOUR POMS FOR RELEASE

**1)** Switch to master syncing to ASF master

```sh
git checkout master
git fetch
git pull # or, git reset --hard asf/master
```

**2)** Set version string for a snapshot

```sh
version=X.Y.Z
# echo "${version}-incubating-SNAPSHOT"

./bin/set_version.sh --pom --version "${version}-incubating-SNAPSHOT"
```

**Note:** _`--pom` option SHOULD only used for SNAPSHOT release._

```sh
git commit -a -m "Prepare for the next Snapshot release of X.Y.Z"
git push asf master
```

**3)** Perform a dryRun

```sh
version=X.Y.Z
# RC should start at 1 and increment if early release candidates fail to release
rc=1
next_version=X.(Y+1).Z

# Confirm that version and rc is defined.
echo "Release version: ${version}-incubating"
echo "Development version: ${next_version}-incubating-SNAPSHOT"

mvn -Papache-release release:prepare \
  -DautoVersionSubmodules=true -DdryRun=true \
  -Darguments='-Dmaven.test.skip.exec=true' -DskipTests=true -Dmaven.test.skip=true \
  -Dtag=v${version}-rc${rc} -DreleaseVersion=${version}-incubating -DdevelopmentVersion=${next_version}-incubating-SNAPSHOT
```

**Note:** `-rcX` is only used for git tag. Release candidates do NOT have `-rcX` for artifact versions in pom. Release candidates are just for voting.

### PUBLISH A SNAPSHOT

```sh
mvn deploy \
  -Darguments='-Dmaven.test.skip.exec=true' -DskipTests=true -Dmaven.test.skip=true
```

**Note:** _You SHOULD verify the deployment under [the Maven Snapshot repository at Apache](https://repository.apache.org/content/repositories/snapshots/org/apache/hivemall/)._

## PREPARE THE RELEASE

### BRANCH THE RELEASE

**1)** Create a branch

```sh
git checkout -b vX.Y.Z
```

**Note:** _Branch name starts with `v` and does not include `-rcX`._

**2)** Send e-mail announcing the release branch

```
To: dev@hivemall.incubator.apache.org
CC: private@hivemall.incubator.apache.org
Subject: New release branch X.Y.Z

Hello Hivemall developers and friends,

We now have a release branch for X.Y.Z release. Trunk has been bumped to X.Y.Z-SNAPSHOT.

  https://github.com/apache/incubator-hivemall/tree/vX.Y.Z

I'll be going over the JIRAs to move every non-blocker from this release to the next release. Release voting will be posted soon.
```

### PREPARE A RELEASE

**1)** Set a release version

```sh
version=X.Y.Z
rc=1
next_version=X.(Y+1).Z

# Confirm that version and rc is defined.
echo "${version}-incubating (Release Candidates ${rc})"
# 0.5.0-incubating (Release Candidates 3)
```

**2)** Update version strings in source codes.

```sh
./bin/set_version.sh --version "${version}-incubating"

# Check list of files to be committed.
git status
git commit -a -m "Bumped version string to ${version}-incubating"
```

**3)** Prepare sets the version numbers in POM and creates a tag.

```sh
mvn -Papache-release release:clean release:prepare \
  -DautoVersionSubmodules=true -DdryRun=false \
  -Darguments='-Dmaven.test.skip.exec=true' -DskipTests=true -Dmaven.test.skip=true \
  -Dtag=v${version}-rc${rc} -DreleaseVersion=${version}-incubating -DdevelopmentVersion=${next_version}-incubating-SNAPSHOT
```

**4)** Update version strings for the development iteration

```sh
./bin/set_version.sh --version "${next_version}-incubating-SNAPSHOT"

# Check list of files to be committed.
git stauts
git commit --amend -a
```

**5)** Push release branch and tag to remote ASF repository

```sh
# Push the release branch
git push asf v${version}

# Push the release tag
git push asf v${version}-rc${rc}
```

## STAGE THE RELEASE FOR A VOTE

The release will automatically be inserted into a temporary staging repository for you.

```sh
mvn -Papache-release release:perform \
  -Darguments='-Dmaven.test.skip.exec=true' -DskipTests=true -Dmaven.test.skip=true \
  -Dgoals=deploy -DlocalRepoDirectory=. -DlocalCheckout=true
```

### Verify nexus release artifacts

1) Verify the staged artifacts in the nexus repo
  - Go to [https://repository.apache.org/](https://repository.apache.org/) and login
  - Under `Build Promotion`, click `Staging Repositories`
  - In the `Staging Repositories` tab there should be a line with profile `org.apache.hivemall`
  - Navigate through the artifact tree and make sure that all javadoc, sources, tests, and jars have .asc (GPG signature). Refer [the ASF page](http://www.apache.org/dev/release-signing.html#verifying-signature) for artifacts verification.

2) Close the nexus staging repo
  - Check the box on in the first column of the row, and press the ‘Close’ button to publish the repository at [https://repository.apache.org/content/repositories/orgapachehivemall-1001/](https://repository.apache.org/content/repositories/orgapachehivemall-1001/) (or a similar URL)

### Upload the artifacts via subversion to a staging area

- Prepare release artifacts in SVN repository

```sh
# Checkout release SVN repository
mkdir -p dist/dev/incubator
cd dist/dev/incubator
svn co https://dist.apache.org/repos/dist/dev/incubator/hivemall/
cd hivemall

# Download release artifacts
wget -e robots=off --no-check-certificate \
 -r -np --reject=html,txt,tmp -nH --cut-dirs=7 \
 https://repository.apache.org/content/repositories/orgapachehivemall-1001/org/apache/hivemall/hivemall/${version}-incubating/

# Rename directory for RC
mv ${version}-incubating ${version}-incubating-rc${rc}

# Put ChangeLog
cd ${version}-incubating-rc${rc}
# Put ChangeLog generated by JIRA
cp ~/Downloads/ChangeLog.html .

# Remove deprecated md5 and sha1
# See https://www.apache.org/dev/release-distribution.html#sigs-and-sums

ls *.md5 *.sha1
rm *.md5 *.sha1

# Add SHA-512 checksum

ls *.xml *.zip *.pom

for f in `ls *.xml *.zip *.pom`; do
  shasum -a 512 ${f} | cut -f1 -d' '  > ${f}.sha512
  ls ${f}.sha512
done
```

- Push release arfifacts to ASF svn repository

```sh
cd dist/dev/incubator/hivemall
# ls ${version}-incubating-rc${rc}

svn add ${version}-incubating-rc${rc}/
svn commit -m "Put hivemall version ${version}-incubating-rc${rc} artifacts"
```

### Verify Release Artifacts

- Check release artifacts are properly deployed in the SVN repository: [https://dist.apache.org/repos/dist/dev/incubator/hivemall/X.Y.Z-incubating-rcZ/](https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc1/)
- Follow [this guide](./verify_artifacts.html) for release artifacts verification.

## PPMC voting

Create a VOTE email thread on [dev@hivemall.incubator.apache.org](mailto:dev@hivemall.incubator.apache.org) to record votes as replies.

```
To: dev@hivemall.incubator.apache.org
CC: private@hivemall.incubator.apache.org
Subject: [VOTE] Release Apache Hivemall (Incubating) v0.5.0 Release Candidate #3

Hi all,

Apache Hivmeall 0.5.0 release candidate #1 is now available for a vote within dev community.

Links to various release artifacts are given below. Please review and cast your vote.

    - The source tarball, including signatures, digests, ChangeLog, etc.:
      https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc1/
	- Sources for the release:
	  https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc1/hivemall-0.5.0-incubating-source-release.zip
	  https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc1/hivemall-0.5.0-incubating-source-release.zip.asc (PGP Signature)
	  https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc1/hivemall-0.5.0-incubating-source-release.zip.sha512 (SHA512 Hash)
    - Git tag for the release:
      https://gitbox.apache.org/repos/asf?p=incubator-hivemall.git;a=shortlog;h=refs/tags/v0.5.0-rc1
    - The Nexus Staging URL:
      https://repository.apache.org/content/repositories/orgapachehivemall-1001/
    - KEYS file for verification:
      https://dist.apache.org/repos/dist/dev/incubator/hivemall/KEYS
    - For information about the contents of this release, see:
      https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc1/ChangeLog.html

Find Podling releases policies in
    https://incubator.apache.org/policy/incubation.html#releases
    http://www.apache.org/legal/release-policy.html

Artifacts verification how-to can be found in
    http://hivemall.incubator.apache.org/verify_artifacts.html

The vote will be open for at least 72 hours and until necessary number of votes are reached.

At lease three +1 from PPMC members are required and we welcome your vote!

[ ] +1  approve (Release this package as Apache Hivemall 0.5.0-incubating)
[ ] +0  no opinion
[ ] -1  disapprove (and reason why)

Here is my +1 (binding).

Regards,
Makoto
```

Once 72 hours has passed (which is generally preferred) and/or at least three +1 (binding) votes have been cast with no -1 (binding) votes, send an email closing the vote and pronouncing the release candidate a success.

```
To: dev@hivemall.incubator.apache.org
Subject: [RESULT][VOTE]: Release Apache Hivemall (Incubating) <release version>

Hi all,

The Apache Hivemall <release version> vote is now closed and has passed as follows:

From PPMC members:
+1 myui (Makoto)
+1 maropu (Takeshi)
+1 lewuathe (Kai)
+1 takuti (Takuya)
+1 ozawa (Tsuyoshi)

From Mentors:
+1 daniel (IPMC)

Other non-binding votes:
+1 Kento (committer)

I will send a vote mail to incubator-general mail list to collect three IPMC +1's votes.
https://incubator.apache.org/policy/incubation.html#releases.

If voting succeeds, v0.5.0 will be released based on v0.5.0-rc2 with removing rc2.

Thanks all of the voters.

Thanks,
Makoto
on behalf of Apache Hivemall PPMC
```

## IPMC Voting

What if vote succeed, then vote in [general@incubator.apache.org](mailto:general@incubator.apache.org). 
Three +1 Incubator PMC votes are required as described in [this page](https://incubator.apache.org/policy/incubation.html#releases).

```
To: general@incubator.apache.org
Subject: [VOTE] Release Apache Hivemall (Incubating) v0.5.0-RC2

Hi all,

The Apache Hivemall community has approved a proposal to release Apache Hivemall v0.5.0 by v0.5.0-RC2.

We now kindly request that the Incubator PMC members review and vote on this incubator release candidate.

The PPMC vote thread is located here:
    https://www.mail-archive.com/dev@hivemall.incubator.apache.org/msg00450.html (vote)
	https://www.mail-archive.com/dev@hivemall.incubator.apache.org/msg00460.html (vote result)

Links to various release artifacts are given below.

    - The source tarball, including signatures, digests, ChangeLog, etc.:
      https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc2/
    - Sources for the release:
      https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc2/hivemall-0.5.0-incubating-rc2-source-release.zip
    - Git tag for the release:
      https://gitbox.apache.org/repos/asf?p=incubator-hivemall.git;a=shortlog;h=refs/tags/v0.5.0-rc2
    - The Nexus Staging URL:
      https://repository.apache.org/content/repositories/orgapachehivemall-1002/
    - KEYS file for verification:
      https://dist.apache.org/repos/dist/dev/incubator/hivemall/KEYS
    - For information about the contents of this release, see:
      https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.5.0-incubating-rc2/ChangeLog.html

Artifacts verification how-to can be found in
    http://hivemall.incubator.apache.org/verify_artifacts.html

Please vote accordingly:

[ ] +1  approve (Release this package as Apache Hivemall 0.5.0-incubating)
[ ] -1  disapprove (and reason why)

The vote will be open for at least 72 hours.

Regards,
Makoto
on behalf of Apache Hivemall PPMC
```

Once 72 hours has passed (which is generally preferred) and/or at least three +1 (binding) votes have been cast with no -1 (binding) votes, send an email closing the vote and pronouncing the release candidate a success.

```
To: general@incubator.apache.org
CC: dev@hivemall.incubator.apache.org
Subject: [RESULT][VOTE]: Release Apache Hivemall (Incubating) <release version>

Hi all,

The vote to release Apache Hivemall (incubating) version v0.5.0
Release Candidate 3 has passed with 3 +1 binding votes and no other votes.

+1 binding:
 ..
 ..
 ..

Vote thread:
 https://lists.apache.org/thread.html/86a0d8df057c6c7b0675f8bbdb211fe6602545c68de2bf9b7b8204c1@%3Cgeneral.incubator.apache.org%3E

I'd like to thank all voters who helped verifying the release.

The Apache Hivemall community will proceed to publish the 0.5.0-incubating release and send announcements.

Best,
Makoto
on behalf of Apache Hivemall PPMC
```

## Finalize release

Once the Incubator general vote has passed, we can publish the source release and artifacts.

### Publish to Maven Central in Nexus

- Go to https://repository.apache.org/ and login
- Under Build Promotion, click Staging Repositories
- In the Staging Repositories tab there should be a line with profile org.apache.hivemall
- click `Release`.
	- You shall now see your artifacts in the release repository created for you. In some hours, it should also appear in Maven Central.

### Update JIRA

Update the JIRA versions page to close all issues, mark the version as `released`, and set the date to the date that the release was approved. You may also need to make a new release entry for the next release.

### Source Release

The Hivemall source release are checked into SVN under: [https://dist.apache.org/repos/dist/release/incubator/hivemall](https://dist.apache.org/repos/dist/release/incubator/hivemall)

```sh
# cd ~/work/dist
# mkdir -p release/incubator
# cd release/incubator
# svn co https://dist.apache.org/repos/dist/release/incubator/hivemall/
# cd hivemall

# Put release artifacts to release directory
mkdir 0.5.0-incubating
cp ../../../dev/incubator/hivemall/0.5.0-incubating-rc3/ChangeLog.html 0.5.0-incubating/
cp ../../../dev/incubator/hivemall/0.5.0-incubating-rc3/hivemall-0.5.0-incubating*.zip* 0.5.0-incubating/
cp ../../../dev/incubator/hivemall/0.5.0-incubating-rc3/hivemall*.jar* 0.5.0-incubating/
ls 0.5.0-incubating/

# Commit to SVN
svn add 0.5.0-incubating/
svn commit -m "Put hivemall version 0.5.0-incubating release artifacts"
```

Check release artifacts in the SVN repository:
 [https://dist.apache.org/repos/dist/release/incubator/hivemall/0.5.0-incubating/](https://dist.apache.org/repos/dist/release/incubator/hivemall/0.5.0-incubating/) 
 ([Mirror](https://www.apache.org/dyn/closer.cgi/incubator/hivemall))

### Merge release branch for the next development iteration

If IPMC vote succeed, then merge the release branch into the master branch.

### Publish the websit

Update [download page](http://hivemall.incubator.apache.org/download.html) etc.

### Announcing the release

Make an announcement about the release on the `user@hivemall.incubator.apache.org`, `dev@hivemall.incubator.apache.org`, `general@incubator.apache.org`, and `announce@apache.org` list as per the Apache Announcement Mailing Lists page.

```
[ANNOUNCE] Apache Hivemall 0.5.0-incubating released

The Apache Hivemall (incubating) project team is proud to announce
Apache Hivemall 0.5.2-incubating has been released.
This is the 2nd Apache release as an Apache Incubator project.

Apache Hivemall is a scalable machine learning library implemented as
Hive UDFs/UDAFs/UDTFs. Hivemall runs on Hadoop-based data processing
frameworks, specifically on Apache Hive, Apache Spark, and Apache Pig.

The release artifacts and ChangeLog can be found at:

 http://hivemall.incubator.apache.org/download.html

Release artifacts in Maven Central:

 https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.apache.hivemall%22%20AND%20v%3A%220.5.2-incubating%22

Find more about our project at:
 - Project Site:  http://hivemall.incubator.apache.org/
 - Github mirror: https://github.com/apache/incubator-hivemall
 - Mailing list(s): dev@hivemall.incubator.apache.org
                    user@hivemall.incubator.apache.org

Thanks,
Makoto
on behalf of Apache Hivemall PPMC

=====
*Disclaimer*
Apache Hivemall (incubating) is an effort undergoing incubation at
The Apache Software Foundation (ASF), sponsored by the name of Apache
Incubator PMC. Incubation is required of all newly accepted projects
until a further review indicates that the infrastructure, communications,
and decision making process have stabilized in a manner consistent
with other successful ASF projects. While incubation status is not
necessarily a reflection of the completeness or stability of the code,
it does indicate that the project has yet to be fully endorsed by the ASF.
```
