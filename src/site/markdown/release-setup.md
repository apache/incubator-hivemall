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

# Release Prerequisites

This document describes the initial setup procedures for making a release of Apache Hivemall.

If it is your first time doing an Apache release, then there is some initial setup involved. You should read this page before proceeding. 

## Software

You would need the following softwares for building and making a release:

- gpg2
- md5sum, sha1sum
- git client
- svn client (_svn is still mandatory in the ASF distribution process. Don't ask me why._)
- JDK 7 (_not JDK 8 nor 9 to support Java 7 or later_)
- maven (>=v3.3.1)

### Installation

	# on Mac
	$ brew install gpg gpg-agent pinentry-mac
	$ brew instal md5sha1sum
	$ brew intall svn
	$ brew install maven
	$ brew install npm
	$ npm install gitbook-cli -g

### Java 7 setup (Optional)

We recommend to use [jEnv](http://www.jenv.be/) for Java 7 environment setup on Mac OS X.

	$ /usr/libexec/java_home -v 1.7
	  /Library/Java/JavaVirtualMachines/jdk1.7.0_80.jdk/Contents/Home
	$ jenv add `/usr/libexec/java_home -v 1.7`
	$ jenv versions
	* system (set by /Users/myui/.jenv/version)
	  1.7
	  1.7.0.80
	  oracle64-1.7.0.80
	
	# configure to use Java 7 for Hivemall
	$ cd incubator-hivemall
	$ jenv local 1.7
	$ java -version
	  java version "1.7.0_80"

## PGP signing

To perform a release, you need to add signatures to release packages.
See the [Signing Releases](http://www.apache.org/dev/release-signing.html) page for information on how to do this.

In a nutshell, you'll need to follow the instructions at [How To OpenPGP](http://www.apache.org/dev/openpgp.html#generate-key) to generate a new code signing key and publish the public key in various places.

### Setting up signing keys

1). Generate a key-pair with gpg using [this instruction](http://www.apache.org/dev/openpgp.html#key-gen-generate-key). The program's default values should be fine. Please use a signing key with an ASF email address (i.e. your-alias@apache.org). Generated Keys should be RSA with at least 4096 bits.

	$ gpg --full-generate-key

Here is my key.

	$ gpg --list-key --keyid-format LONG
	
	pub   rsa4096/93F4D08DC8CE801B 2017-11-01 [SC]
	      7A6BA1A10CC6ABF47159152193F4D08DC8CE801B
	uid                 [ultimate] Makoto Yui (CODE SIGNING KEY) <myui@apache.org>
	sub   rsa4096/C3F1C8E219A64221 2017-11-01 [E]

Public key is `93F4D08DC8CE801B` in the above case.

2). Send your public PGP key to a public keyserver.

	$ gpg --keyserver pgp.mit.edu --send-keys <your-public-pgp-key>

3). Update the PGP key fingerprint of your account on [id.apache.org](http://id.apache.org). Find your PGP key fingerprint by 

	$ gpg --fingerprint

4). Update KEYS file in the git repo to your public key be listed in it.

	$ export YOUR_NAME="Makoto Yui"
	$ (gpg --list-sigs ${YOUR_NAME} && gpg --armor --export ${YOUR_NAME} && echo) >> KEYS
	
	# Update git repository
	$ git add KEYS
	$ git commit -m "Added the public key of YOUR NAME"
	$ git push origin master

5). Add your public key to KEYS file in the subversion repository:

	- dev: https://dist.apache.org/repos/dist/dev/hivemall/KEYS
	- release: https://dist.apache.org/repos/dist/release/hivemall/KEYS

	# checkout dist repos
	$ svn co --depth immediates https://dist.apache.org/repos/dist dist
	$ cd dist
	$ svn up --set-depth infinity dev/incubator/hivemall
	$ svn up --set-depth infinity release/incubator/hivemall
	
	# edit KEYS files
	$ svn add * --force
	$ svn ci -m "Updated KEYS file of Incubator Hivemall" && svn up

6). Once you have followed these instructions, you should have:

	* Your public key viewable at https://people.apache.org/keys/committer/your-asf-id.asc
	* Your public key also viewable at https://people.apache.org/keys/group/hivemall.asc

#### Configure PGP signing on git (optional)

After completing this, you should also [configure git](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work) to use your key for signing.

If your signing key is identified by `01234567`, then you can configure git with:

    $ git config --global user.signingkey 01234567

If you are using gpg, you'll then need to tell git to use it.

    $ git config --global gpg.program gpg

You can enable GPG signing always true on the particular git repository by:

	$ git config commit.gpgsign true

To sign all commits by default in any local repository on your computer, run

	$ git config --global commit.gpgsign true

Use pientry to omit typing a passphrase for each commit.

	$ echo "pinentry-program /usr/local/bin/pinentry-mac" >> ~/.gnupg/gpg-agent.conf
	$ echo -e "use-agent\nno-tty" >> ~/.gnupg/gpg.conf

Tips: You may get an error about a passphrase not being provided when signing with git.  If this happens try running the command below, which should case the passphrase prompt to show in the terminal.

    export GPG_TTY=`tty`

## Configure Maven for publishing artifacts

Update your `~/.m2/settings.xml` following [this instructions](http://www.apache.org/dev/publishing-maven-artifacts.html#dev-env).

```xml
<settings>
...
  <servers>
    <!-- To publish a snapshot of some part of Maven -->
    <server>
      <id>apache.snapshots.https</id>
      <username> <!-- YOUR APACHE LDAP USERNAME --> </username>
      <password> <!-- YOUR APACHE LDAP PASSWORD (encrypted) --> </password>
    </server>
    <!-- To stage a release of some part of Maven -->
    <server>
      <id>apache.releases.https</id>
      <username> <!-- YOUR APACHE LDAP USERNAME --> </username>
      <password> <!-- YOUR APACHE LDAP PASSWORD (encrypted) --> </password>
    </server>
   ...
  </servers>
</settings>
```