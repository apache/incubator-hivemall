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
	- Ensure that all JIRA issues that are addressed in this release are marked with the release version in the `FixVersion` field of the issue.

# Making a release

## Update ChangeLog

1. Create a release note in JIRA following [this instructions](https://confluence.atlassian.com/adminjiraserver071/creating-release-notes-802592502.html).



## Code Validation

	$ mvn apache-rat:check

## Create a branch for release

## Create a Source Release

## Upload the Source Release

## Tag the release

## Staging artifacts in Maven
