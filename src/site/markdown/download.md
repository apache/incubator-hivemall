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

# Release History

Detailed ChangeLog of each version can be found in the JIRA link.

| RELEASE | DATE | DESCRIPTION | COMMIT | DOWNLOAD |
|:--------|:-----|:------------|:-------|:---------|
| 0.5.0 | [2018-03-05](https://markmail.org/thread/imnf6azzxksrbdg4) | The first Apache release. There are tremendous changes since [v0.4.2-rc.2](https://github.com/myui/hivemall/releases/tag/v0.4.2-rc.2) which is the last release before entering Apache Incubator. ([ChangeLog](http://www.apache.org/dist/incubator/hivemall/0.5.0-incubating/ChangeLog.html)) | [9610fdd](https://github.com/apache/incubator-hivemall/commit/9610fdd93628defa735ea8ba23703d0836bbe2f1) | [src.zip](http://www.apache.org/dyn/closer.cgi/incubator/hivemall/0.5.0-incubating/) [signs](http://www.apache.org/dist/incubator/hivemall/0.5.0-incubating/)<br/>[jars](https://search.maven.org/#search%7Cga%7C1%7Cg%3A%22org.apache.hivemall%22%20AND%20v%3A%220.5.0-incubating%22) [DDL](https://github.com/apache/incubator-hivemall/tree/v0.5.0/resources/ddl)|
| .. | .. | .. | .. | .. |

Past releases of Apache Hivemall can be found in [this page](https://github.com/myui/hivemall/releases).

# Release plan

Here is the release plan of Apache Hivemall. Date of release subject to change though.

| Version | Date       | Description |
|:-------:|:----------:|:-----------|
| [0.5.2](https://issues.apache.org/jira/issues/?jql=project+%3D+HIVEMALL+AND+fixVersion+%3D+0.5.2)   | 2018-05-xx | • Merge [Brickhouse UDFs](https://issues.apache.org/jira/browse/HIVEMALL-145)<br/> • Support [Field-aware Factorization Machines](https://issues.apache.org/jira/browse/HIVEMALL-24)<br/> • Support [Word2Vec](https://issues.apache.org/jira/browse/HIVEMALL-118) |
| [0.6](https://issues.apache.org/jira/issues/?jql=project%20%3D%20HIVEMALL%20AND%20fixVersion%20%3D%200.6.0)     | 2018-08-xx | • Make [experimental xgboost support](https://github.com/apache/incubator-hivemall/pull/95) official <br/> • Support [Multi-nominal Logistic Regression](https://github.com/apache/incubator-hivemall/pull/93)<br/> and more |
| 0.7     | 2018-11-xx | • Prediction server with REST APIs |
