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

Apache Hivemall: Hive scalable machine learning library
=======================================================
[![Build Status](https://travis-ci.org/apache/incubator-hivemall.svg?branch=master)](https://travis-ci.org/apache/incubator-hivemall)
[![Documentation Status](https://img.shields.io/:docs-latest-green.svg)](http://hivemall.incubator.apache.org/userguide/)
[![License](http://img.shields.io/:license-Apache_v2-blue.svg)](https://github.com/apache/incubator-hivemall/blob/master/LICENSE)
[![Coverage Status](https://coveralls.io/repos/github/apache/incubator-hivemall/badge.svg?branch=master)](https://coveralls.io/github/apache/incubator-hivemall?branch=master)
[![Twitter Follow](https://img.shields.io/twitter/follow/ApacheHivemall.svg?style=social&label=Follow)](https://twitter.com/ApacheHivemall)

[Apache Hivemall](http://hivemall.incubator.apache.org/) is a scalable machine learning library that runs on Apache Hive, Apache Spark, and Apache Pig. Hivemall is designed to be scalable to the number of training instances as well as the number of training features.

[<img src="src/site/resources/images/apache-incubator-logo.png" alt="Apache Incubator" width=200>](http://hivemall.incubator.apache.org/)

Usage
-----

[![Hivemall](https://gist.githubusercontent.com/myui/d29241262f9313dec706/raw/caead313efd829b42a4a4183285e8b53cf26ab62/hadoopsummit14_slideshare.png)](http://www.slideshare.net/myui/dots20161029-myui/11)

Find more examples on [our user guide](http://hivemall.incubator.apache.org/userguide/index.html) and find a brief introduction to Hivemall in [this slide](http://www.slideshare.net/myui/hadoopsummit16-myui).

Support
-------

Support is through [user@hivemall.incubator.apache.org](http://hivemall.incubator.apache.org/mail-lists.html), not by a direct e-mail.

Contributing
------------

If you are planning to contribute to this repository, we first request you to create an issue at [our JIRA page](https://issues.apache.org/jira/projects/HIVEMALL) even if the topic is not related to source code itself (e.g., documentation, new idea and proposal).

All Hivemall functions are defined under [resources/ddl](resources/ddl). In order to update the definition files, the following script helps inserting function name and class path of your new UDF:

```
$ ./bin/update_ddls.sh
```

Moreover, don't forget to update function list in the document as well:

```
$ ./bin/update_func_md.sh
```

Note that, before creating a pull request including Java code, please make sure your code follows our coding conventions by applying formatter:

```
$ ./bin/format_code.sh
```
