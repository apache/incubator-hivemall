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

# Download data

Get dataset of [Kaggle Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) from one of the following sources:

1. [Original competition data](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) (by Criteo Labs) [~20GB]
2. [Subset of the original competition data](http://labs.criteo.com/2014/02/dataset/) (by Criteo Labs) [~30MB]
3. [Tiny sample data](https://github.com/guestwalk/kaggle-2014-criteo) (by the winners of the competition) [~20bytes]

It should be noted that you must accept and agree with **CRITEO LABS DATA TERM OF USE** before downloading the data.

# Convert data into CSV format

Here, you can use a script prepared by one of the Hivemall PPMC members: **[takuti/criteo-ffm](https://github.com/takuti/criteo-ffm)**.

Clone the repository:

```sh
git clone git@github.com:takuti/criteo-ffm.git
cd criteo-ffm
```

A script [`data.sh`](https://github.com/takuti/criteo-ffm/blob/master/data.sh) downloads the original data and converts them into CSV format:

```sh
./data.sh  # downloads the original data and generates `train.csv` and `test.csv`
ln -s train.csv tr.csv
ln -s test.csv te.csv
```

Or, since the original data is very huge, starting from the tiny sample data bundled into the repository would be better:

```sh
ln -s train.tiny.csv tr.csv
ln -s test.tiny.csv te.csv
```

# Create tables

Load the CSV files to Hive tables as:

```sh
hadoop fs -put tr.csv /criteo/train
hadoop fs -put te.csv /criteo/test
```

```sql
CREATE DATABASE IF NOT EXISTS criteo;
use criteo;
```

```sql
DROP TABLE IF EXISTS train;
CREATE EXTERNAL TABLE train (
  id bigint,
  label int,
  -- quantitative features
  i1 int,i2 int,i3 int,i4 int,i5 int,i6 int,i7 int,i8 int,i9 int,i10 int,i11 int,i12 int,i13 int,
  -- categorical features
  c1 string,c2 string,c3 string,c4 string,c5 string,c6 string,c7 string,c8 string,c9 string,c10 string,c11 string,c12 string,c13 string,c14 string,c15 string,c16 string,c17 string,c18 string,c19 string,c20 string,c21 string,c22 string,c23 string,c24 string,c25 string,c26 string
) ROW FORMAT
DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE LOCATION '/criteo/train';
```

```sql
DROP TABLE IF EXISTS test;
CREATE EXTERNAL TABLE test (
  label int,
  -- quantitative features
  i1 int,i2 int,i3 int,i4 int,i5 int,i6 int,i7 int,i8 int,i9 int,i10 int,i11 int,i12 int,i13 int,
  -- categorical features
  c1 string,c2 string,c3 string,c4 string,c5 string,c6 string,c7 string,c8 string,c9 string,c10 string,c11 string,c12 string,c13 string,c14 string,c15 string,c16 string,c17 string,c18 string,c19 string,c20 string,c21 string,c22 string,c23 string,c24 string,c25 string,c26 string
) ROW FORMAT
DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE LOCATION '/criteo/test';
```