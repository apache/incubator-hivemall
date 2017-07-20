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
        
This page introduces how to find change-points using **Singular Spectrum Transformation** (SST) on Hivemall. The following papers describe the details of this technique:

* T. Idé and K. Inoue. [Knowledge Discovery from Heterogeneous Dynamic Systems using Change-Point Correlations](http://epubs.siam.org/doi/abs/10.1137/1.9781611972757.63). SDM'05.
* T. Idé and K. Tsuda. [Change-Point Detection using Krylov Subspace Learning](http://epubs.siam.org/doi/abs/10.1137/1.9781611972771.54). SDM'07.

<!-- toc -->

# Outlier vs Change-Point

It is important that anomaly detectors are generally categorized into outlier and change-point detectors. Outliers are some spiky "local" data points which are suddenly observed in a series of normal samples, and [Local Outlier Detection](lof.md) is an algorithm to detect outliers. On the other hand, change-points indicate "global" change on a wider scale in terms of characteristics of data points.

In this page, we specially focus on change-point detection. More concretely, the following sections introduce a way to detect change-points on Hivemall, by using a specific technique named Singular Spectrum Transformation (SST).

# Data Preparation

## Get Twitter's data

We use time series data points provided by Twitter in the following article: [Introducing practical and robust anomaly detection in a time series](https://blog.twitter.com/2015/introducing-practical-and-robust-anomaly-detection-in-a-time-series). In fact, the dataset is originally created for R, but we can get CSV version of the same data from [HERE](https://github.com/apache/incubator-hivemall/blob/master/core/src/test/resources/hivemall/anomaly/twitter.csv.gz?raw=true).

Once you uncompressed the downloaded `.gz` file, you can see a CSV file:

```
$ head twitter.csv
182.478
176.231
183.917
177.798
165.469
181.878
184.502
183.303
177.578
171.641
```

These values are sequential data points. Our goal is to detect change-points in the samples. Here, let us insert a dummy timestamp into each line as follows:

```
$ awk '{printf "%d#%s\n", NR, $0}' < twitter.csv > twitter.t
```

```
$ head twitter.t
1#182.478
2#176.231
3#183.917
4#177.798
5#165.469
6#181.878
7#184.502
8#183.303
9#177.578
10#171.641
```

Now, Hive can understand sequence of the samples by just looking dummy timestamp.

## Importing data as a Hive table

### Create a Hive table

You first need to launch a Hive console and run the following operations:

```
create database twitter;
use twitter;
```

```sql
CREATE EXTERNAL TABLE timeseries (
  num INT,
  value DOUBLE
) ROW FORMAT DELIMITED
FIELDS TERMINATED BY '#'
STORED AS TEXTFILE
LOCATION '/dataset/twitter/timeseries';
```

### Load data into the table

Next, the `.t` file we have generated before can be loaded to the table by:

```
$ hadoop fs -put twitter.t /dataset/twitter/timeseries
```

`timeseries` table in `twitter` database should be:

| num | value |
|:---:|:---:|
|1|182.478|
|2|176.231|
|3|183.917|
|4|177.798|
|5|165.469|
|...|...|

# Change-Point Detection using SST

We are now ready to detect change-points. A UDF `sst()` takes a `double` value as the first argument, and you can set options in the second argument. 

What the following query does is to detect change-points from a `value` column in the `timeseries` table. An option `"-threshold 0.005"` means that a data point is detected as a change-point if its score is greater than 0.005.

```
use twitter;
```

```sql
SELECT
  num,
  sst(value, "-threshold 0.005") AS result
FROM
  timeseries
ORDER BY num ASC
;
```

For instance, partial outputs obtained as a result of this query are:

| num | result |
|:---:|:---|
|...|...|
|7551  |  {"changepoint_score":0.00453049288071683,"is_changepoint":false}|
|7552 |   {"changepoint_score":0.004711244102524104,"is_changepoint":false}|
|7553  |  {"changepoint_score":0.004814871928978115,"is_changepoint":false}|
|7554 |   {"changepoint_score":0.004968089640799422,"is_changepoint":false}|
|7555 |   {"changepoint_score":0.005709056330104878,"is_changepoint":true}|
|7556   | {"changepoint_score":0.0044279766655132,"is_changepoint":false}|
|7557  |  {"changepoint_score":0.0034694956722586268,"is_changepoint":false}|
|7558  |  {"changepoint_score":0.002549056569322694,"is_changepoint":false}|
|7559  |  {"changepoint_score":0.0017395109108403473,"is_changepoint":false}|
|7560  |  {"changepoint_score":0.0010629833145070489,"is_changepoint":false}|
|...|...|

Obviously, the 7555-th sample is detected as a change-point in this example.
