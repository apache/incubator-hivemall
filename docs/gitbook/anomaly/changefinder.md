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

In a context of anomaly detection, there are two types of anomalies, ***outlier*** and ***change-point***, as discussed in [this section](sst.md#outlier-vs-change-point). Hivemall has two functions which respectively detect outliers and change-points; the former is [Local Outlier Detection](lof.md), and the latter is [Singular Spectrum Transformation](sst.md).

In some cases, we might want to detect outlier and change-point simultaneously in order to figure out characteristics of a time series both in a local and global scale. **ChangeFinder** is an anomaly detection technique which enables us to detect both of outliers and change-points in a single framework. A key reference for the technique is:

* K. Yamanishi and J. Takeuchi. [A Unifying Framework for Detecting Outliers and Change Points from Non-Stationary Time Series Data](https://dl.acm.org/citation.cfm?id=775148). KDD'02.

<!-- toc -->

# Outlier and Change-Point Detection using ChangeFinder

By using Twitter's time series data we prepared in [this section](sst.md#data-preparation), let us try to use ChangeFinder on Hivemall.

```
use twitter;
```

A function `changefinder()` can be used in a very similar way to `sst()`, a UDF for [Singular Spectrum Transformation](sst.md). The following query detects outliers and change-points with different thresholds:

```sql
SELECT
  num,
  changefinder(value, "-outlier_threshold 0.03 -changepoint_threshold 0.0035") AS result
FROM
  timeseries
ORDER BY num ASC
;
```

As a consequence, finding outliers and change-points in the data points should be easy:

| num | result |
|:---:|:---|
|...|...|
|16  |    {"outlier_score":0.051287243859365894,"changepoint_score":0.003292139657059704,"is_outlier":true,"is_changepoint":false}|
|17  |    {"outlier_score":0.03994335565212781,"changepoint_score":0.003484242549446824,"is_outlier":true,"is_changepoint":false}|
|18  |    {"outlier_score":0.9153515196592132,"changepoint_score":0.0036439645550477373,"is_outlier":true,"is_changepoint":true}|
|19  |    {"outlier_score":0.03940593403992665,"changepoint_score":0.0035825157392152134,"is_outlier":true,"is_changepoint":true}|
|20  |    {"outlier_score":0.27172093630215555,"changepoint_score":0.003542822324886785,"is_outlier":true,"is_changepoint":true}|
|21  |    {"outlier_score":0.006784031454620809,"changepoint_score":0.0035029441620275975,"is_outlier":false,"is_changepoint":true}|
|22  |    {"outlier_score":0.011838969816513334,"changepoint_score":0.003519599336202336,"is_outlier":false,"is_changepoint":true}|
|23  |    {"outlier_score":0.09609857927656007,"changepoint_score":0.003478729798944702,"is_outlier":true,"is_changepoint":false}|
|24  |    {"outlier_score":0.23927000145081978,"changepoint_score":0.0034338476757061237,"is_outlier":true,"is_changepoint":false}|
|25 |     {"outlier_score":0.04645945042821564,"changepoint_score":0.0034052091926036914,"is_outlier":true,"is_changepoint":false}|
|...|...|

# ChangeFinder for Multi-Dimensional Data

ChangeFinder additionally supports multi-dimensional data. Let us try this on synthetic data.

## Data preparation

You first need to get synthetic 5-dimensional data from [HERE](https://github.com/apache/incubator-hivemall/blob/master/core/src/test/resources/hivemall/anomaly/synthetic5d.t.gz?raw=true) and uncompress to a `synthetic5d.t` file:

```
$ head synthetic5d.t
0#71.45185411564131#54.456141290891466#71.78932846605129#76.73002575911214#81.71265594077099
1#58.374230566196786#57.9798651697631#75.65793151143754#73.76101930504493#69.50315805346253
2#66.3595943896099#52.866595973073295#76.7987325026338#78.95890786682095#74.67527753118893
3#58.242560151043236#52.449574430621226#73.20383710416358#77.81502394558085#76.59077723631032
4#55.89878019680371#52.69611781315756#75.02482987204824#74.11154526135637#75.86881583921179
5#56.93554246767561#56.55687136423391#74.4056583421317#73.82419594611444#71.3017150863033
6#65.55704393868689#52.136347983404974#71.14213602046532#72.87394198561904#73.40278960429114
7#56.65735280596217#57.293605941063035#75.36713340281246#80.70254745535183#75.32423746923857
8#61.22095211566127#53.47603728473668#77.48215321523912#80.7760107465893#74.43951386292905
9#52.47574856682803#52.03250504263378#77.59550963025158#76.16623830860391#76.98394610743863
```

The first column indicates a dummy timestamp, and the following four columns are values in each dimension. 

Second, the following Hive operations create a Hive table for the data:

```
create database synthetic;
use synthetic;
```

```sql
CREATE EXTERNAL TABLE synthetic5d (
	num INT,
  value1 DOUBLE,
	value2 DOUBLE,
	value3 DOUBLE,
	value4 DOUBLE,
	value5 DOUBLE
) ROW FORMAT DELIMITED
FIELDS TERMINATED BY '#'
STORED AS TEXTFILE
LOCATION '/dataset/synthetic/synthetic5d';
```

Finally, you can load the synthetic data to the table by:

```
$ hadoop fs -put synthetic5d.t /dataset/synthetic/synthetic5d
```

## Detecting outliers and change-points of the 5-dimensional data

Using `changefinder()` for multi-dimensional data requires us to pass the first argument as an array. In our case, the data is 5-dimensional, so the first argument should be an array with 5 elements. Except for that point, basic usage of the function is same as the previous 1-dimensional example:

```sql
SELECT
  num,
  changefinder(array(value1, value2, value3, value4, value5), 
               "-outlier_threshold 0.015 -changepoint_threshold 0.0045") AS result
FROM
  synthetic5d
ORDER BY num ASC
;
```

Output might be:

| num | result |
|:---:|:---|
|...|...|
|90   |   {"outlier_score":0.014014718350674471,"changepoint_score":0.004520174906936474,"is_outlier":false,"is_changepoint":true}|
|91   |   {"outlier_score":0.013145554693405614,"changepoint_score":0.004480713237042799,"is_outlier":false,"is_changepoint":false}|
|92   |   {"outlier_score":0.011631759675989617,"changepoint_score":0.004442031415725316,"is_outlier":false,"is_changepoint":false}|
|93  |    {"outlier_score":0.012140065235943798,"changepoint_score":0.004404170732687428,"is_outlier":false,"is_changepoint":false}|
|94   |   {"outlier_score":0.012555903663657997,"changepoint_score":0.0043670553008087355,"is_outlier":false,"is_changepoint":false}|
|95   |   {"outlier_score":0.013503247137325314,"changepoint_score":0.0043306667027628466,"is_outlier":false,"is_changepoint":false}|
|96   |   {"outlier_score":0.013896893553710932,"changepoint_score":0.004294969164345527,"is_outlier":false,"is_changepoint":false}|
|97   |   {"outlier_score":0.01322874844578159,"changepoint_score":0.004259994590721001,"is_outlier":false,"is_changepoint":false}|
|98  |    {"outlier_score":0.019383618511936707,"changepoint_score":0.004225604978710543,"is_outlier":true,"is_changepoint":false}|
|99  |    {"outlier_score":0.01121758589038846,"changepoint_score":0.004191881992962213,"is_outlier":false,"is_changepoint":false}|
|...|...|
