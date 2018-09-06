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

# Approximate Counting using HyperLogLog

`count(distinct value)` can often cause memory exhausted errors where input data and the cardinality of value are large.

[HyperLogLog](https://en.wikipedia.org/wiki/HyperLogLog) is an efficient algorithm for approximating the number of distinct elements in a [multiset](https://en.wikipedia.org/wiki/Multiset). 
Hivemall implements [HyperLogLog++](https://en.wikipedia.org/wiki/HyperLogLog#HLL.2B.2B) in `approx_count_distinct`.

## Usage

`approx_count_distinct` is less accurate than COUNT(DISTINCT expression), but performs better on huge input.

```sql
select
    count(distinct rowid) as actual,
    approx_count_distinct(rowid) as default_p 
from
    train;
```

| actual | default_p |
|:------:|:---------:|
| 45840617 | 45567770 |


```sql
select
    approx_count_distinct(rowid, '-p 4') as p4,
    approx_count_distinct(rowid, '-p 6 -sp 6') as p6_sp6,
    approx_count_distinct(rowid, '-p 14') as p14,
    approx_count_distinct(rowid, '-p 15') as p15,
    approx_count_distinct(rowid, '-p 16') as p16,
    approx_count_distinct(rowid, '-p 24') as p24,
    approx_count_distinct(rowid, '-p 25') as p25,
    approx_count_distinct(rowid, '-p 15 -sp 15') as p15_sp15
from
    train;
```

| p4 | p6_sp6 | p14 | p15 | p16 | p24 | p25 | p15_sp15 |
|:--:|:------:|:---:|:---:|:---:|:---:|:---:|:--------:|
| 38033066 | 49332600 | 45051015 | 45567770 | 45614484 | 45831359 | 45832280 | 45567770 |

> #### Note
>
> `p` controls expected precision and memory consumption tradeoff and `default p=15` generally works well. Find More information on [this paper](https://ai.google/research/pubs/pub40671).

## Function Signature

You can find the function signature and options of `approx_count_distinct` is as follows:

```sql
select 
    approx_count_distinct(rowid, '-help')
from
    train;
```

```
usage: HLLEvaluator [-help] [-p <arg>] [-sp <arg>]
 -help       Show function help
 -p <arg>    The size of registers for the normal set. `p` MUST be in the
             range [4,sp] and 15 by the default
 -sp <arg>   The size of registers for the sparse set. `sp` MUST be in the
             range [4,32] and 25 by the defaul
```
