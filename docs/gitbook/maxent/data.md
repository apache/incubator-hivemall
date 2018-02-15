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

# Data Preparation

This example uses [realTeam.dat](https://github.com/SowaLabs/OpenNLP/blob/master/opennlp-maxent/samples/sports/realTeam.dat) dataset.

The dataset contains football games' results. We try to predict outcome of a game (lose/win/tie) knowing whether the game was played at home or away from home, and two other measurements.

There are 100 observations in the dataset. And the first lines are 

```sh
away pdiff=0.6875 ptwins=0.5 lose
away pdiff=1.0625 ptwins=0.5 win
home pdiff=0.8125 ptwins=0.5 lose
home pdiff=0.9375 ptwins=0.5 win
away pdiff=0.6875 ptwins=0.6666 lose
home pdiff=1.0625 ptwins=0.3333 win
away pdiff=0.8125 ptwins=0.6666 win
home pdiff=0.9375 ptwins=0.3333 win
home pdiff=0.6875 ptwins=0.75 win
away pdiff=1.0625 ptwins=0.25 tie
…
```

That can be transformed easily [into](maxent/realTimeDouble.dat) 

```sh
2.0  0.6875     0.5  0
2.0  1.0625     0.5  2
1.0  0.8125     0.5  0
1.0  0.9375     0.5  2
2.0  0.6875     0.6666     0
1.0  1.0625     0.3333     2
2.0  0.8125     0.6666     2
1.0  0.9375     0.3333     2
1.0  0.6875     0.75 2
2.0  1.0625     0.25 1
…
```

using any tool of choice, for example via Java

```java
public static double place(String outcome) {
    if (outcome.equals("home")) {
        return 1;
    }
    return 2;
}

public static int outcome(String outcome) {
    if (outcome.equals("lose")) {
        return 0;
    } else if (outcome.equals("win")) {
        return 2;
    }
    return 1;
}
```

or a set of consecutive sed commands:

```sh
cat realTime.dat | sed 's/home/1/' | sed 's/away/2/' | sed 's/pdiff=//' | sed 's/ptwins=//' | sed 's/lose/0/' | sed 's/tie/1/' | sed 's/win/2/'
```

or via Hive:

```sh
hadoop fs -mkdir realtime
hadoop fs -put realTime.dat realtime
```

```sql
CREATE EXTERNAL TABLE realtime (location string, pdiff string, ptwins string, result string) ROW FORMAT DELIMITED FIELDS TERMINATED BY ' ' LINES
TERMINATED BY '\n' 
STORED AS TEXTFILE 
LOCATION '/user/<username>/realtime';
```

```sql 
CREATE TABLE realtimedata AS
SELECT 
CASE 
WHEN location == 'home' THEN cast(1 as double) 
WHEN location=='away' THEN cast(2 as double) 
END loc,
cast(regexp_replace(pdiff,'pdiff=','') as double) pdiff,
cast(regexp_replace(ptwins, 'ptwins=','') as double) ptwins,
CASE 
WHEN result == 'lose' THEN 0 
WHEN result == 'win' THEN 2
WHEN result == 'tie' THEN 1 
END result 
FROM realtime;
```

The outcome of the data preparation is a table with an array of doubles representing features and a label (integer):

```sql 
CREATE TABLE realtimetrain AS 
SELECT array(loc, pdiff, ptwins) AS features, result AS label 
FROM realtimedata;
```
 
