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

# What's One-hot encoding?

Ont-hot encoding is a method to encode categorical features by a 1-of-K (thus called 1-hot) encoding scheme.

Suppose the following table:

| Company | Price |
|:-:|:-:|
| VW | 290 |
| Toyota | 300 |
| Honda | 190 |
| Honda | 250 |

A one-hot encoding output is expected as follows:

| Company_VW | Company_Toyota | Company_Honda | Price |
|:-:|:-:|:-:|:-:|
| 1 | 0 | 0 | 290 |
| 0 | 1 | 0 | 300 |
| 0 | 0 | 1 | 190 |
| 0 | 0 | 1 | 250 |

The above one-hot table is a dense feature format and it can be expressed as follows by a sparse format:

| Company | Price |
|:-:|:-:|
| {1} | 290 |
| {2} | 300 |
| {3} | 190 |
| {3} | 250 |

The mapping for company name is {VW->1, Toyota->2, Honda->3}.

Now, suppose encoding two categorical variables as follows into a sparse vector.

| category1 | category2 |
|:-:|:-:|
|cat | mammal |
|dog |mammal |
|human | mammal |
|seahawk | bird |
|wasp | insect |
|wasp | insect |
|cat | mammal |
|dog | mammal |
|human | mammal |

The one-hot encoded feature vector could be as follows:

| category1 | category2 | encoded_features |
|:-:|:-:|:-:|
| cat | mammal | {1,6} |
| dog | mammal | {2,6} |
| human | mammal | {3,6} |
| seahawk | bird | {4,7} |
| wasp | insect | {5,8} |

We use this `test` table for explaration.

```sql
drop table test;
create table test (species string, category string, count int);

truncate table test;
insert into table test values
  ('cat','mammal',9), 
  ('dog','mammal',10),
  ('human','mammal',10),
  ('seahawk','bird',101),
  ('wasp','insect',3),
  ('wasp','insect',9),
  ('cat','mammal',101),
  ('dog','mammal',1),
  ('human','mammal',9);
```

# One-hot encoding table

You can get one-hot encoding table for spieces as follows:

```sql
WITH t as (
  select onehot_encoding(species) m
  from test
)
select m.f1 from t;
```

| f1 |
|:-|
| {"seahawk":1,"cat":2,"human":3,"wasp":4,"dog":5} |

```sql
WITH t as (
  select onehot_encoding(species, category) m
  from test
)
select m.f1, m.f2 from t;
```

| f1 | f2 |
| {"seahawk":1,"cat":2,"human":3,"wasp":4,"dog":5} | {"bird":6,"insect":7,"mammal":8} |

You can create a mapping table as follows:

```sql
create table mapping as
WITH t as (
  select onehot_encoding(species, category) m
  from test
)
select m.f1, m.f2 from t;

desc mapping;

col_name    | data_type
------------|----------------
f1          | map<string,int>                             
f2          | map<string,int>   
```

# How to use One-hot encoding

The following query applies one-hot encoding using the mapping table.

```sql
select
  t.species, m.f1[t.species],
  t.category, m.f2[t.category]
from
  test t
  CROSS JOIN mapping m;

cat 2   mammal  8 
dog 5   mammal  8 
human   3   mammal  8 
seahawk 1   bird    6 
wasp    4   insect  7 
wasp    4   insect  7 
cat 2   mammal  8 
dog 5   mammal  8
human   3   mammal  8
```

You can create a sparse feature vector as follows:

```sql
select
  array(m.f1[t.species],m.f2[t.category],feature('count',count)) as sparse_feature 
from
  test t
  CROSS JOIN mapping m;

sparse_feature
["2","8","count:9"]
["5","8","count:10"]
["3","8","count:10"]
["1","6","count:101"]
["4","7","count:3"]
["4","7","count:9"]
["2","8","count:101"]
["5","8","count:1"]
["3","8","count:9"]
```

It also can be achieved by a single query as follows:

```sql
WITH mapping as (
  select 
    m.f1, m.f2 
  from (
    select onehot_encoding(species, category) m
    from test
  ) tmp
)
select
  array(m.f1[t.species],m.f2[t.category],feature('count',count)) as sparse_features
from
  test t
  CROSS JOIN mapping m;
```

Note that one-hot encoding is required only for categorical variables. Feature hasing is another scalable way to encode categorical variables to numerical index.