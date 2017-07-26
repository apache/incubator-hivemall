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

# Training the model

Assume that we already have a table **t_test_maxent** with features and labels in the following format:

| docid | doc  |
|:---:|:---|
| 1  | "Fruits and vegetables are healthy." |
|2 | "I like apples, oranges, and avocados. I do not like the flu or colds." |
| ... | ... |

Training the model requires **train_maxent_classifier**:

```sql
CREATE TABLE tmodel_maxent
STORED AS SEQUENCEFILE
AS
select
train_maxent_classifier(features, klass, "-attrs
Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,Q,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,Q,Q,Q,Q,Q,Q,Q,Q")
from
t_test_maxent;
```

# Predict

```sql
create table tmodel_combined as
select predict_maxent_classifier(b.model, b.attributes, a.features) result, klass from t_test_maxent a join tmodel_maxent b;
```