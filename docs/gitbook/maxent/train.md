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

Once the data is prepared, in this case in a **realtimedata** data. Training the model requires **train_maxent_classifier**:

```sql
CREATE TABLE realtimemodel
STORED AS SEQUENCEFILE
AS
select
  train_maxent_classifier(features, label, "-attrs C,Q,Q")
from
  realtimetrain;
```

# Predict

```sql
CREATE TABLE realtime_predicted AS
SELECT
predict_maxent_classifier(b.model, b.attributes, a.features) result, a.label 
FROM 
realtimetrain a JOIN realtimemodel b;
 
SELECT count(1) 
FROM realtime_predicted 
WHERE result.value == label;
```