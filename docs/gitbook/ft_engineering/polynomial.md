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

[Polynomial features](http://en.wikipedia.org/wiki/Polynomial_kernel) allows you to do [non-linear regression](https://class.coursera.org/ml-005/lecture/23)/classification with a linear model.

> #### Caution
>
> Polynomial Features assumes normalized inputs because `x**n` easily becomes INF/-INF where `n` is large.

# Polynomial Features

As [a similar to one in Scikit-Learn](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html), `polynomial_feature(array<String> features, int degree [, boolean interactionOnly=false, boolean truncate=true])` is a function to generate polynomial and interaction features.

```sql
select polynomial_features(array("a:0.5","b:0.2"), 2);
> ["a:0.5","a^a:0.25","a^b:0.1","b:0.2","b^b:0.040000003"]

select polynomial_features(array("a:0.5","b:0.2"), 3);
> ["a:0.5","a^a:0.25","a^a^a:0.125","a^a^b:0.05","a^b:0.1","a^b^b:0.020000001","b:0.2","b^b:0.040000003","b^b^b:0.008"]

-- interaction only
select polynomial_features(array("a:0.5","b:0.2"), 3, true);
> ["a:0.5","a^b:0.1","b:0.2"]

select polynomial_features(array("a:0.5","b:0.2","c:0.3"), 3, true);
> ["a:0.5","a^b:0.1","a^b^c:0.030000001","a^c:0.15","b:0.2","b^c:0.060000002","c:0.3"]

-- interaction only + no truncate
select polynomial_features(array("a:0.5","b:1.0", "c:0.3"), 3, true, false);
> ["a:0.5","a^b:0.5","a^b^c:0.15","a^c:0.15","b:1.0","b^c:0.3","c:0.3"]

-- interaction only + truncate
select polynomial_features(array("a:0.5","b:1.0","c:0.3"), 3, true, true);
> ["a:0.5","a^c:0.15","b:1.0","c:0.3"]

-- truncate
select polynomial_features(array("a:0.5","b:1.0", "c:0.3"), 3, false, true);
> ["a:0.5","a^a:0.25","a^a^a:0.125","a^a^c:0.075","a^c:0.15","a^c^c:0.045","b:1.0","c:0.3","c^c:0.09","c^c^c:0.027000003"]

-- do not truncate
select polynomial_features(array("a:0.5","b:1.0", "c:0.3"), 3, false, false);
> ["a:0.5","a^a:0.25","a^a^a:0.125","a^a^b:0.25","a^a^c:0.075","a^b:0.5","a^b^b:0.5","a^b^c:0.15","a^c:0.15","a^c^c:0.045","b:1.0","b^b:1.0","b^b^b:1.0","b^b^c:0.3","b^c:0.3","b^c^c:0.09","c:0.3","c^c:0.09","c^c^c:0.027000003"]
> 
```

_Note: `truncate` is used to eliminate unnecessary combinations._

# Powered Features

The `powered_features(array<String> features, int degree [, boolean truncate=true] )` is a function to generate polynomial features.

```sql
select powered_features(array("a:0.5","b:0.2"), 3);
> ["a:0.5","a^2:0.25","a^3:0.125","b:0.2","b^2:0.040000003","b^3:0.008"]
```
