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

This page describes a list of Hivemall functions. See also a [list of generic Hivemall functions](./generic_funcs.md) for more general-purpose functions such as array and map UDFs.

<!-- toc -->

# Regression

- `train_arow_regr(array<int|bigint|string> features, float target [, constant string options])` - a standard AROW (Adaptive Reguralization of Weight Vectors) regressor that uses `y - w^Tx` for the loss function.
  ```sql
  SELECT 
    feature,
    argmin_kld(weight, covar) as weight
  FROM (
    SELECT 
       train_arow_regr(features,label) as (feature,weight,covar)
    FROM 
       training_data
   ) t 
  GROUP BY feature
  ```
Reference: <a href="https://papers.nips.cc/paper/3848-adaptive-regularization-of-weight-vectors.pdf" target="_blank">K. Crammer, A. Kulesza, and M. Dredze, "Adaptive Regularization of Weight Vectors", In Proc. NIPS, 2009.</a><br/>
- `train_arowe2_regr(array<int|bigint|string> features, float target [, constant string options])` - a refined version of AROW (Adaptive Reguralization of Weight Vectors) regressor that usages adaptive epsilon-insensitive hinge loss `|w^t - y| - epsilon * stddev` for the loss function
  ```sql
  SELECT 
    feature,
    argmin_kld(weight, covar) as weight
  FROM (
    SELECT 
       train_arowe2_regr(features,label) as (feature,weight,covar)
    FROM 
       training_data
   ) t 
  GROUP BY feature
  ```

- `train_arowe_regr(array<int|bigint|string> features, float target [, constant string options])` - a refined version of AROW (Adaptive Reguralization of Weight Vectors) regressor that usages epsilon-insensitive hinge loss `|w^t - y| - epsilon` for the loss function
  ```sql
  SELECT 
    feature,
    argmin_kld(weight, covar) as weight
  FROM (
    SELECT 
       train_arowe_regr(features,label) as (feature,weight,covar)
    FROM 
       training_data
   ) t 
  GROUP BY feature
  ```

- `train_pa1_regr(array<int|bigint|string> features, float target [, constant string options])` - PA-1 regressor that returns a relation consists of `(int|bigint|string) feature, float weight`.
  ```sql
  SELECT 
   feature,
   avg(weight) as weight
  FROM 
   (SELECT 
       train_pa1_regr(features,label) as (feature,weight)
    FROM 
       training_data
   ) t 
  GROUP BY feature
  ```
Reference: <a href="http://jmlr.csail.mit.edu/papers/volume7/crammer06a/crammer06a.pdf" target="_blank">Koby Crammer et.al., Online Passive-Aggressive Algorithms. Journal of Machine Learning Research, 2006.</a><br/>
- `train_pa1a_regr(array<int|bigint|string> features, float target [, constant string options])` - Returns a relation consists of `(int|bigint|string) feature, float weight`.

- `train_pa2_regr(array<int|bigint|string> features, float target [, constant string options])` - Returns a relation consists of `(int|bigint|string) feature, float weight`.

- `train_pa2a_regr(array<int|bigint|string> features, float target [, constant string options])` - Returns a relation consists of `(int|bigint|string) feature, float weight`.

- `train_regressor(list<string|int|bigint> features, double label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight&gt;
  ```
  Build a prediction model by a generic regressor
  ```

# Classification

## Binary classification

- `kpa_predict(@Nonnull double xh, @Nonnull double xk, @Nullable float w0, @Nonnull float w1, @Nonnull float w2, @Nullable float w3)` - Returns a prediction value in Double

- `train_arow(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight, float covar&gt;
  ```
  Build a prediction model by Adaptive Regularization of Weight Vectors (AROW) binary classifier
  ```
Reference: <a href="https://papers.nips.cc/paper/3848-adaptive-regularization-of-weight-vectors.pdf" target="_blank">K. Crammer, A. Kulesza, and M. Dredze, "Adaptive Regularization of Weight Vectors", In Proc. NIPS, 2009.</a><br/>
- `train_arowh(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight, float covar&gt;
  ```
  Build a prediction model by AROW binary classifier using hinge loss
  ```

- `train_classifier(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight&gt;
  ```
  Build a prediction model by a generic classifier
  ```

- `train_cw(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight, float covar&gt;
  ```
  Build a prediction model by Confidence-Weighted (CW) binary classifier
  ```

- `train_kpa(array<string|int|bigint> features, int label [, const string options])` - returns a relation &lt;h int, hk int, float w0, float w1, float w2, float w3&gt;

- `train_pa(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight&gt;
  ```
  Build a prediction model by Passive-Aggressive (PA) binary classifier
  ```

- `train_pa1(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight&gt;
  ```
  Build a prediction model by Passive-Aggressive 1 (PA-1) binary classifier
  ```

- `train_pa2(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight&gt;
  ```
  Build a prediction model by Passive-Aggressive 2 (PA-2) binary classifier
  ```

- `train_perceptron(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight&gt;
  ```
  Build a prediction model by Perceptron binary classifier
  ```

- `train_scw(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight, float covar&gt;
  ```
  Build a prediction model by Soft Confidence-Weighted (SCW-1) binary classifier
  ```

- `train_scw2(list<string|int|bigint> features, int label [, const string options])` - Returns a relation consists of &lt;string|int|bigint feature, float weight, float covar&gt;
  ```
  Build a prediction model by Soft Confidence-Weighted 2 (SCW-2) binary classifier
  ```

## Multiclass classification

- `train_multiclass_arow(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight, float covar&gt;
  ```
  Build a prediction model by Adaptive Regularization of Weight Vectors (AROW) multiclass classifier
  ```

- `train_multiclass_arowh(list<string|int|bigint> features, int|string label [, const string options])` - Returns a relation consists of &lt;int|string label, string|int|bigint feature, float weight, float covar&gt;
  ```
  Build a prediction model by Adaptive Regularization of Weight Vectors (AROW) multiclass classifier using hinge loss
  ```

- `train_multiclass_cw(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight, float covar&gt;
  ```
  Build a prediction model by Confidence-Weighted (CW) multiclass classifier
  ```

- `train_multiclass_pa(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight&gt;
  ```
  Build a prediction model by Passive-Aggressive (PA) multiclass classifier
  ```

- `train_multiclass_pa1(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight&gt;
  ```
  Build a prediction model by Passive-Aggressive 1 (PA-1) multiclass classifier
  ```

- `train_multiclass_pa2(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight&gt;
  ```
  Build a prediction model by Passive-Aggressive 2 (PA-2) multiclass classifier
  ```

- `train_multiclass_perceptron(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight&gt;
  ```
  Build a prediction model by Perceptron multiclass classifier
  ```

- `train_multiclass_scw(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight, float covar&gt;
  ```
  Build a prediction model by Soft Confidence-Weighted (SCW-1) multiclass classifier
  ```

- `train_multiclass_scw2(list<string|int|bigint> features, {int|string} label [, const string options])` - Returns a relation consists of &lt;{int|string} label, {string|int|bigint} feature, float weight, float covar&gt;
  ```
  Build a prediction model by Soft Confidence-Weighted 2 (SCW-2) multiclass classifier
  ```

# Matrix factorization

- `bprmf_predict(List<Float> Pu, List<Float> Qi[, double Bi])` - Returns the prediction value

- `mf_predict(array<double> Pu, array<double> Qi[, double Bu, double Bi[, double mu]])` - Returns the prediction value

- `train_bprmf(INT user, INT posItem, INT negItem [, String options])` - Returns a relation &lt;INT i, FLOAT Pi, FLOAT Qi [, FLOAT Bi]&gt;

- `train_mf_adagrad(INT user, INT item, FLOAT rating [, CONSTANT STRING options])` - Returns a relation consists of &lt;int idx, array&lt;float&gt; Pu, array&lt;float&gt; Qi [, float Bu, float Bi [, float mu]]&gt;

- `train_mf_sgd(INT user, INT item, FLOAT rating [, CONSTANT STRING options])` - Returns a relation consists of &lt;int idx, array&lt;float&gt; Pu, array&lt;float&gt; Qi [, float Bu, float Bi [, float mu]]&gt;

# Factorization machines

- `ffm_predict(float Wi, array<float> Vifj, array<float> Vjfi, float Xi, float Xj)` - Returns a prediction value in Double

- `fm_predict(Float Wj, array<float> Vjf, float Xj)` - Returns a prediction value in Double

- `train_ffm(array<string> x, double y [, const string options])` - Returns a prediction model

- `train_fm(array<string> x, double y [, const string options])` - Returns a prediction model

# Recommendation

- `train_slim( int i, map<int, double> r_i, map<int, map<int, double>> topKRatesOfI, int j, map<int, double> r_j [, constant string options])` - Returns row index, column index and non-zero weight value of prediction model

# Anomaly detection

- `changefinder(double|array<double> x [, const string options])` - Returns outlier/change-point scores and decisions using ChangeFinder. It will return a tuple &lt;double outlier_score, double changepoint_score [, boolean is_anomaly [, boolean is_changepoint]]

- `sst(double|array<double> x [, const string options])` - Returns change-point scores and decisions using Singular Spectrum Transformation (SST). It will return a tuple &lt;double changepoint_score [, boolean is_changepoint]&gt;

# Topic modeling

- `lda_predict(string word, float value, int label, float lambda[, const string options])` - Returns a list which consists of &lt;int label, float prob&gt;

- `plsa_predict(string word, float value, int label, float prob[, const string options])` - Returns a list which consists of &lt;int label, float prob&gt;

- `train_lda(array<string> words[, const string options])` - Returns a relation consists of &lt;int topic, string word, float score&gt;

- `train_plsa(array<string> words[, const string options])` - Returns a relation consists of &lt;int topic, string word, float score&gt;

# Preprocessing

- `add_bias(feature_vector in array<string>)` - Returns features with a bias in array&lt;string&gt;

- `add_feature_index(ARRAY[DOUBLE]: dense feature vector)` - Returns a feature vector with feature indices

- `extract_feature(feature_vector in array<string>)` - Returns features in array&lt;string&gt;

- `extract_weight(feature_vector in array<string>)` - Returns the weights of features in array&lt;string&gt;

- `feature(<string|int|long|short|byte> feature, <number> value)` - Returns a feature string

- `feature_index(feature_vector in array<string>)` - Returns feature indices in array&lt;index&gt;

- `sort_by_feature(map in map<int,float>)` - Returns a sorted map

## Data amplification

- `amplify(const int xtimes, *)` - amplify the input records x-times

- `rand_amplify(const int xtimes [, const string options], *)` - amplify the input records x-times in map-side

## Feature binning

- `build_bins(number weight, const int num_of_bins[, const boolean auto_shrink = false])` - Return quantiles representing bins: array&lt;double&gt;

- `feature_binning(array<features::string> features, map<string, array<number>> quantiles_map)` - returns a binned feature vector as an array&lt;features::string&gt; _FUNC_(number weight, array&lt;number&gt; quantiles) - returns bin ID as int
  ```sql
  WITH extracted as (
    select 
      extract_feature(feature) as index,
      extract_weight(feature) as value
    from
      input l
      LATERAL VIEW explode(features) r as feature
  ),
  mapping as (
    select
      index, 
      build_bins(value, 5, true) as quantiles -- 5 bins with auto bin shrinking
    from
      extracted
    group by
      index
  ),
  bins as (
     select 
      to_map(index, quantiles) as quantiles 
     from
      mapping
  )
  select
    l.features as original,
    feature_binning(l.features, r.quantiles) as features
  from
    input l
    cross join bins r

  > ["name#Jacob","gender#Male","age:20.0"] ["name#Jacob","gender#Male","age:2"]
  > ["name#Isabella","gender#Female","age:20.0"]    ["name#Isabella","gender#Female","age:2"]
  ```

## Feature format conversion

- `conv2dense(int feature, float weight, int nDims)` - Return a dense model in array&lt;float&gt;

- `quantify(boolean output, col1, col2, ...)` - Returns an identified features

- `to_dense_features(array<string> feature_vector, int dimensions)` - Returns a dense feature in array&lt;float&gt;

- `to_libsvm_format(array<string> feautres [, double/integer target, const string options])` - Returns a string representation of libsvm
  ```sql
  Usage:
   select to_libsvm_format(array('apple:3.4','orange:2.1'))
   > 6284535:3.4 8104713:2.1
   select to_libsvm_format(array('apple:3.4','orange:2.1'), '-features 10')
   > 3:2.1 7:3.4
   select to_libsvm_format(array('7:3.4','3:2.1'), 5.0)
   > 5.0 3:2.1 7:3.4
  ```

- `to_sparse_features(array<float> feature_vector)` - Returns a sparse feature in array&lt;string&gt;

## Feature hashing

- `array_hash_values(array<string> values, [string prefix [, int numFeatures], boolean useIndexAsPrefix])` returns hash values in array&lt;int&gt;

- `feature_hashing(array<string> features [, const string options])` - returns a hashed feature vector in array&lt;string&gt;
  ```sql
  select feature_hashing(array('aaa:1.0','aaa','bbb:2.0'), '-libsvm');
   ["4063537:1.0","4063537:1","8459207:2.0"]

  select feature_hashing(array('aaa:1.0','aaa','bbb:2.0'), '-features 10');
   ["7:1.0","7","1:2.0"]

  select feature_hashing(array('aaa:1.0','aaa','bbb:2.0'), '-features 10 -libsvm');
   ["1:2.0","7:1.0","7:1"]

  ```

- `mhash(string word)` returns a murmurhash3 INT value starting from 1

- `prefixed_hash_values(array<string> values, string prefix [, boolean useIndexAsPrefix])` returns array&lt;string&gt; that each element has the specified prefix

- `sha1(string word [, int numFeatures])` returns a SHA-1 value

## Feature paring

- `feature_pairs(feature_vector in array<string>, [, const string options])` - Returns a relation &lt;string i, string j, double xi, double xj&gt;

- `polynomial_features(feature_vector in array<string>)` - Returns a feature vectorhaving polynomial feature space

- `powered_features(feature_vector in array<string>, int degree [, boolean truncate])` - Returns a feature vector having a powered feature space

## Ranking

- `bpr_sampling(int userId, List<int> posItems [, const string options])`- Returns a relation consists of &lt;int userId, int itemId&gt;

- `item_pairs_sampling(array<int|long> pos_items, const int max_item_id [, const string options])`- Returns a relation consists of &lt;int pos_item_id, int neg_item_id&gt;

- `populate_not_in(list items, const int max_item_id [, const string options])`- Returns a relation consists of &lt;int item&gt; that item does not exist in the given items

## Feature scaling

- `l1_normalize(ftvec string)` - Returned a L1 normalized value

- `l2_normalize(ftvec string)` - Returned a L2 normalized value

- `rescale(value, min, max)` - Returns rescaled value by min-max normalization

- `zscore(value, mean, stddev)` - Returns a standard score (zscore)

## Feature selection

- `chi2(array<array<number>> observed, array<array<number>> expected)` - Returns chi2_val and p_val of each columns as &lt;array&lt;double&gt;, array&lt;double&gt;&gt;

- `snr(array<number> features, array<int> one-hot class label)` - Returns Signal Noise Ratio for each feature as array&lt;double&gt;

## Feature transformation and vectorization

- `add_field_indices(array<string> features)` - Returns arrays of string that field indices (&lt;field&gt;:&lt;feature&gt;)* are augmented

- `binarize_label(int/long positive, int/long negative, ...)` - Returns positive/negative records that are represented as (..., int label) where label is 0 or 1

- `categorical_features(array<string> featureNames, feature1, feature2, .. [, const string options])` - Returns a feature vector array&lt;string&gt;

- `ffm_features(const array<string> featureNames, feature1, feature2, .. [, const string options])` - Takes categorical variables and returns a feature vector array&lt;string&gt; in a libffm format &lt;field&gt;:&lt;index&gt;:&lt;value&gt;

- `indexed_features(double v1, double v2, ...)` - Returns a list of features as array&lt;string&gt;: [1:v1, 2:v2, ..]

- `onehot_encoding(PRIMITIVE feature, ...)` - Compute onehot encoded label for each feature
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

  ["2","8","count:9"]
  ["5","8","count:10"]
  ["1","6","count:101"]
  ```

- `quantified_features(boolean output, col1, col2, ...)` - Returns an identified features in a dense array&lt;double&gt;

- `quantitative_features(array<string> featureNames, feature1, feature2, .. [, const string options])` - Returns a feature vector array&lt;string&gt;

- `vectorize_features(array<string> featureNames, feature1, feature2, .. [, const string options])` - Returns a feature vector array&lt;string&gt;

# Geospatial functions

- `haversine_distance(double lat1, double lon1, double lat2, double lon2, [const boolean mile=false])`::double - return distance between two locations in km [or miles] using `haversine` formula
  ```sql
  Usage: select latlon_distance(lat1, lon1, lat2, lon2) from ...
  ```

- `lat2tiley(double lat, int zoom)`::int - Returns the tile number of the given latitude and zoom level

- `lon2tilex(double lon, int zoom)`::int - Returns the tile number of the given longitude and zoom level

- `map_url(double lat, double lon, int zoom [, const string option])` - Returns a URL string
  ```
  OpenStreetMap: http://tile.openstreetmap.org/${zoom}/${xtile}/${ytile}.png
  Google Maps: https://www.google.com/maps/@${lat},${lon},${zoom}z
  ```

- `tile(double lat, double lon, int zoom)`::bigint - Returns a tile number 2^2n where n is zoom level. _FUNC_(lat,lon,zoom) = xtile(lon,zoom) + ytile(lat,zoom) * 2^zoom
  ```
  refer https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames for detail
  ```

- `tilex2lon(int x, int zoom)`::double - Returns longitude of the given tile x and zoom level

- `tiley2lat(int y, int zoom)`::double - Returns latitude of the given tile y and zoom level

# Distance measures

- `angular_distance(ftvec1, ftvec2)` - Returns an angular distance of the given two vectors
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    angular_distance(l.features, r.features) as distance,
    distance2similarity(angular_distance(l.features, r.features)) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    distance asc;

  doc1    doc2    distance        similarity
  1       3       0.31678355      0.75942624
  1       2       0.33333337      0.75
  2       3       0.09841931      0.91039914
  2       1       0.33333337      0.75
  3       2       0.09841931      0.91039914
  3       1       0.31678355      0.75942624
  ```

- `cosine_distance(ftvec1, ftvec2)` - Returns a cosine distance of the given two vectors
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    cosine_distance(l.features, r.features) as distance,
    distance2similarity(cosine_distance(l.features, r.features)) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    distance asc;

  doc1    doc2    distance        similarity
  1       3       0.45566893      0.6869694
  1       2       0.5     0.6666667
  2       3       0.04742068      0.95472616
  2       1       0.5     0.6666667
  3       2       0.04742068      0.95472616
  3       1       0.45566893      0.6869694
  ```

- `euclid_distance(ftvec1, ftvec2)` - Returns the square root of the sum of the squared differences: sqrt(sum((x - y)^2))
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    euclid_distance(l.features, r.features) as distance,
    distance2similarity(euclid_distance(l.features, r.features)) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    distance asc;

  doc1    doc2    distance        similarity
  1       2       2.4494898       0.28989795
  1       3       2.6457512       0.2742919
  2       3       1.0     0.5
  2       1       2.4494898       0.28989795
  3       2       1.0     0.5
  3       1       2.6457512       0.2742919
  ```

- `hamming_distance(integer A, integer B)` - Returns Hamming distance between A and B
  ```sql
  select 
    hamming_distance(0,3) as c1, 
    hamming_distance("0","3") as c2 -- 0=0x00, 3=0x11
  ;

  c1      c2
  2       2
  ```

- `jaccard_distance(integer A, integer B [,int k=128])` - Returns Jaccard distance between A and B
  ```sql
  select 
    jaccard_distance(0,3) as c1, 
    jaccard_distance("0","3") as c2, -- 0=0x00, 0=0x11
    jaccard_distance(0,4) as c3
  ;

  c1      c2      c3
  0.03125 0.03125 0.015625
  ```

- `kld(double mu1, double sigma1, double mu2, double sigma2)` - Returns KL divergence between two distributions

- `manhattan_distance(list x, list y)` - Returns sum(|x - y|)
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    manhattan_distance(l.features, r.features) as distance,
    distance2similarity(angular_distance(l.features, r.features)) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    distance asc;

  doc1    doc2    distance        similarity
  1       2       4.0     0.75
  1       3       5.0     0.75942624
  2       3       1.0     0.91039914
  2       1       4.0     0.75
  3       2       1.0     0.91039914
  3       1       5.0     0.75942624
  ```

- `minkowski_distance(list x, list y, double p)` - Returns sum(|x - y|^p)^(1/p)
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    minkowski_distance(l.features, r.features, 1) as distance1, -- p=1 (manhattan_distance)
    minkowski_distance(l.features, r.features, 2) as distance2, -- p=2 (euclid_distance)
    minkowski_distance(l.features, r.features, 3) as distance3, -- p=3
    manhattan_distance(l.features, r.features) as manhattan_distance,
    euclid_distance(l.features, r.features) as euclid_distance
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    distance1 asc;

  doc1    doc2    distance1       distance2       distance3       manhattan_distance      euclid_distance
  1       2       4.0     2.4494898       2.1544347       4.0     2.4494898
  1       3       5.0     2.6457512       2.2239802       5.0     2.6457512
  2       3       1.0     1.0     1.0     1.0     1.0
  2       1       4.0     2.4494898       2.1544347       4.0     2.4494898
  3       2       1.0     1.0     1.0     1.0     1.0
  3       1       5.0     2.6457512       2.2239802       5.0     2.6457512
  ```

- `popcnt(a [, b])` - Returns a popcount value
  ```sql
  select 
    popcnt(3),
    popcnt("3"),  -- 3=0x11
    popcnt(array(1,3));

  2       2       3
  ```

# Locality-sensitive hashing

- `bbit_minhash(array<> features [, int numHashes])` - Returns a b-bits minhash value

- `minhash(ANY item, array<int|bigint|string> features [, constant string options])` - Returns n different k-depth signatures (i.e., clusterid) for each item &lt;clusterid, item&gt;

- `minhashes(array<> features [, int numHashes, int keyGroup [, boolean noWeight]])` - Returns minhash values

# Similarity measures

- `angular_similarity(ftvec1, ftvec2)` - Returns an angular similarity of the given two vectors
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    angular_similarity(l.features, r.features) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    similarity desc;

  doc1    doc2    similarity
  1       3       0.68321645
  1       2       0.6666666
  2       3       0.9015807
  2       1       0.6666666
  3       2       0.9015807
  3       1       0.68321645
  ```

- `cosine_similarity(ftvec1, ftvec2)` - Returns a cosine similarity of the given two vectors
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    cosine_similarity(l.features, r.features) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    similarity desc;

  doc1    doc2    similarity
  1       3       0.5443311
  1       2       0.5
  2       3       0.9525793
  2       1       0.5
  3       2       0.9525793
  3       1       0.5443311
  ```

- `dimsum_mapper(array<string> row, map<int col_id, double norm> colNorms [, const string options])` - Returns column-wise partial similarities

- `distance2similarity(float d)` - Returns 1.0 / (1.0 + d)

- `euclid_similarity(ftvec1, ftvec2)` - Returns a euclid distance based similarity, which is `1.0 / (1.0 + distance)`, of the given two vectors
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    euclid_similarity(l.features, r.features) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    similarity desc;

  doc1    doc2    similarity
  1       2       0.28989795
  1       3       0.2742919
  2       3       0.5
  2       1       0.28989795
  3       2       0.5
  3       1       0.2742919
  ```

- `jaccard_similarity(A, B [,int k])` - Returns Jaccard similarity coefficient of A and B
  ```sql
  WITH docs as (
    select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features
    union all
    select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
    union all
    select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features
  ) 
  select
    l.docid as doc1,
    r.docid as doc2,
    jaccard_similarity(l.features, r.features) as similarity
  from 
    docs l
    CROSS JOIN docs r
  where
    l.docid != r.docid
  order by 
    doc1 asc,
    similarity desc;

  doc1    doc2    similarity
  1       2       0.14285715
  1       3       0.0
  2       3       0.6
  2       1       0.14285715
  3       2       0.6
  3       1       0.0
  ```

# Evaluation

- `auc(array rankItems | double score, array correctItems | int label [, const int recommendSize = rankItems.size ])` - Returns AUC

- `average_precision(array rankItems, array correctItems [, const int recommendSize = rankItems.size])` - Returns MAP

- `f1score(array[int], array[int])` - Return a F1 score

- `fmeasure(array|int|boolean actual, array|int| boolean predicted [, const string options])` - Return a F-measure (f1score is the special with beta=1.0)

- `hitrate(array rankItems, array correctItems [, const int recommendSize = rankItems.size])` - Returns HitRate

- `logloss(double predicted, double actual)` - Return a Logrithmic Loss

- `mae(double predicted, double actual)` - Return a Mean Absolute Error

- `mrr(array rankItems, array correctItems [, const int recommendSize = rankItems.size])` - Returns MRR

- `mse(double predicted, double actual)` - Return a Mean Squared Error

- `ndcg(array rankItems, array correctItems [, const int recommendSize = rankItems.size])` - Returns nDCG

- `precision_at(array rankItems, array correctItems [, const int recommendSize = rankItems.size])` - Returns Precision

- `r2(double predicted, double actual)` - Return R Squared (coefficient of determination)

- `recall_at(array rankItems, array correctItems [, const int recommendSize = rankItems.size])` - Returns Recall

- `rmse(double predicted, double actual)` - Return a Root Mean Squared Error

# Sketching

- `approx_count_distinct(expr x [, const string options])` - Returns an approximation of count(DISTINCT x) using HyperLogLogPlus algorithm

- `bloom(string key)` - Constructs a BloomFilter by aggregating a set of keys
  ```sql
  CREATE TABLE satisfied_movies AS 
    SELECT bloom(movieid) as movies
    FROM (
      SELECT movieid
      FROM ratings
      GROUP BY movieid
      HAVING avg(rating) >= 4.0
    ) t;
  ```

- `bloom_and(string bloom1, string bloom2)` - Returns the logical AND of two bloom filters
  ```sql
  SELECT bloom_and(bf1, bf2) FROM xxx;
  ```

- `bloom_contains(string bloom, string key)` or _FUNC_(string bloom, array&lt;string&gt; keys) - Returns true if the bloom filter contains all the given key(s). Returns false if key is null.
  ```sql
  WITH satisfied_movies as (
    SELECT bloom(movieid) as movies
    FROM (
      SELECT movieid
      FROM ratings
      GROUP BY movieid
      HAVING avg(rating) >= 4.0
    ) t
  )
  SELECT
    l.rating,
    count(distinct l.userid) as cnt
  FROM
    ratings l 
    CROSS JOIN satisfied_movies r
  WHERE
    bloom_contains(r.movies, l.movieid) -- includes false positive
  GROUP BY 
    l.rating;

  l.rating        cnt
  1       1296
  2       2770
  3       5008
  4       5824
  5       5925
  ```

- `bloom_contains_any(string bloom, string key)` or _FUNC_(string bloom, array&lt;string&gt; keys)- Returns true if the bloom filter contains any of the given key
  ```sql
  WITH data1 as (
    SELECT explode(array(1,2,3,4,5)) as id
  ),
  data2 as (
    SELECT explode(array(1,3,5,6,8)) as id
  ),
  bloom as (
    SELECT bloom(id) as bf
    FROM data1
  )
  SELECT 
    l.* 
  FROM 
    data2 l
    CROSS JOIN bloom r
  WHERE
    bloom_contains_any(r.bf, array(l.id))
  ```

- `bloom_not(string bloom)` - Returns the logical NOT of a bloom filters
  ```sql
  SELECT bloom_not(bf) FROM xxx;
  ```

- `bloom_or(string bloom1, string bloom2)` - Returns the logical OR of two bloom filters
  ```sql
  SELECT bloom_or(bf1, bf2) FROM xxx;
  ```

# Ensemble learning

- `argmin_kld(float mean, float covar)` - Returns mean or covar that minimize a KL-distance among distributions
  ```
  The returned value is (1.0 / (sum(1.0 / covar))) * (sum(mean / covar)
  ```

- `max_label(double value, string label)` - Returns a label that has the maximum value

- `maxrow(ANY compare, ...)` - Returns a row that has maximum value in the 1st argument

## Bagging

- `voted_avg(double value)` - Returns an averaged value by bagging for classification

- `weight_voted_avg(expr)` - Returns an averaged value by considering sum of positive/negative weights

# Decision trees and RandomForest

- `train_gradient_tree_boosting_classifier(array<double|string> features, int label [, string options])` - Returns a relation consists of &lt;int iteration, int model_type, array&lt;string&gt; pred_models, double intercept, double shrinkage, array&lt;double&gt; var_importance, float oob_error_rate&gt;

- `train_randomforest_classifier(array<double|string> features, int label [, const string options, const array<double> classWeights])`- Returns a relation consists of &lt;string model_id, double model_weight, string model, array&lt;double&gt; var_importance, int oob_errors, int oob_tests&gt;

- `train_randomforest_regressor(array<double|string> features, double target [, string options])` - Returns a relation consists of &lt;int model_id, int model_type, string model, array&lt;double&gt; var_importance, double oob_errors, int oob_tests&gt;

- `decision_path(string modelId, string model, array<double|string> features [, const string options] [, optional array<string> featureNames=null, optional array<string> classNames=null])` - Returns a decision path for each prediction in array&lt;string&gt;
  ```sql
  SELECT
    t.passengerid,
    decision_path(m.model_id, m.model, t.features, '-classification')
  FROM
    model_rf m
    LEFT OUTER JOIN
    test_rf t;
   | 892 | ["2 [0.0] = 0.0","0 [3.0] = 3.0","1 [696.0] != 107.0","7 [7.8292] <= 7.9104","1 [696.0] != 828.0","1 [696.0] != 391.0","0 [0.961038961038961, 0.03896103896103896]"] |

  -- Show 100 frequent branches
  WITH tmp as (
    SELECT
      decision_path(m.model_id, m.model, t.features, '-classification -no_verbose -no_leaf', array('pclass','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked'), array('no','yes')) as path
    FROM
      model_rf m
      LEFT OUTER JOIN -- CROSS JOIN
      test_rf t
  )
  select
    r.branch,
    count(1) as cnt
  from
    tmp l
    LATERAL VIEW explode(l.path) r as branch
  group by
    r.branch
  order by
    cnt desc
  limit 100;
  ```

- `guess_attribute_types(ANY, ...)` - Returns attribute types
  ```sql
  select guess_attribute_types(*) from train limit 1;
   Q,Q,C,C,C,C,Q,C,C,C,Q,C,Q,Q,Q,Q,C,Q
  ```

- `rf_ensemble(int yhat [, array<double> proba [, double model_weight=1.0]])` - Returns ensembled prediction results in &lt;int label, double probability, array&lt;double&gt; probabilities&gt;

- `tree_export(string model, const string options, optional array<string> featureNames=null, optional array<string> classNames=null)` - exports a Decision Tree model as javascript/dot]

- `tree_predict(string modelId, string model, array<double|string> features [, const string options | const boolean classification=false])` - Returns a prediction result of a random forest in &lt;int value, array&lt;double&gt; a posteriori&gt; for classification and &lt;double&gt; for regression

# XGBoost

- `train_xgboost(array<string|double> features, <int|double> target, const string options)` - Returns a relation consists of &lt;string model_id, array&lt;string&gt; pred_model&gt;
  ```sql
  SELECT 
    train_xgboost(features, label, '-objective binary:logistic -iters 10') 
      as (model_id, model)
  from (
    select features, label
    from xgb_input
    cluster by rand(43) -- shuffle
  ) shuffled;
  ```

- `xgboost_batch_predict(PRIMITIVE rowid, array<string|double> features, string model_id, array<string> pred_model [, string options])` - Returns a prediction result as (string rowid, array&lt;double&gt; predicted)
  ```sql
  select
    rowid, 
    array_avg(predicted) as predicted,
    avg(predicted[0]) as predicted0
  from (
    select
      xgboost_batch_predict(rowid, features, model_id, model) as (rowid, predicted)
    from
      xgb_model l
      LEFT OUTER JOIN xgb_input r
  ) t
  group by rowid;
  ```

- `xgboost_predict(PRIMITIVE rowid, array<string|double> features, string model_id, array<string> pred_model [, string options])` - Returns a prediction result as (string rowid, array&lt;double&gt; predicted)
  ```sql
  select
    rowid, 
    array_avg(predicted) as predicted,
    avg(predicted[0]) as predicted0
  from (
    select
      xgboost_predict(rowid, features, model_id, model) as (rowid, predicted)
    from
      xgb_model l
      LEFT OUTER JOIN xgb_input r
  ) t
  group by rowid;
  ```

- `xgboost_predict_one(PRIMITIVE rowid, array<string|double> features, string model_id, array<string> pred_model [, string options])` - Returns a prediction result as (string rowid, double predicted)
  ```sql
  select
    rowid, 
    avg(predicted) as predicted
  from (
    select
      xgboost_predict_one(rowid, features, model_id, model) as (rowid, predicted)
    from
      xgb_model l
      LEFT OUTER JOIN xgb_input r
  ) t
  group by rowid;
  ```

- `xgboost_predict_triple(PRIMITIVE rowid, array<string|double> features, string model_id, array<string> pred_model [, string options])` - Returns a prediction result as (string rowid, string label, double probability)
  ```sql
  select
    rowid,
    label,
    avg(prob) as prob
  from (
    select
      xgboost_predict_triple(rowid, features, model_id, model) as (rowid, label, prob)
    from
      xgb_model l
      LEFT OUTER JOIN xgb_input r
  ) t
  group by rowid, label;
  ```

- `xgboost_version()` - Returns the version of xgboost
  ```sql
  SELECT xgboost_version();
  ```

# Term Vector Model

- `bm25(double termFrequency, int docLength, double avgDocLength, int numDocs, int numDocsWithTerm [, const string options])` - Return an Okapi BM25 score in double. Refer http://hivemall.incubator.apache.org/userguide/ft_engineering/bm25.html for usage

- `tf(string text)` - Return a term frequency in &lt;string, float&gt;

# Others

- `hivemall_version()` - Returns the version of Hivemall
  ```sql
  SELECT hivemall_version();
  ```

- `lr_datagen(options string)` - Generates a logistic regression dataset
  ```sql
  WITH dual AS (SELECT 1) SELECT lr_datagen('-n_examples 1k -n_features 10') FROM dual;
  ```

- `bm25(double termFrequency, int docLength, double avgDocLength, int numDocs, int numDocsWithTerm [, const string options])` - Return an Okapi BM25 score in double. Refer http://hivemall.incubator.apache.org/userguide/ft_engineering/bm25.html for usage

- `tf(string text)` - Return a term frequency in &lt;string, float&gt;

