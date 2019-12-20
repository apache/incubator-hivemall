# v0.6.0 - Dec 19, 2019

Major new features in this release includes:

 - xgboost (v0.90) support (find [usage](http://hivemall.apache.org/userguide/binaryclass/news20b_xgboost.html))
 - Improvements on tokenize_ja
   - Part-of-Speech (PoS) support in `tokenze_ja` (find [usage](http://hivemall.apache.org/userguide/misc/tokenizer.html#part-of-speech))
   - new `stoptag_excludes` UDF (find [usage](http://hivemall.apache.org/userguide/misc/tokenizer.html#japanese-tokenizer))
 - libsvm format converter is supported (find [usage](https://hivemall.apache.org/userguide/misc/funcs.html#feature-format-conversion))
 - Improve support for Json conversion (`to_json/from_json`) (find [usage](http://hivemall.apache.org/userguide/misc/generic_funcs.html#json))
 - Introduced tracing functionality for DecisionTree predictions (find [usage](http://hivemall.apache.org/userguide/binaryclass/titanic_rf.html#tracing-predictions))
 - Support for `max_by`, `min_by`, `majority_vote` UDAFs (find [usage](http://hivemall.apache.org/userguide/misc/generic_funcs.html#aggregation))
 - Supoprt `argmax/argmin` and `argsort/argrank` UDFs (find [usage](http://hivemall.apache.org/userguide/misc/generic_funcs.html#array))
 - Refined RandomForest implementation for Sparse feature support and training speed. (find [usage](http://hivemall.apache.org/userguide/binaryclass/news20_rf.html))
 
Please refer [ChangeLog](https://dist.apache.org/repos/dist/dev/incubator/hivemall/0.6.0-incubating-rc1/ChangeLog.html) for the detail of changes.

# v0.5.2 - Dec 3, 2018

Major updated in this release includes:

 - Supported Apache Spark v2.3.
 - Brickhouse UDFs
 - Field-aware Factorization Machines

Please refer [ChangeLog](https://www.apache.org/dist/incubator/hivemall/0.5.2-incubating/ChangeLog.html) for the detail of changes.

# v0.5.0 - Mar 5, 2018

The first Apache release. 

Please Refer [ChangeLog](https://www.apache.org/dist/incubator/hivemall/0.5.0-incubating/ChangeLog.html) for the detail of changes.