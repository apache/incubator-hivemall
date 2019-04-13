## Introduction

Expanding numeric labels to actual count of samples can contribute to accuracy improvement in some cases. `binarize_label` explode a record that keeps the count of positive/negative labeled samples into corresponding actual count of samples. For example,

|positive|negative|features|
|:----|:----|:----|
|2|3|"[a:1, b:2]"|

is converted into 

|features|label|
|:----|:----|
|"[a:1, b:2]"|0|
|"[a:1, b:2]"|0|
|"[a:1, b:2]"|1|
|"[a:1, b:2]"|1|
|"[a:1, b:2]"|1|

## Function signature

`binarize_label(int/long positive, int/long negative, ANY arg1, ANY arg2, ..., ANY argN)` 
returns (ANY arg1, ANY arg2, ..., ANY argN, int label) where label is 0 or 1.

## Usage

```sql
WITH input as (
  select 2 as positive, 3 as negative, array('a:1','b:2') as features
  UNION ALL
  select 2 as positive, 1 as negative, array('c:3','d:4') as features
)
SELECT
  binarize_label(positive, negative, features)
from 
  input;
```

|features|label|
|:----|:----|
| ["a:1","b:2"]  | 1 |
| ["a:1","b:2"]  | 1 |
| ["a:1","b:2"]  | 0 |
| ["a:1","b:2"]  | 0 |
| ["a:1","b:2"]  | 0 |
| ["c:3","d:4"]  | 1 |
| ["c:3","d:4"]  | 1 |
| ["c:3","d:4"]  | 0 |


> #### Caution
>
> Don't forget to shuffle converted training instances in a random order, e.g., by `CLUSTER BY rand()`.
