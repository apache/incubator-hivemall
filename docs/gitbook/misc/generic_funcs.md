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

This page describes a list of useful Hivemall generic functions. See also a [list of machine-learning-related functions](./funcs.md).

<!-- toc -->

# Array

- `array_append(array<T> arr, T elem)` - Append an element to the end of an array

- `array_avg(array<number>)` - Returns an array&lt;double&gt; in which each element is the mean of a set of numbers

- `array_concat(array<ANY> x1, array<ANY> x2, ..)` - Returns a concatenated array
  ```sql
  SELECT array_concat(array(1),array(2,3));
   [1,2,3]
  ```

- `array_flatten(array<array<ANY>>)` - Returns an array with the elements flattened.

- `array_intersect(array<ANY> x1, array<ANY> x2, ..)` - Returns an intersect of given arrays
  ```sql
  SELECT array_intersect(array(1,3,4),array(2,3,4),array(3,5));
   [3]
  ```

- `array_remove(array<int|text> original, int|text|array<int> target)` - Returns an array that the target is removed from the original array
  ```sql
  SELECT array_remove(array(1,null,3),array(null));
   [3]

  SELECT array_remove(array("aaa","bbb"),"bbb");
   ["aaa"]
  ```

- `array_slice(array<ANY> values, int offset [, int length])` - Slices the given array by the given offset and length parameters.
  ```sql
  SELECT 
    array_slice(array(1,2,3,4,5,6), 2,4),
    array_slice(
     array("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
     0, -- offset
     2 -- length
    ),
    array_slice(
     array("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
     6, -- offset
     3 -- length
    ),
    array_slice(
     array("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
     6, -- offset
     10 -- length
    ),
    array_slice(
     array("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
     6 -- offset
    ),
    array_slice(
     array("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
     -3 -- offset
    ),
    array_slice(
     array("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"),
     -3, -- offset
     2 -- length
    );

   [3,4]
   ["zero","one"] 
   ["six","seven","eight"]
   ["six","seven","eight","nine","ten"]
   ["six","seven","eight","nine","ten"]
   ["eight","nine","ten"]
   ["eight","nine"]
  ```

- `array_sum(array<number>)` - Returns an array&lt;double&gt; in which each element is summed up

- `array_union(array1, array2, ...)` - Returns the union of a set of arrays

- `conditional_emit(array<boolean> conditions, array<primitive> features)` - Emit features of a row according to various conditions
  ```sql
  WITH input as (
     select array(true, false, true) as conditions, array("one", "two", "three") as features
     UNION ALL
     select array(true, true, false), array("four", "five", "six")
  )
  SELECT
    conditional_emit(
       conditions, features
    )
  FROM 
    input;
   one
   three
   four
   five
  ```

- `element_at(array<T> list, int pos)` - Returns an element at the given position

- `first_element(x)` - Returns the first element in an array 

- `float_array(nDims)` - Returns an array&lt;float&gt; of nDims elements

- `last_element(x)` - Return the last element in an array

- `select_k_best(array<number> array, const array<number> importance, const int k)` - Returns selected top-k elements as array&lt;double&gt;

- `sort_and_uniq_array(array<int>)` - Takes array&lt;int&gt; and returns a sorted array with duplicate elements eliminated
  ```sql
  SELECT sort_and_uniq_array(array(3,1,1,-2,10));
   [-2,1,3,10]
  ```

- `subarray_endwith(array<int|text> original, int|text key)` - Returns an array that ends with the specified key
  ```sql
  SELECT subarray_endwith(array(1,2,3,4), 3);
   [1,2,3]
  ```

- `subarray_startwith(array<int|text> original, int|text key)` - Returns an array that starts with the specified key
  ```sql
  SELECT subarray_startwith(array(1,2,3,4), 2);
   [2,3,4]
  ```

- `to_string_array(array<ANY>)` - Returns an array of strings

- `to_ordered_list(PRIMITIVE value [, PRIMITIVE key, const string options])` - Return list of values sorted by value itself or specific key
  ```sql
  WITH t as (
      SELECT 5 as key, 'apple' as value
      UNION ALL
      SELECT 3 as key, 'banana' as value
      UNION ALL
      SELECT 4 as key, 'candy' as value
      UNION ALL
      SELECT 2 as key, 'donut' as value
      UNION ALL
      SELECT 3 as key, 'egg' as value
  )
  SELECT                                             -- expected output
      to_ordered_list(value, key, '-reverse'),       -- [apple, candy, (banana, egg | egg, banana), donut] (reverse order)
      to_ordered_list(value, key, '-k 2'),           -- [apple, candy] (top-k)
      to_ordered_list(value, key, '-k 100'),         -- [apple, candy, (banana, egg | egg, banana), dunut]
      to_ordered_list(value, key, '-k 2 -reverse'),  -- [donut, (banana | egg)] (reverse top-k = tail-k)
      to_ordered_list(value, key),                   -- [donut, (banana, egg | egg, banana), candy, apple] (natural order)
      to_ordered_list(value, key, '-k -2'),          -- [donut, (banana | egg)] (tail-k)
      to_ordered_list(value, key, '-k -100'),        -- [donut, (banana, egg | egg, banana), candy, apple]
      to_ordered_list(value, key, '-k -2 -reverse'), -- [apple, candy] (reverse tail-k = top-k)
      to_ordered_list(value, '-k 2'),                -- [egg, donut] (alphabetically)
      to_ordered_list(key, '-k -2 -reverse'),        -- [5, 4] (top-2 keys)
      to_ordered_list(key)                           -- [2, 3, 3, 4, 5] (natural ordered keys)
  FROM
      t
  ```

# Bitset

- `bits_collect(int|long x)` - Returns a bitset in array&lt;long&gt;

- `bits_or(array<long> b1, array<long> b2, ..)` - Returns a logical OR given bitsets
  ```sql
  SELECT unbits(bits_or(to_bits(array(1,4)),to_bits(array(2,3))));
   [1,2,3,4]
  ```

- `to_bits(int[] indexes)` - Returns an bitset representation if the given indexes in long[]
  ```sql
  SELECT to_bits(array(1,2,3,128));
   [14,-9223372036854775808]
  ```

- `unbits(long[] bitset)` - Returns an long array of the give bitset representation
  ```sql
  SELECT unbits(to_bits(array(1,4,2,3)));
   [1,2,3,4]
  ```

# Compression

- `deflate(TEXT data [, const int compressionLevel])` - Returns a compressed BINARY object by using Deflater. The compression level must be in range [-1,9]
  ```sql
  SELECT base91(deflate('aaaaaaaaaaaaaaaabbbbccc'));
   AA+=kaIM|WTt!+wbGAA
  ```

- `inflate(BINARY compressedData)` - Returns a decompressed STRING by using Inflater
  ```sql
  SELECT inflate(unbase91(base91(deflate('aaaaaaaaaaaaaaaabbbbccc'))));
   aaaaaaaaaaaaaaaabbbbccc
  ```

# Datetime

- `sessionize(long timeInSec, long thresholdInSec [, String subject])`- Returns a UUID string of a session.
  ```sql
  SELECT 
    sessionize(time, 3600, ip_addr) as session_id, 
    time, ip_addr
  FROM (
    SELECT time, ipaddr 
    FROM weblog 
    DISTRIBUTE BY ip_addr, time SORT BY ip_addr, time DESC
  ) t1
  ```

# JSON

- `from_json(string jsonString, const string returnTypes [, const array<string>|const string columnNames])` - Return Hive object.
  ```sql
  SELECT
    from_json(
      '{ "person" : { "name" : "makoto" , "age" : 37 } }',
      'struct<name:string,age:int>', 
      array('person')
    ),
    from_json(
      '[0.1,1.1,2.2]',
      'array<double>'
    ),
    from_json(to_json(
      ARRAY(
        NAMED_STRUCT("country", "japan", "city", "tokyo"), 
        NAMED_STRUCT("country", "japan", "city", "osaka")
      )
    ),'array<struct<country:string,city:string>>'),
    from_json(to_json(
      ARRAY(
        NAMED_STRUCT("country", "japan", "city", "tokyo"), 
        NAMED_STRUCT("country", "japan", "city", "osaka")
      ),
      array('city')
    ), 'array<struct<country:string,city:string>>'),
    from_json(to_json(
      ARRAY(
        NAMED_STRUCT("country", "japan", "city", "tokyo"), 
        NAMED_STRUCT("country", "japan", "city", "osaka")
      )
    ),'array<struct<city:string>>');

   {"name":"makoto","age":37}
   [0.1,1.1,2.2]
   [{"country":"japan","city":"tokyo"},{"country":"japan","city":"osaka"}]
   [{"country":"japan","city":"tokyo"},{"country":"japan","city":"osaka"}]
   [{"city":"tokyo"},{"city":"osaka"}]
  ```

- `to_json(ANY object [, const array<string>|const string columnNames])` - Returns Json string
  ```sql
  SELECT 
    NAMED_STRUCT("Name", "John", "age", 31),
    to_json(
       NAMED_STRUCT("Name", "John", "age", 31)
    ),
    to_json(
       NAMED_STRUCT("Name", "John", "age", 31),
       array('Name', 'age')
    ),
    to_json(
       NAMED_STRUCT("Name", "John", "age", 31),
       array('name', 'age')
    ),
    to_json(
       NAMED_STRUCT("Name", "John", "age", 31),
       array('age')
    ),
    to_json(
       NAMED_STRUCT("Name", "John", "age", 31),
       array()
    ),
    to_json(
       null,
       array()
    ),
    to_json(
      struct("123", "456", 789, array(314,007)),
      array('ti','si','i','bi')
    ),
    to_json(
      struct("123", "456", 789, array(314,007)),
      'ti,si,i,bi'
    ),
    to_json(
      struct("123", "456", 789, array(314,007))
    ),
    to_json(
      NAMED_STRUCT("country", "japan", "city", "tokyo")
    ),
    to_json(
      NAMED_STRUCT("country", "japan", "city", "tokyo"), 
      array('city')
    ),
    to_json(
      ARRAY(
        NAMED_STRUCT("country", "japan", "city", "tokyo"), 
        NAMED_STRUCT("country", "japan", "city", "osaka")
      )
    ),
    to_json(
      ARRAY(
        NAMED_STRUCT("country", "japan", "city", "tokyo"), 
        NAMED_STRUCT("country", "japan", "city", "osaka")
      ),
      array('city')
    );

   {"name":"John","age":31}
   {"name":"John","age":31}
   {"Name":"John","age":31}
   {"name":"John","age":31}
   {"age":31}
   {}
   NULL
   {"ti":"123","si":"456","i":789,"bi":[314,7]}
   {"ti":"123","si":"456","i":789,"bi":[314,7]}
   {"col1":"123","col2":"456","col3":789,"col4":[314,7]}
   {"country":"japan","city":"tokyo"}
   {"city":"tokyo"}
   [{"country":"japan","city":"tokyo"},{"country":"japan","city":"osaka"}]
   [{"country":"japan","city":"tokyo"},{"country":"japan","city":"osaka"}]
  ```

# Map

- `map_get_sum(map<int,float> src, array<int> keys)` - Returns sum of values that are retrieved by keys

- `map_key_values(map)` - Returns a array of key-value pairs.

- `map_tail_n(map SRC, int N)` - Returns the last N elements from a sorted array of SRC

- `merge_maps(x)` - Returns a map which contains the union of an aggregation of maps. Note that an existing value of a key can be replaced with the other duplicate key entry.
  ```sql
  SELECT 
    merge_maps(m) 
  FROM (
    SELECT map('A',10,'B',20,'C',30) 
    UNION ALL 
    SELECT map('A',10,'B',20,'C',30)
  ) t
  ```

- `to_map(key, value)` - Convert two aggregated columns into a key-value map

- `to_ordered_map(key, value [, const int k|const boolean reverseOrder=false])` - Convert two aggregated columns into an ordered key-value map
  ```sql
  with t as (
      select 10 as key, 'apple' as value
      union all
      select 3 as key, 'banana' as value
      union all
      select 4 as key, 'candy' as value
  )
  select
      to_ordered_map(key, value, true),   -- {10:"apple",4:"candy",3:"banana"} (reverse)
      to_ordered_map(key, value, 1),      -- {10:"apple"} (top-1)
      to_ordered_map(key, value, 2),      -- {10:"apple",4:"candy"} (top-2)
      to_ordered_map(key, value, 3),      -- {10:"apple",4:"candy",3:"banana"} (top-3)
      to_ordered_map(key, value, 100),    -- {10:"apple",4:"candy",3:"banana"} (top-100)
      to_ordered_map(key, value),         -- {3:"banana",4:"candy",10:"apple"} (natural)
      to_ordered_map(key, value, -1),     -- {3:"banana"} (tail-1)
      to_ordered_map(key, value, -2),     -- {3:"banana",4:"candy"} (tail-2)
      to_ordered_map(key, value, -3),     -- {3:"banana",4:"candy",10:"apple"} (tail-3)
      to_ordered_map(key, value, -100)    -- {3:"banana",4:"candy",10:"apple"} (tail-100)
  from t
  ```

# MapReduce

- `distcache_gets(filepath, key, default_value [, parseKey])` - Returns map&lt;key_type, value_type&gt;|value_type

- `jobconf_gets()` - Returns the value from JobConf

- `jobid()` - Returns the value of mapred.job.id

- `rowid()` - Returns a generated row id of a form {TASK_ID}-{SEQUENCE_NUMBER}

- `rownum()` - Returns a generated row number `sprintf(`%d%04d`,sequence,taskId)` in long
  ```sql
  SELECT rownum() as rownum, xxx from ...
  ```

- `taskid()` - Returns the value of mapred.task.partition

# Math

- `l2_norm(double xi)` - Return L2 norm of a vector which has the given values in each dimension

- `sigmoid(x)` - Returns 1.0 / (1.0 + exp(-x))

# Matrix

- `transpose_and_dot(array<number> matrix0_row, array<number> matrix1_row)` - Returns dot(matrix0.T, matrix1) as array&lt;array&lt;double&gt;&gt;, shape = (matrix0.#cols, matrix1.#cols)

# Sanity Checks

- `assert(boolean condition)` or _FUNC_(boolean condition, string errMsg)- Throws HiveException if condition is not met

- `raise_error()` or _FUNC_(string msg) - Throws an error

# Text processing

- `base91(BINARY bin)` - Convert the argument from binary to a BASE91 string
  ```sql
  SELECT base91(deflate('aaaaaaaaaaaaaaaabbbbccc'));
   AA+=kaIM|WTt!+wbGAA
  ```

- `is_stopword(string word)` - Returns whether English stopword or not

- `normalize_unicode(string str [, string form])` - Transforms `str` with the specified normalization form. The `form` takes one of NFC (default), NFD, NFKC, or NFKD
  ```sql
  SELECT normalize_unicode('ﾊﾝｶｸｶﾅ','NFKC');
   ハンカクカナ

  SELECT normalize_unicode('㈱㌧㌦Ⅲ','NFKC');
   (株)トンドルIII
  ```

- `singularize(string word)` - Returns singular form of a given English word
  ```sql
  SELECT singularize(lower("Apples"));

   "apple"
  ```

- `split_words(string query [, string regex])` - Returns an array&lt;text&gt; containing splitted strings

- `tokenize(string englishText [, boolean toLowerCase])` - Returns tokenized words in array&lt;string&gt;

- `unbase91(string)` - Convert a BASE91 string to a binary
  ```sql
  SELECT inflate(unbase91(base91(deflate('aaaaaaaaaaaaaaaabbbbccc'))));
   aaaaaaaaaaaaaaaabbbbccc
  ```

- `word_ngrams(array<string> words, int minSize, int maxSize])` - Returns list of n-grams for given words, where `minSize &lt;= n &lt;= maxSize`
  ```sql
  SELECT word_ngrams(tokenize('Machine learning is fun!', true), 1, 2);

   ["machine","machine learning","learning","learning is","is","is fun","fun"]
  ```

# Timeseries

- `moving_avg(NUMBER value, const int windowSize)` - Returns moving average of a time series using a given window
  ```sql
  SELECT moving_avg(x, 3) FROM (SELECT explode(array(1.0,2.0,3.0,4.0,5.0,6.0,7.0)) as x) series;
   1.0
   1.5
   2.0
   3.0
   4.0
   5.0
   6.0
  ```

# Vector

- `vector_add(array<NUMBER> x, array<NUMBER> y)` - Perform vector ADD operation.

- `vector_dot(array<NUMBER> x, array<NUMBER> y)` - Performs vector dot product.

# Others

- `convert_label(const int|const float)` - Convert from -1|1 to 0.0f|1.0f, or from 0.0f|1.0f to -1|1

- `each_top_k(int K, Object group, double cmpKey, *)` - Returns top-K values (or tail-K values when k is less than 0)

- `generate_series(const int|bigint start, const int|bigint end)` - Generate a series of values, from start to end. A similar function to PostgreSQL's [generate_serics](http://www.postgresql.org/docs/current/static/functions-srf.html)
  ```sql
  SELECT generate_series(2,4);

   2
   3
   4

  SELECT generate_series(5,1,-2);

   5
   3
   1

  SELECT generate_series(4,3);

  > (no return)

  SELECT date_add(current_date(),value),value from (SELECT generate_series(1,3)) t;

   2018-04-21      1
   2018-04-22      2
   2018-04-23      3

  WITH input as (
   SELECT 1 as c1, 10 as c2, 3 as step
   UNION ALL
   SELECT 10, 2, -3
  )
  SELECT generate_series(c1, c2, step) as series
  FROM input;

   1
   4
   7
   10
   10
   7
   4
  ```

- `try_cast(ANY src, const string typeName)` - Explicitly cast a value as a type. Returns null if cast fails.
  ```sql
  SELECT try_cast(array(1.0,2.0,3.0), 'array<string>')
  SELECT try_cast(map('A',10,'B',20,'C',30), 'map<string,double>')
  ```

- `x_rank(KEY)` - Generates a pseudo sequence number starting from 1 for each key

