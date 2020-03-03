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

# Aggregation

- `majority_vote(Primitive x)` - Returns the most frequent value of x
  ```sql
  -- see https://issues.apache.org/jira/browse/HIVE-17406 
  WITH data as (
    select
      explode(array('1', '2', '2', '2', '5', '4', '1', '2')) as k
  )
  select
    majority_vote(k) as k
  from 
    data;
  2
  ```

- `max_by(x, y)` - Returns the value of x associated with the maximum value of y over all input values.
  ```sql
  -- see https://issues.apache.org/jira/browse/HIVE-17406 
  WITH data as (
    select 'jake' as name, 18 as age
    union all
    select 'tom' as name, 64 as age
    union all
    select 'lisa' as name, 32 as age
  )
  select
    max_by(name, age) as name
  from
    data;
  tom
  ```

- `min_by(x, y)` - Returns the value of x associated with the minimum value of y over all input values.
  ```sql
  -- see https://issues.apache.org/jira/browse/HIVE-17406 
  WITH data as (
    select 'jake' as name, 18 as age
    union all
    select 'tom' as name, 64 as age
    union all
    select 'lisa' as name, 32 as age
  )
  select
    min_by(name, age) as name
  from
    data;

  jake
  ```

# Array

- `arange([int start=0, ] int stop, [int step=1])` - Return evenly spaced values within a given interval
  ```sql
  SELECT arange(5), arange(1, 5), arange(1, 5, 1), arange(0, 5, 1);
  > [0,1,2,3,4]     [1,2,3,4]       [1,2,3,4]       [0,1,2,3,4]

  SELECT arange(1, 6, 2);
  > 1, 3, 5

  SELECT arange(-1, -6, 2);
  -1, -3, -5
  ```

- `argmax(array<T> a)` - Returns the first index of the maximum value
  ```sql
  SELECT argmax(array(5,2,0,1));
  0
  ```

- `argmin(array<T> a)` - Returns the first index of the minimum value
  ```sql
  SELECT argmin(array(5,2,0,1));
  2
  ```

- `argrank(array<ANY> a)` - Returns the indices that would sort an array.
  ```sql
  SELECT argrank(array(5,2,0,1)), argsort(argsort(array(5,2,0,1)));
  [3, 2, 0, 1]     [3, 2, 0, 1]
  ```

- `argsort(array<ANY> a)` - Returns the indices that would sort an array.
  ```sql
  SELECT argsort(array(5,2,0,1));
  2, 3, 1, 0

  SELECT array_slice(array(5,2,0,1), argsort(array(5,2,0,1)));
  0, 1, 2, 5
  ```

- `array_append(array<T> arr, T elem)` - Append an element to the end of an array
  ```sql
  SELECT array_append(array(1,2),3);
   1,2,3

  SELECT array_append(array('a','b'),'c');
   "a","b","c"
  ```

- `array_avg(array<number>)` - Returns an array&lt;double&gt; in which each element is the mean of a set of numbers
  ```sql
  WITH input as (
    select array(1.0, 2.0, 3.0) as nums
    UNION ALL
    select array(2.0, 3.0, 4.0) as nums
  )
  select
    array_avg(nums)
  from
    input;

  ["1.5","2.5","3.5"]
  ```

- `array_concat(array<ANY> x1, array<ANY> x2, ..)` - Returns a concatenated array
  ```sql
  SELECT array_concat(array(1),array(2,3));
   [1,2,3]
  ```

- `array_flatten(array<array<ANY>>)` - Returns an array with the elements flattened.
  ```sql
  SELECT array_flatten(array(array(1,2,3),array(4,5),array(6,7,8)));
   [1,2,3,4,5,6,7,8]
  ```

- `array_intersect(array<ANY> x1, array<ANY> x2, ..)` - Returns an intersect of given arrays
  ```sql
  SELECT array_intersect(array(1,3,4),array(2,3,4),array(3,5));
   [3]
  ```

- `array_remove(array<PRIMITIVE> values, PRIMITIVE|array<PRIMITIVE> target)` - Returns an array that the target elements are removed from the original array
  ```sql
  select array_remove(array(2.0,2.1,3.0,4.0,2.0),2), array_remove(array(2.0,3.0,4.0),array(3,2.0));
  [2.1,3,4]       [4]

  SELECT array_remove(array(1,null,3),null);
  [1,3]

  SELECT array_remove(array(1,null,3,null,5),null);
  [1,3,5]

  SELECT array_remove(array(1,null,3),array(null));
  [1,3]

  SELECT array_remove(array('aaa','bbb'),'bbb');
  ["aaa"]

  SELECT array_remove(array('aaa','bbb','ccc','bbb'), array('bbb','ccc'));
  ["aaa"]

  select array_remove(array(null),null);
  []

  select array_remove(array(null,'bbb'),'aaa');
  [null,"bbb"]
  ```

- `array_slice(array<ANY> values, int offset [, int length])` - Slices the given array by the given offset and length parameters.
  ```sql
  SELECT 
    array_slice(array(1,2,3,4,5,6),2,4),
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
  ```sql
  WITH input as (
    select array(1.0, 2.0, 3.0) as nums
    UNION ALL
    select array(2.0, 3.0, 4.0) as nums
  )
  select
    array_sum(nums)
  from
    input;

  ["3.0","5.0","7.0"]
  ```

- `array_to_str(array arr [, string sep=','])` - Convert array to string using a sperator
  ```sql
  SELECT array_to_str(array(1,2,3),'-');
  1-2-3
  ```

- `array_union(array1, array2, ...)` - Returns the union of a set of arrays
  ```sql
  SELECT array_union(array(1,2),array(1,2));
  [1,2]

  SELECT array_union(array(1,2),array(2,3),array(2,5));
  [1,2,3,5]
  ```

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
  ```sql
  SELECT element_at(array(1,2,3,4),0);
   1

  SELECT element_at(array(1,2,3,4),-2);
   3
  ```

- `first_element(x)` - Returns the first element in an array
  ```sql
  SELECT first_element(array('a','b','c'));
   a

  SELECT first_element(array());
   NULL
  ```

- `float_array(nDims)` - Returns an array&lt;float&gt; of nDims elements

- `last_element(x)` - Return the last element in an array
  ```sql
  SELECT last_element(array('a','b','c'));
   c
  ```

- `select_k_best(array<number> array, const array<number> importance, const int k)` - Returns selected top-k elements as array&lt;double&gt;

- `sort_and_uniq_array(array<int>)` - Takes array&lt;int&gt; and returns a sorted array with duplicate elements eliminated
  ```sql
  SELECT sort_and_uniq_array(array(3,1,1,-2,10));
   [-2,1,3,10]
  ```

- `subarray(array<ANY> values, int fromIndex [, int toIndex])`- Returns a slice of the original array between the inclusive fromIndex and the exclusive toIndex.
  ```sql
  SELECT 
    subarray(array(0,1,2,3,4,5),4),
    subarray(array(0,1,2,3,4,5),3,4),
    subarray(array(0,1,2,3,4,5),3,3),
    subarray(array(0,1,2,3,4,5),3,2),
    subarray(array(0,1,2,3,4,5),0,2),
    subarray(array(0,1,2,3,4,5),-1,2),
    subarray(array(1,2,3,4,5,6),4),
    subarray(array(1,2,3,4,5,6),4,6),
    subarray(array(1,2,3,4,5,6),2,4),
    subarray(array(1,2,3,4,5,6),0,2),
    subarray(array(1,2,3,4,5,6),4,6),
    subarray(array(1,2,3,4,5,6),4,7);

   [4,5]
   [3]
   []
   []
   [0,1]
   [0,1]
   [5,6]
   [5,6]
   [3,4]
   [1,2]
   [5,6]
   [5,6]
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
  ```sql
  select to_string_array(array(1.0,2.0,3.0));

  ["1.0","2.0","3.0"]
  ```

- `to_ordered_list(PRIMITIVE value [, PRIMITIVE key, const string options])` - Return list of values sorted by value itself or specific key
  ```sql
  WITH data as (
      SELECT 5 as key, 'apple' as value
      UNION ALL
      SELECT 3 as key, 'banana' as value
      UNION ALL
      SELECT 4 as key, 'candy' as value
      UNION ALL
      SELECT 1 as key, 'donut' as value
      UNION ALL
      SELECT 2 as key, 'egg' as value 
      UNION ALL
      SELECT 4 as key, 'candy' as value -- both key and value duplicates
  )
  SELECT                                                  -- expected output
      to_ordered_list(value, key, '-reverse'),            -- [apple, candy, candy, (banana, egg | egg, banana), donut] (reverse order)
      to_ordered_list(value, key, '-k 2'),                -- [apple, candy] (top-k)
      to_ordered_list(value, key, '-k 100'),              -- [apple, candy, candy, (banana, egg | egg, banana), dunut]
      to_ordered_list(value, key, '-k 2 -reverse'),       -- [donut, (banana | egg)] (reverse top-k = tail-k)
      to_ordered_list(value, key),                        -- [donut, (banana, egg | egg, banana), candy, candy, apple] (natural order)
      to_ordered_list(value, key, '-k -2'),               -- [donut, (banana | egg)] (tail-k)
      to_ordered_list(value, key, '-k -100'),             -- [donut, (banana, egg | egg, banana), candy, candy, apple]
      to_ordered_list(value, key, '-k -2 -reverse'),      -- [apple, candy] (reverse tail-k = top-k)
      to_ordered_list(value, '-k 2'),                     -- [egg, donut] (alphabetically)
      to_ordered_list(key, '-k -2 -reverse'),             -- [5, 4] (top-2 keys)
      to_ordered_list(key),                               -- [1, 2, 3, 4, 4, 5] (natural ordered keys)
      to_ordered_list(value, key, '-k 2 -kv_map'),        -- {5:"apple",4:"candy"}
      to_ordered_list(value, key, '-k 2 -vk_map'),        -- {"apple":5,"candy":4}
      to_ordered_list(value, key, '-k -2 -kv_map'),       -- {1:"donut",2:"egg"}
      to_ordered_list(value, key, '-k -2 -vk_map'),       -- {"donut":1,"egg":2}
      to_ordered_list(value, key, '-k 4 -dedup -vk_map'), -- {"apple":5,"candy":4,"banana":3,"egg":2}
      to_ordered_list(value, key, '-k 4 -vk_map'),        -- {"apple":5,"candy":4,"banana":3}
      to_ordered_list(value, key, '-k 4 -dedup'),         -- ["apple","candy","banana","egg"]
      to_ordered_list(value, key, '-k 4')                 -- ["apple","candy","candy","banana"]
  FROM
      data
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
    from_json(to_json(map('one',1,'two',2)), 'map<string,int>'),
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
  ```

  ```
   {"one":1,"two":2}
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
  ```

  ```
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

- `map_exclude_keys(Map<K,V> map, array<K> filteringKeys)` - Returns the filtered entries of a map not having specified keys
  ```sql
  SELECT map_exclude_keys(map(1,'one',2,'two',3,'three'),array(2,3));
  {1:"one"}
  ```

- `map_get(MAP<K> a, K n)` - Returns the value corresponding to the key in the map.
  ```sql
  Note this is a workaround for a Hive issue that non-constant expression for map indexes not supported.
  See https://issues.apache.org/jira/browse/HIVE-1955

  WITH tmp as (
    SELECT "one" as key
    UNION ALL
    SELECT "two" as key
  )
  SELECT map_get(map("one",1,"two",2),key)
  FROM tmp;

  > 1
  > 2
  ```

- `map_get_sum(map<int,float> src, array<int> keys)` - Returns sum of values that are retrieved by keys

- `map_include_keys(Map<K,V> map, array<K> filteringKeys)` - Returns the filtered entries of a map having specified keys
  ```sql
  SELECT map_include_keys(map(1,'one',2,'two',3,'three'),array(2,3));
  {2:"two",3:"three"}
  ```

- `map_key_values(MAP<K, V> map)` - Returns a array of key-value pairs in array&lt;named_struct&lt;key,value&gt;&gt;
  ```sql
  SELECT map_key_values(map("one",1,"two",2));

  > [{"key":"one","value":1},{"key":"two","value":2}]
  ```

- `map_roulette(Map<K, number> map [, (const)` int/bigint seed]) - Returns a map key based on weighted random sampling of map values. Average of values is used for null values
  ```sql
  -- `map_roulette(map<key, number> [, integer seed])` returns key by weighted random selection
  SELECT 
    map_roulette(to_map(a, b)) -- 25% Tom, 21% Zhang, 54% Wang
  FROM ( -- see https://issues.apache.org/jira/browse/HIVE-17406
    select 'Wang' as a, 54 as b
    union all
    select 'Zhang' as a, 21 as b
    union all
    select 'Tom' as a, 25 as b
  ) tmp;
  > Wang

  -- Weight random selection with using filling nulls with the average value
  SELECT
    map_roulette(map(1, 0.5, 'Wang', null)), -- 50% Wang, 50% 1
    map_roulette(map(1, 0.5, 'Wang', null, 'Zhang', null)) -- 1/3 Wang, 1/3 1, 1/3 Zhang
  ;

  -- NULL will be returned if every key is null
  SELECT 
    map_roulette(map()),
    map_roulette(map(null, null, null, null));
  > NULL    NULL

  -- Return NULL if all weights are zero
  SELECT
    map_roulette(map(1, 0)),
    map_roulette(map(1, 0, '5', 0))
  ;
  > NULL    NULL

  -- map_roulette does not support non-numeric weights or negative weights.
  SELECT map_roulette(map('Wong', 'A string', 'Zhao', 2));
  > HiveException: Error evaluating map_roulette(map('Wong':'A string','Zhao':2))
  SELECT map_roulette(map('Wong', 'A string', 'Zhao', 2));
  > UDFArgumentException: Map value must be greather than or equals to zero: -2
  ```

- `map_tail_n(map SRC, int N)` - Returns the last N elements from a sorted array of SRC

- `merge_maps(Map x)` - Returns a map which contains the union of an aggregation of maps. Note that an existing value of a key can be replaced with the other duplicate key entry.
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
  ```sql
  WITH input as (
    select 'aaa' as key, 111 as value
    UNION all
    select 'bbb' as key, 222 as value
  )
  select to_map(key, value)
  from input;

  > {"bbb":222,"aaa":111}
  ```

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

- `infinity()` - Returns the constant representing positive infinity.

- `is_finite(x)` - Determine if x is finite.
  ```sql
  SELECT is_finite(333), is_finite(infinity());
  true false
  ```

- `is_infinite(x)` - Determine if x is infinite.

- `is_nan(x)` - Determine if x is not-a-number.

- `l2_norm(double x)` - Return a L2 norm of the given input x.
  ```sql
  WITH input as (
    select generate_series(1,3) as v
  )
  select l2_norm(v) as l2norm
  from input;
  3.7416573867739413 = sqrt(1^2+2^2+3^2))
  ```

- `nan()` - Returns the constant representing not-a-number.
  ```sql
  SELECT nan(), is_nan(nan());
  NaN true
  ```

- `sigmoid(x)` - Returns 1.0 / (1.0 + exp(-x))
  ```sql
  WITH input as (
    SELECT 3.0 as x
    UNION ALL
    SELECT -3.0 as x
  )
  select 
    1.0 / (1.0 + exp(-x)),
    sigmoid(x)
  from
    input;
  0.04742587317756678   0.04742587357759476
  0.9525741268224334    0.9525741338729858
  ```

# Vector/Matrix

- `transpose_and_dot(array<number> X, array<number> Y)` - Returns dot(X.T, Y) as array&lt;array&lt;double&gt;&gt;, shape = (X.#cols, Y.#cols)
  ```sql
  WITH input as (
    select array(1.0, 2.0, 3.0, 4.0) as x, array(1, 2) as y
    UNION ALL
    select array(2.0, 3.0, 4.0, 5.0) as x, array(1, 2) as y
  )
  select
    transpose_and_dot(x, y) as xy,
    transpose_and_dot(y, x) as yx
  from 
    input;

  [["3.0","6.0"],["5.0","10.0"],["7.0","14.0"],["9.0","18.0"]]   [["3.0","5.0","7.0","9.0"],["6.0","10.0","14.0","18.0"]]

  ```

- `vector_add(array<NUMBER> x, array<NUMBER> y)` - Perform vector ADD operation.
  ```sql
  SELECT vector_add(array(1.0,2.0,3.0), array(2, 3, 4));
  [3.0,5.0,7.0]
  ```

- `vector_dot(array<NUMBER> x, array<NUMBER> y)` - Performs vector dot product.
  ```sql
  SELECT vector_dot(array(1.0,2.0,3.0),array(2.0,3.0,4.0));
  20

  SELECT vector_dot(array(1.0,2.0,3.0),2);
  [2.0,4.0,6.0]
  ```

# Sanity Checks

- `assert(boolean condition)` or _FUNC_(boolean condition, string errMsg)- Throws HiveException if condition is not met
  ```sql
  SELECT count(1) FROM stock_price WHERE assert(price > 0.0);
  SELECT count(1) FROM stock_price WHERE assert(price > 0.0, 'price MUST be more than 0.0')
  ```

- `raise_error()` or _FUNC_(string msg) - Throws an error
  ```sql
  SELECT product_id, price, raise_error('Found an invalid record') FROM xxx WHERE price < 0.0
  ```

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

- `str_contains(string query, array<string> searchTerms [, boolean orQuery=false])` - Returns true if the given query contains search terms
  ```sql
  select
    str_contains('There are apple and orange', array('apple')), -- or=false
    str_contains('There are apple and orange', array('apple', 'banana'), true), -- or=true
    str_contains('There are apple and orange', array('apple', 'banana'), false); -- or=false
  > true, true, false
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

# Others

- `convert_label(const int|const float)` - Convert from -1|1 to 0.0f|1.0f, or from 0.0f|1.0f to -1|1

- `each_top_k(int K, Object group, double cmpKey, *)` - Returns top-K values (or tail-K values when k is less than 0)

- `generate_series(const int|bigint start, const int|bigint end)` - Generate a series of values, from start to end. A similar function to PostgreSQL's [generate_serics](https://www.postgresql.org/docs/current/static/functions-srf.html)
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

   (no return)

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

