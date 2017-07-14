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
        
# Tokenizer for English Texts

Hivemall provides simple English text tokenizer UDF that has following syntax:
```sql
tokenize(text input, optional boolean toLowerCase = false)
```

# Tokenizer for Non-English Texts

Hivemall-NLP module provides some Non-English Text tokenizer UDFs as follows.

First of all, you need to issue the following DDLs to use the NLP module. Note NLP module is not included in `hivemall-with-dependencies.jar`.

> add jar /path/to/hivemall-nlp-xxx-with-dependencies.jar;

> source /path/to/define-additional.hive;

## Japanese Tokenizer

Japanese text tokenizer UDF uses [Kuromoji](https://github.com/atilika/kuromoji). 

The signature of the UDF is as follows:

```sql
tokenize_ja(text input, optional const text mode = "normal", optional const array<string> stopWords, const array<string> stopTags, const array<string> userDict)
```

> #### Note
> `tokenize_ja` is supported since Hivemall v0.4.1, and the fifth argument is supported since v0.5-rc.1 and later.

Its basic usage is as follows:
```sql
select tokenize_ja("kuromojiを使った分かち書きのテストです。第二引数にはnormal/search/extendedを指定できます。デフォルトではnormalモードです。");
```
> ["kuromoji","使う","分かち書き","テスト","第","二","引数","normal","search","extended","指定","デフォルト","normal","モード"]

In addition, the third and fourth argument respectively allow you to use your own list of stop words and stop tags. For example, the following query simply ignores "kuromoji" (as a stop word) and noun word "分かち書き" (as a stop tag):

```sql
select tokenize_ja("kuromojiを使った分かち書きのテストです。", "normal", array("kuromoji"), array("名詞-一般"));
```

> ["を","使う","た","の","テスト","です"]

Moreover, the fifth argument `userDict` enables you to register a user-defined custom dictionary in [Kuromoji official format](https://github.com/atilika/kuromoji/blob/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt):

```sql
select tokenize_ja("日本経済新聞＆関西国際空港", "normal", null, null, 
                   array(
                     "日本経済新聞,日本 経済 新聞,ニホン ケイザイ シンブン,カスタム名詞", 
                     "関西国際空港,関西 国際 空港,カンサイ コクサイ クウコウ,テスト名詞"
                   ));
```

> ["日本","経済","新聞","関西","国際","空港"]

Note that you can pass `null` to each of the third and fourth argument to explicitly use Kuromoji's default stop words and stop tags. 

If you have a large custom dictionary as an external file, `userDict` can also be `const string userDictURL` which indicates URL of the external file on somewhere like Amazon S3:

```sql
select tokenize_ja("日本経済新聞＆関西国際空港", "normal", null, null,
                   "https://raw.githubusercontent.com/atilika/kuromoji/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt");
```

> ["日本","経済","新聞","関西","国際","空港"]

For detailed APIs, please refer Javadoc of [JapaneseAnalyzer](https://lucene.apache.org/core/5_3_1/analyzers-kuromoji/org/apache/lucene/analysis/ja/JapaneseAnalyzer.html) as well.

## Chinese Tokenizer

Chinese text tokenizer UDF uses [SmartChineseAnalyzer](http://lucene.apache.org/core/5_3_1/analyzers-smartcn/org/apache/lucene/analysis/cn/smart/SmartChineseAnalyzer.html). 

The signature of the UDF is as follows:
```sql
tokenize_cn(string line, optional const array<string> stopWords)
```

Its basic usage is as follows:
```sql
select tokenize_cn("Smartcn为Apache2.0协议的开源中文分词系统，Java语言编写，修改的中科院计算所ICTCLAS分词系统。");
```
> [smartcn, 为, apach, 2, 0, 协议, 的, 开源, 中文, 分词, 系统, java, 语言, 编写, 修改, 的, 中科院, 计算, 所, ictcla, 分词, 系统]

For detailed APIs, please refer Javadoc of [SmartChineseAnalyzer](http://lucene.apache.org/core/5_3_1/analyzers-smartcn/org/apache/lucene/analysis/cn/smart/SmartChineseAnalyzer.html) as well.