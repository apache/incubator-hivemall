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

First of all, you need to issue the following DDLs to use the NLP module. Note NLP module is not included in [hivemall-with-dependencies.jar](https://github.com/myui/hivemall/releases).

> add jar /tmp/[hivemall-nlp-xxx-with-dependencies.jar](https://github.com/myui/hivemall/releases);

> source /tmp/[define-additional.hive](https://github.com/myui/hivemall/releases);

## Japanese Tokenizer

Japanese text tokenizer UDF uses [Kuromoji](https://github.com/atilika/kuromoji). 

The signature of the UDF is as follows:
```sql
tokenize_ja(text input, optional const text mode = "normal", optional const array<string> stopWords, optional const array<string> stopTags)
```
_Caution: `tokenize_ja` is supported since Hivemall v0.4.1 and later._

It's basic usage is as follows:
```sql
select tokenize_ja("kuromojiを使った分かち書きのテストです。第二引数にはnormal/search/extendedを指定できます。デフォルトではnormalモードです。");
```
> ["kuromoji","使う","分かち書き","テスト","第","二","引数","normal","search","extended","指定","デフォルト","normal","モード"]

For detailed APIs, please refer Javadoc of [JapaneseAnalyzer](https://lucene.apache.org/core/5_3_1/analyzers-kuromoji/org/apache/lucene/analysis/ja/JapaneseAnalyzer.html) as well.

## Chinese Tokenizer

Chinese text tokenizer UDF uses [SmartChineseAnalyzer](http://lucene.apache.org/core/5_3_1/analyzers-smartcn/org/apache/lucene/analysis/cn/smart/SmartChineseAnalyzer.html). 

The signature of the UDF is as follows:
```sql
tokenize_cn(string line, optional const array<string> stopWords)
```

It's basic usage is as follows:
```sql
select tokenize_cn("Smartcn为Apache2.0协议的开源中文分词系统，Java语言编写，修改的中科院计算所ICTCLAS分词系统。");
```
> [smartcn, 为, apach, 2, 0, 协议, 的, 开源, 中文, 分词, 系统, java, 语言, 编写, 修改, 的, 中科院, 计算, 所, ictcla, 分词, 系统]

For detailed APIs, please refer Javadoc of [SmartChineseAnalyzer](http://lucene.apache.org/core/5_3_1/analyzers-smartcn/org/apache/lucene/analysis/cn/smart/SmartChineseAnalyzer.html) as well.