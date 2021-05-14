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

# Tokenizer for English Texts

Hivemall provides simple English text tokenizer UDF that has following syntax:
```sql
tokenize(text input, optional boolean toLowerCase = false)
```

# Tokenizer for Non-English Texts

## Japanese Tokenizer

Japanese text tokenizer UDF uses [Kuromoji](https://github.com/atilika/kuromoji). 

The signature of the UDF is as follows:

```sql
-- uses Kuromoji default dictionary by the default
tokenize_ja(text input, optional const text mode = "normal", optional const array<string> stopWords, const array<string> stopTags, const array<string> userDict)

-- tokenize_ja_neologd uses mecab-ipa-neologd for it's dictionary.
tokenize_ja_neologd(text input, optional const text mode = "normal", optional const array<string> stopWords, const array<string> stopTags, const array<string> userDict)
```

> #### Note
> `tokenize_ja_neologd` returns tokenized strings in an array by using the NEologd dictionary. [mecab-ipadic-NEologd](https://github.com/neologd/mecab-ipadic-neologd) is a customized system dictionary for MeCab inclucing new vocablaries extracted from many resources on the Web. 

See differences between with and without Neologd as follows:

```sql
select tokenize_ja("彼女はペンパイナッポーアッポーペンと恋ダンスを踊った。");
>["彼女","ペンパイナッポーアッポーペン","恋","ダンス","踊る"]

select tokenize_ja_neologd("彼女はペンパイナッポーアッポーペンと恋ダンスを踊った。");
> ["彼女","ペンパイナッポーアッポーペン","恋ダンス","踊る"]
```

You can print versions for Kuromoji UDFs as follows:

```sql
select tokenize_ja();
> ["8.8.2"]

select tokenize_ja_neologd();
> ["8.8.2-20200910.2"]
```

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

```sql
select tokenize_ja("kuromojiを使った分かち書きのテストです。", "normal", array("kuromoji"), stoptags_exclude(array("名詞")));
```
> ["分かち書き","テスト"]

`stoptags_exclude(array<string> tags, [, const string lang='ja'])` is a useful UDF for getting [stoptags](https://github.com/apache/lucene-solr/blob/master/lucene/analysis/kuromoji/src/resources/org/apache/lucene/analysis/ja/stoptags.txt) excluding given part-of-speech tags as seen below:

```sql
select stoptags_exclude(array("名詞-固有名詞"));
```

> ["その他","その他-間投","フィラー","副詞","副詞-一般","副詞-助詞類接続","助動詞","助詞","助詞-並立助詞"
,"助詞-係助詞","助詞-副助詞","助詞-副助詞／並立助詞／終助詞","助詞-副詞化","助詞-接続助詞","助詞-格助詞
","助詞-格助詞-一般","助詞-格助詞-引用","助詞-格助詞-連語","助詞-特殊","助詞-終助詞","助詞-連体化","助
詞-間投助詞","動詞","動詞-接尾","動詞-自立","動詞-非自立","名詞","名詞-サ変接続","名詞-ナイ形容詞語幹",
"名詞-一般","名詞-代名詞","名詞-代名詞-一般","名詞-代名詞-縮約","名詞-副詞可能","名詞-動詞非自立的","名
詞-引用文字列","名詞-形容動詞語幹","名詞-接尾","名詞-接尾-サ変接続","名詞-接尾-一般","名詞-接尾-人名","
名詞-接尾-副詞可能","名詞-接尾-助動詞語幹","名詞-接尾-助数詞","名詞-接尾-地域","名詞-接尾-形容動詞語幹"
,"名詞-接尾-特殊","名詞-接続詞的","名詞-数","名詞-特殊","名詞-特殊-助動詞語幹","名詞-非自立","名詞-非自
立-一般","名詞-非自立-副詞可能","名詞-非自立-助動詞語幹","名詞-非自立-形容動詞語幹","形容詞","形容詞-接
尾","形容詞-自立","形容詞-非自立","感動詞","接続詞","接頭詞","接頭詞-動詞接続","接頭詞-名詞接続","接頭
詞-形容詞接続","接頭詞-数接","未知語","記号","記号-アルファベット","記号-一般","記号-句点","記号-括弧閉
","記号-括弧開","記号-空白","記号-読点","語断片","連体詞","非言語音"]

### Custom dictionary

Moreover, the fifth argument `userDict` enables you to register a user-defined custom dictionary in [Kuromoji official format](https://github.com/atilika/kuromoji/blob/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt):

```sql
select tokenize_ja("日本経済新聞＆関西国際空港", "normal", null, null, 
                   array(
                     "日本経済新聞,日本 経済 新聞,ニホン ケイザイ シンブン,カスタム名詞", 
                     "関西国際空港,関西 国際 空港,カンサイ コクサイ クウコウ,テスト名詞"
                   ));
```

> ["日本","経済","新聞","関西","国際","空港"]

Note that you can pass `null` to each of the third and fourth argument to explicitly use Kuromoji's [default stop words](https://github.com/apache/lucene-solr/blob/master/lucene/analysis/kuromoji/src/resources/org/apache/lucene/analysis/ja/stopwords.txt) and [stop tags](https://github.com/apache/lucene-solr/blob/master/lucene/analysis/kuromoji/src/resources/org/apache/lucene/analysis/ja/stoptags.txt).

If you have a large custom dictionary as an external file, `userDict` can also be `const string userDictURL` which indicates URL of the external file on somewhere like Amazon S3:

```sql
select tokenize_ja("日本経済新聞＆関西国際空港", "normal", null, null,
                   "https://raw.githubusercontent.com/atilika/kuromoji/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt");

> ["日本","経済","新聞","関西","国際","空港"]
```

> #### Note
> Dictionary SHOULD be accessible through http/https protocol. And, it SHOULD be compressed using gzip with `.gz` suffix because the maximum dictionary size is limited to 32MB and read timeout is set to 60 sec. Also, connection must be established in 10 sec.
>
> If you want to use HTTP Basic Authentication, please use the following form: `https://user:password@www.sitreurl.com/my_dict.txt.gz` (see Sec 3.1 of [rfc1738](https://www.ietf.org/rfc/rfc1738.txt))

For detailed APIs, please refer Javadoc of [JapaneseAnalyzer](https://lucene.apache.org/core/5_3_1/analyzers-kuromoji/org/apache/lucene/analysis/ja/JapaneseAnalyzer.html) as well.


### Part-of-speech

From Hivemall v0.6.0, the second argument can also accept the following option format:

```
 -mode <arg>   The tokenization mode. One of ['normal', 'search',
               'extended', 'default' (normal)]
 -pos          Return part-of-speech information
```

Then, you can get part-of-speech information as follows:

```sql
WITH tmp as (
  select
    tokenize_ja('kuromojiを使った分かち書きのテストです。','-mode search -pos') as r
)
select
  r.tokens,
  r.pos,
  r.tokens[0] as token0,
  r.pos[0] as pos0
from
  tmp;
```

| tokens |pos | token0 | pos0 |
|:-:|:-:|:-:|:-:|
| ["kuromoji","使う","分かち書き","テスト"] | ["名詞-一般","動詞-自立","名詞-一般","名詞-サ変接続"] | kuromoji | 名詞-一般 |

Note that when `-pos` option is specified, `tokenize_ja` returns a struct record containing `array<string> tokens` and `array<string> pos` as the elements.

## Chinese Tokenizer

Chinese text tokenizer UDF uses [SmartChineseAnalyzer](https://lucene.apache.org/core/5_3_1/analyzers-smartcn/org/apache/lucene/analysis/cn/smart/SmartChineseAnalyzer.html). 

The signature of the UDF is as follows:

```sql
tokenize_cn(string line, optional const array<string> stopWords)
```

Its basic usage is as follows:

```sql
select tokenize_cn("Smartcn为Apache2.0协议的开源中文分词系统，Java语言编写，修改的中科院计算所ICTCLAS分词系统。");

> [smartcn, 为, apach, 2, 0, 协议, 的, 开源, 中文, 分词, 系统, java, 语言, 编写, 修改, 的, 中科院, 计算, 所, ictcla, 分词, 系统]
```

For detailed APIs, please refer Javadoc of [SmartChineseAnalyzer](https://lucene.apache.org/core/5_3_1/analyzers-smartcn/org/apache/lucene/analysis/cn/smart/SmartChineseAnalyzer.html) as well.

## Korean Tokenizer

Korean toknizer internally uses [lucene-analyzers-nori](analyzers-nori: Korean Morphological Analyzer) for tokenization.

The signature of the UDF is as follows:

```sql
tokenize_ko(
       String line [, const string mode = "discard" (or const string opts),
       const array<string> stopWords,
       const array<string>
       stopTags,
       const array<string> userDict (or const string userDictURL)]
) - returns tokenized strings in array<string> 
```

> #### Note
> Instead of mode, the 2nd argument can take options starting with `-`.

You can get usage as follows:

```sql
select tokenize_ko("", "-help");

usage: tokenize_ko(String line [, const string mode = "discard" (or const
       string opts), const array<string> stopWords, const array<string>
       stopTags, const array<string> userDict (or const string
       userDictURL)]) - returns tokenized strings in array<string> [-help]
       [-mode <arg>] [-outputUnknownUnigrams]
 -help                    Show function help
 -mode <arg>              The tokenization mode. One of ['node', 'discard'
                          (default), 'mixed']
 -outputUnknownUnigrams   outputs unigrams for unknown words.
```

> #### Note
> For details options, please refer [Lucene API document](https://lucene.apache.org/core/8_8_2/analyzers-nori/org/apache/lucene/analysis/ko/KoreanAnalyzer.html). `none`, `discord` (default), or `mixed` are supported for the mode argument.

See the following examples for the usage.

```sql
-- show version of lucene-analyzers-nori
select tokenize_ko();
> 8.8.2

select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!');
> ["중요","기능","개발","주","고맙"]

-- explicitly using default options
select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!', '-mode discard', 
  -- stopwords (null to use default)
  -- see https://github.com/apache/incubator-hivemall/blob/master/nlp/src/main/resources/hivemall/nlp/tokenizer/ext/stopwords-ko.txt 
  null, 
  -- stoptags
  -- see https://lucene.apache.org/core/8_8_2/analyzers-nori/org/apache/lucene/analysis/ko/POS.Tag.html
  array(
   'E',   -- Verbal endings
   'IC',  -- Interjection
   'J',   -- Ending Particle
   'MAG', -- General Adverb
   'MAJ', -- Conjunctive adverb
   'MM',  -- Determiner
   'SP',  -- Space 
   'SSC', -- Closing brackets
   'SSO', -- Opening brackets
   'SC',  -- Separator
   'SE',  -- Ellipsis
   'XPN', -- Prefix
   'XSA', -- Adjective Suffix
   'XSN', -- Noun Suffix
   'XSV', -- Verb Suffix
   'UNA', -- Unknown
   'NA',  -- Unknown
   'VSV'  -- Unknown
  )
);
> ["중요","기능","개발","주","고맙"]

-- None mode, without General Adverb (MAG)
select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!', 
  -- No decomposition for compound.
  '-mode none', 
  -- stopwords (null to use default)
  null, 
  array(
   'E',   -- Verbal endings
   'IC',  -- Interjection
   'J',   -- Ending Particle
   -- 'MAG', -- General Adverb
   'MAJ', -- Conjunctive adverb
   'MM',  -- Determiner
   'SP',  -- Space 
   'SSC', -- Closing brackets
   'SSO', -- Opening brackets
   'SC',  -- Separator
   'SE',  -- Ellipsis
   'XPN', -- Prefix
   'XSA', -- Adjective Suffix
   'XSN', -- Noun Suffix
   'XSV', -- Verb Suffix
   'UNA', -- Unknown
   'NA',  -- Unknown
   'VSV'  -- Unknown
  )
);
> ["중요","기능","개발","줘서","정말","고마워요"]

-- discard mode: Decompose compounds and discards the original form (default).
-- https://lucene.apache.org/core/8_8_2/analyzers-nori/org/apache/lucene/analysis/ko/KoreanTokenizer.DecompoundMode.html
select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!', '-mode discard');
> ["중요","기능","개발","주","고맙"]

-- default stopward (null), with stoptags
select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!', '-mode discard', null, array('E', 'VV'));
> ["중요","하","새","기능","을","개발","하","주","정말","고맙"]

-- mixed mode: Decompose compounds and keeps the original form.
select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!', 'mixed');
> ["중요","기능","개발","줘서","주","고마워요","고맙"]

select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!', '-mode mixed');
> ["중요","기능","개발","줘서","주","고마워요","고맙"]

-- node mode: No decomposition for compound.
select tokenize_ko('중요한 새 기능을 개발해줘서 정말 고마워요!', '-mode none');
> ["중요","기능","개발","줘서","고마워요"]

select tokenize_ko('Hello, world.', '-mode none');
> ["hello","world"]

select tokenize_ko('Hello, world.', '-mode none -outputUnknownUnigrams');
> ["h","e","l","l","o","w","o","r","l","d"]

select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard');
> ["나","c","언어","프로그래밍","언어","사랑"]

select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard', array(), null);
> ["나","는","c","언어","를","프로그래밍","언어","로","사랑","하","ᆫ다"]

-- default stopward (null), default stoptags (null)
select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard');
select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard', null, null);
> ["나","c","언어","프로그래밍","언어","사랑"]

-- no stopward (empty array), default stoptags (null)
select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard', array());
select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard', array(), null);
> ["나","c","언어","프로그래밍","언어","사랑"]

-- no stopward (empty array), no stoptags (emptry array), custom dict
select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard', array(), array(), array('C++'));
> ["나","는","c++","언어","를","프로그래밍","언어","로","사랑","하","ᆫ다"]

> -- default stopward (null), default stoptags (null), custom dict
select tokenize_ko('나는 C++ 언어를 프로그래밍 언어로 사랑한다.', '-mode discard', null, null, array('C++'));
> ["나","c++","언어","프로그래밍","언어","사랑"]
```

### Custom dictionary

Moreover, the fifth argument `userDictURL` enables you to register a user-defined custom dictionary placed in http/https accessible external site. Find the dictionary format [here from Lucene's one](https://raw.githubusercontent.com/apache/lucene/main/lucene/analysis/nori/src/test/org/apache/lucene/analysis/ko/userdict.txt).


```sql
select tokenize_ko('나는 c++ 프로그래밍을 즐긴다.', '-mode discard', null, null, 'https://raw.githubusercontent.com/apache/lucene/main/lucene/analysis/nori/src/test/org/apache/lucene/analysis/ko/userdict.txt');

> ["나","c++","프로그래밍","즐기"]
```

> #### Note
> Dictionary SHOULD be accessible through http/https protocol. And, it SHOULD be compressed using gzip with `.gz` suffix because the maximum dictionary size is limited to 32MB and read timeout is set to 60 sec. Also, connection must be established in 10 sec.
