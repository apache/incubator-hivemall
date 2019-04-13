/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package hivemall.nlp.tokenizer;

import hivemall.TestUtils;
import hivemall.utils.hadoop.HiveUtils;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class StoptagsExcludeUDFTest {

    @Test
    public void testGetStoptagsJA() {
        List<String> actual = StoptagsExcludeUDF.getStoptags(StoptagsExcludeUDF.STOPTAGS_JA,
            new String[] {"形容詞"});
        String[] expected = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                "フィラー", "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual);
    }

    @Test
    public void testGetStoptagsJA2() {
        List<String> actual = StoptagsExcludeUDF.getStoptags(StoptagsExcludeUDF.STOPTAGS_JA,
            new String[] {"形容詞", "フィラー"});
        String[] expected = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                //"フィラー", 
                "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual);
    }

    @Test
    public void testGetStoptagsJa3() {
        List<String> actual = StoptagsExcludeUDF.getStoptags(StoptagsExcludeUDF.STOPTAGS_JA,
            new String[] {"形容詞", "フィラー", "名詞-固有名詞", "名詞-数"});
        String[] expected = new String[] {"名詞", "名詞-一般",
                // "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                // "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                // "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", 
                "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能", "名詞-サ変接続", "名詞-形容動詞語幹",
                // "名詞-数",                 
                "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能", "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊",
                "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般", "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続",
                "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能", "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的",
                "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞", "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続",
                "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                //"フィラー", 
                "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual);
    }

    @Test
    public void testGetStoptagsJaContainsUnmatchedEntry() {
        List<String> actual = StoptagsExcludeUDF.getStoptags(StoptagsExcludeUDF.STOPTAGS_JA,
            new String[] {"形容詞", "フィラー", "名詞-非"});
        String[] expected = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                //"フィラー", 
                "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual);
    }

    @Test
    public void testOneArgument() throws IOException, HiveException {
        StoptagsExcludeUDF udf = new StoptagsExcludeUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector)});

        List<String> actual = udf.evaluate(new DeferredObject[] {new GenericUDF.DeferredJavaObject(
            Arrays.asList(new Text("形容詞"), new Text("フィラー")))});
        String[] expected = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                //"フィラー", 
                "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual);

        actual = udf.evaluate(new DeferredObject[] {
                new GenericUDF.DeferredJavaObject(Arrays.asList(new Text("形容詞")))});
        expected = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                "フィラー", "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual);

        udf.close();
    }

    @Test
    public void testOneConstArgument() throws IOException, HiveException {
        StoptagsExcludeUDF udf = new StoptagsExcludeUDF();

        udf.initialize(
            new ObjectInspector[] {ObjectInspectorFactory.getStandardConstantListObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                Arrays.asList(new Text("形容詞"), new Text("フィラー")))});

        List<String> actual1 = udf.evaluate(new DeferredObject[] {});

        String[] expected = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                //"フィラー", 
                "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual1);

        List<String> actual2 = udf.evaluate(new DeferredObject[] {});
        Assert.assertSame(actual2, actual1);

        udf.close();
    }

    @Test
    public void testTwoArguments() throws IOException, HiveException {
        StoptagsExcludeUDF udf = new StoptagsExcludeUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector),
                HiveUtils.getConstStringObjectInspector("ja")});

        List<String> actual = udf.evaluate(new DeferredObject[] {new GenericUDF.DeferredJavaObject(
            Arrays.asList(new Text("形容詞"), new Text("フィラー")))});
        String[] expected = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                // "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", 
                "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞", "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用",
                "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞", "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞",
                "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊", "助動詞", "感動詞", "記号", "記号-一般",
                "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉", "記号-アルファベット", "その他", "その他-間投",
                //"フィラー", 
                "非言語音", "語断片", "未知語"};
        Arrays.sort(expected);
        Assert.assertEquals(Arrays.asList(expected), actual);

        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testTwoArgumentsUnsupportedLang() throws IOException, HiveException {
        StoptagsExcludeUDF udf = new StoptagsExcludeUDF();

        udf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableStringObjectInspector),
                HiveUtils.getConstStringObjectInspector("kr")});

        udf.close();
    }

    @Test
    public void testSerialization() throws IOException, HiveException {
        StoptagsExcludeUDF udf = new StoptagsExcludeUDF();

        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector)});

        // serialization after evaluation
        byte[] serialized = TestUtils.serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, StoptagsExcludeUDF.class);

        udf.close();
    }

}
