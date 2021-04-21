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

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class KuromojiNEologdUDFTest {

    @Test
    public void testNoArgument() throws IOException, HiveException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[0];
        udf.initialize(argOIs);
        Object result = udf.evaluate(new DeferredObject[0]);
        Assert.assertNotNull(result);
        udf.close();
    }

    @Test
    public void testOneArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testTwoArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        udf.initialize(argOIs);
        udf.close();
    }

    public void testExpectedMode() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("normal"));
        udf.initialize(argOIs);
        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidMode() throws IOException, HiveException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("unsupported mode"));
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("こんにちは。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        udf.evaluate(args);

        udf.close();
    }

    @Test
    public void testThreeArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[3];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testFourArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[4];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testFiveArgumentArray() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // userDictUrl
        argOIs[4] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testFiveArgumentString() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // userDictUrl
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testEvaluateReturnsVersion() throws IOException, HiveException {
        KuromojiNEologdUDF udf = new KuromojiNEologdUDF();
        DeferredObject[] args = new DeferredObject[0];
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(1, tokens.size());
        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testEvaluateOneRow() throws IOException, HiveException {
        KuromojiNEologdUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("10日放送の「中居正広のミになる図書館」（テレビ朝日系）で、SMAPの中居正広が、篠原信一の過去の勘違いを明かす一幕があった。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(12, tokens.size());
        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testEvaluateTwoRows() throws IOException, HiveException {
        KuromojiNEologdUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("10日放送の「中居正広のミになる図書館」（テレビ朝日系）で、SMAPの中居正広が、篠原信一の過去の勘違いを明かす一幕があった。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(12, tokens.size());

        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("きゃりーぱみゅぱみゅ。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(1, tokens.size());

        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testEvaluateUserDictArray() throws IOException, HiveException {
        KuromojiNEologdUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // userDictArray (from https://raw.githubusercontent.com/atilika/kuromoji/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt)
        List<String> userDict = new ArrayList<String>();
        userDict.add("日本経済新聞,日本 経済 新聞,ニホン ケイザイ シンブン,カスタム名詞");
        userDict.add("関西国際空港,関西 国際 空港,カンサイ コクサイ クウコウ,テスト名詞");
        argOIs[4] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, userDict);
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("日本経済新聞。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };

        List<Text> tokens = (List<Text>) udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(3, tokens.size());

        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test(expected = UDFArgumentException.class)
    public void testEvaluateInvalidUserDictURL() throws IOException, HiveException {
        KuromojiNEologdUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // userDictUrl
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("http://google.com/"));
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("こんにちは。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };

        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);

        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testEvaluateUserDictURL() throws IOException, HiveException {
        KuromojiNEologdUDF udf = new KuromojiNEologdUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // userDictUrl (Kuromoji official sample user defined dict on GitHub)
        // e.g., "日本経済新聞" will be "日本", "経済", and "新聞"
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text(
                "https://raw.githubusercontent.com/atilika/kuromoji/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt"));
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。日本経済新聞。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };

        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(8, tokens.size());

        udf.close();
    }
}
