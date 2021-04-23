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
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class TokenizeKoUDFTest {

    private TokenizeKoUDF udf;

    @Before
    public void setUp() {
        this.udf = new TokenizeKoUDF();
    }

    @Test
    public void testNoArgument() throws IOException, HiveException {
        GenericUDF udf = new TokenizeKoUDF();
        ObjectInspector[] argOIs = new ObjectInspector[0];
        udf.initialize(argOIs);
        Object result = udf.evaluate(new DeferredObject[0]);
        Assert.assertNotNull(result);
        udf.close();
    }

    @Test
    public void testShowHelp() throws IOException {
        GenericUDF udf = new TokenizeKoUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        argOIs[1] = HiveUtils.getConstStringObjectInspector("-help");
        try {
            udf.initialize(argOIs);
            Assert.fail("should not reach here");
        } catch (UDFArgumentException e) {
            String errmsg = e.getMessage();
            Assert.assertTrue(errmsg.contains("usage:"));
        } finally {
            udf.close();
        }
    }

    @Test
    public void testOneArgument() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[1];
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("소설 무궁화꽃이 피었습니다.");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(5, tokens.size());
        Assert.assertEquals("소설 무궁 화 꽃 피", getString(tokens));

        udf.close();
    }

    @Test
    public void testNullUserDict() throws HiveException, IOException {
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
        // userDict
        argOIs[4] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);


        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("I enjoy C++ programming.");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(4, tokens.size());
        Assert.assertEquals("i enjoy c programming", getString(tokens));

        udf.close();
    }

    @Test
    public void testNullMode() throws UDFArgumentException, IOException {
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

    @Test
    public void testModeMixed() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("mixed"));
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("소설 무궁화꽃이 피었습니다.");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(6, tokens.size());
        Assert.assertEquals("소설 무궁화 무궁 화 꽃 피", getString(tokens));

        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidMode() throws IOException, HiveException {
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("unsupported mode"));
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testUserDictArray() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("mixed"));
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // userDict
        argOIs[4] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, Arrays.asList("C++"));
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("I enjoy C++ programming.");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(4, tokens.size());
        Assert.assertEquals("i enjoy c++ programming", getString(tokens));

        udf.close();
    }

    @Test
    public void testUserDictUrl() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("discard"));
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // userDict
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text(
                "https://raw.githubusercontent.com/apache/lucene/044d152d954f1e22aac5a53792011da54c680617/lucene/analysis/nori/src/test/org/apache/lucene/analysis/ko/userdict.txt"));

        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("나는 c++ 프로그래밍을 즐긴다");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(4, tokens.size());
        Assert.assertEquals("나 c++ 프로그래밍 즐기", getString(tokens));

        udf.close();
    }

    @Test
    public void testStopTags() throws HiveException, IOException {
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
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, Arrays.asList("E", "VV"));

        // userDict
        argOIs[4] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);

        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("소설 무궁화꽃이 피었습니다.");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(5, tokens.size());
        Assert.assertEquals("소설 무궁 화 꽃 이", getString(tokens));

        udf.close();
    }


    @Test
    public void testWithoutDictCplusplus() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[4];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("discard"));
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);

        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("나는 c++ 프로그래밍을 즐긴다");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(4, tokens.size());
        Assert.assertEquals("나 c 프로그래밍 즐기", getString(tokens));

        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidStopTag() throws UDFArgumentException, IOException {
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
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, Arrays.asList("E", "?"));
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testOutputUnknownUnigramsTrue() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[4];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;

        // opts
        argOIs[1] = HiveUtils.getConstStringObjectInspector("-mode discard -outputUnknownUnigrams"); // mode        

        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);

        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);

        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("Hello, world.");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(10, tokens.size());
        Assert.assertEquals("h e l l o w o r l d", getString(tokens));

        udf.close();
    }

    @Test
    public void testOutputUnknownUnigramsFalse() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[4];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;

        // opts
        argOIs[1] = HiveUtils.getConstStringObjectInspector("-mode discard"); // mode        

        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);

        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);

        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("Hello, world.");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(2, tokens.size());
        Assert.assertEquals("hello world", getString(tokens));

        udf.close();
    }

    private static String getString(List<Text> tokens) {
        StringBuilder sb = new StringBuilder();
        for (Text token : tokens) {
            sb.append(token.toString()).append(" ");
        }
        return sb.toString().trim();
    }

}
