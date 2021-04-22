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
import org.apache.hadoop.io.BooleanWritable;
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
    public void test() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[1];
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("소설 무궁화꽃이 피었습니다.");
            }

            @Override
            public void prepare(int arg) throws HiveException {
            }
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(5, tokens.size());
        Assert.assertEquals("소설 무궁 화 꽃 피", getString(tokens));

        udf.close();
    }

    @Test
    public void testNullUserList() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("I enjoy C++ programming.");
            }

            @Override
            public void prepare(int arg) throws HiveException {
            }
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(4, tokens.size());
        Assert.assertEquals("i enjoy c programming", getString(tokens));

        udf.close();
    }

    @Test
    public void testNullMode() throws UDFArgumentException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[3];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testMode() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[3];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("mixed"));
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("소설 무궁화꽃이 피었습니다.");
            }

            @Override
            public void prepare(int arg) throws HiveException {
            }
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(6, tokens.size());
        Assert.assertEquals("소설 무궁화 무궁 화 꽃 피", getString(tokens));

        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidMode() throws IOException, HiveException {
        ObjectInspector[] argOIs = new ObjectInspector[3];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("unsupported mode"));
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testNonnullUserList() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[3];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, Arrays.asList("C++"));
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("mixed"));
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("I enjoy C++ programming.");
            }

            @Override
            public void prepare(int arg) throws HiveException {
            }
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(4, tokens.size());
        Assert.assertEquals("i enjoy c++ programming", getString(tokens));

        udf.close();
    }

    @Test
    public void testStopTags() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[4];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, Arrays.asList("E", "VV"));
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("소설 무궁화꽃이 피었습니다.");
            }

            @Override
            public void prepare(int arg) throws HiveException {
            }
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(5, tokens.size());
        Assert.assertEquals("소설 무궁 화 꽃 이", getString(tokens));

        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidStopTag() throws UDFArgumentException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[4];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, Arrays.asList("E", "?"));
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testOutputUnknownUnigramsTrue() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // outputUnknowUnigrams
        PrimitiveTypeInfo booleanType = new PrimitiveTypeInfo();
        booleanType.setTypeName("boolean");
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            booleanType, new BooleanWritable(true));
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("Hello, world.");
            }

            @Override
            public void prepare(int arg) throws HiveException {
            }
        };
        List<Text> tokens = udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(10, tokens.size());
        Assert.assertEquals("h e l l o w o r l d", getString(tokens));

        udf.close();
    }

    @Test
    public void testOutputUnknownUnigramsFalse() throws HiveException, IOException {
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // userDict
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[2] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // outputUnknowUnigrams
        PrimitiveTypeInfo booleanType = new PrimitiveTypeInfo();
        booleanType.setTypeName("boolean");
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            booleanType, new BooleanWritable(false));
        udf.initialize(argOIs);

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[1];
        args[0] = new GenericUDF.DeferredObject() {
            public Text get() throws HiveException {
                return new Text("Hello, world.");
            }

            @Override
            public void prepare(int arg) throws HiveException {
            }
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
