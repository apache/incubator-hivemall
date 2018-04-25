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
import java.util.List;

import hivemall.TestUtils;

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

public class SmartcnUDFTest {

    @Test
    public void testOneArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new SmartcnUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testTwoArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new SmartcnUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // stopWords
        argOIs[1] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testEvaluateOneRow() throws IOException, HiveException {
        SmartcnUDF udf = new SmartcnUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("Smartcn为Apache2.0协议的开源中文分词系统，Java语言编写，修改的中科院计算所ICTCLAS分词系统。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);
        Assert.assertNotNull(tokens);
        udf.close();
    }

    @Test
    public void testSerialization() throws IOException, HiveException {
        final SmartcnUDF udf = new SmartcnUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        // serialization after initialization
        byte[] serialized = TestUtils.serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, SmartcnUDF.class);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("Smartcn为Apache2.0协议的开源中文分词系统，Java语言编写，修改的中科院计算所ICTCLAS分词系统。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = udf.evaluate(args);

        // serialization after evaluation
        serialized = TestUtils.serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, SmartcnUDF.class);

        udf.close();
    }
}
