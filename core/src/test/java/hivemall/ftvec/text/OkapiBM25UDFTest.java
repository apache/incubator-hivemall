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
package hivemall.ftvec.text;

import static org.junit.Assert.assertEquals;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Before;
import org.junit.Test;

public class OkapiBM25UDFTest {

    private static final double EPSILON = 1e-8;
    private static final GenericUDF.DeferredJavaObject VALID_TERM_FREQ =
            new GenericUDF.DeferredJavaObject(new Integer(3));
    private static final GenericUDF.DeferredJavaObject VALID_DOC_LEN =
            new GenericUDF.DeferredJavaObject(new Integer(9));
    private static final GenericUDF.DeferredJavaObject VALID_AVG_DOC_LEN =
            new GenericUDF.DeferredJavaObject(new Double(10.35));
    private static final GenericUDF.DeferredJavaObject VALID_NUM_DOCS =
            new GenericUDF.DeferredJavaObject(new Integer(20));
    private static final GenericUDF.DeferredJavaObject VALID_NUM_DOCS_WITH_TERM =
            new GenericUDF.DeferredJavaObject(new Integer(5));

    private OkapiBM25UDF udf = null;


    @Before
    public void init() throws Exception {
        udf = new OkapiBM25UDF();
    }

    @Test
    public void testEvaluate() throws Exception {

        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[] {VALID_TERM_FREQ,
                VALID_DOC_LEN, VALID_AVG_DOC_LEN, VALID_NUM_DOCS, VALID_NUM_DOCS_WITH_TERM};

        DoubleWritable expected = WritableUtils.val(0.940637195691);
        DoubleWritable actual = udf.evaluate(args);
        assertEquals(expected.get(), actual.get(), EPSILON);
    }

    @Test
    public void testEvaluateWithCustomK1() throws Exception {

        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    HiveUtils.getConstStringObjectInspector("-k1 1.5")});

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[] {VALID_TERM_FREQ,
                VALID_DOC_LEN, VALID_AVG_DOC_LEN, VALID_NUM_DOCS, VALID_NUM_DOCS_WITH_TERM};

        DoubleWritable expected = WritableUtils.val(1.00244958206);
        DoubleWritable actual = udf.evaluate(args);
        assertEquals(expected.get(), actual.get(), EPSILON);
    }

    @Test
    public void testEvaluateWithCustomB() throws Exception {

        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    HiveUtils.getConstStringObjectInspector("-b 0.8")});

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[] {VALID_TERM_FREQ,
                VALID_DOC_LEN, VALID_AVG_DOC_LEN, VALID_NUM_DOCS, VALID_NUM_DOCS_WITH_TERM};

        DoubleWritable expected = WritableUtils.val(0.942443797219);
        DoubleWritable actual = udf.evaluate(args);
        assertEquals(expected.get(), actual.get(), EPSILON);
    }

    @Test(expected = HiveException.class)
    public void testInputArgIsNull() throws Exception {

        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args =
                new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(null),
                        VALID_DOC_LEN, VALID_AVG_DOC_LEN, VALID_NUM_DOCS, VALID_NUM_DOCS_WITH_TERM};

        udf.evaluate(args);
    }

    @Test(expected = HiveException.class)
    public void testTermFrequencyIsNegative() throws Exception {
        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args =
                new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(new Integer(-1)),
                        VALID_DOC_LEN, VALID_AVG_DOC_LEN, VALID_NUM_DOCS, VALID_NUM_DOCS_WITH_TERM};

        udf.evaluate(args);
    }

    @Test(expected = HiveException.class)
    public void testDocLengthIsLessThanOne() throws Exception {
        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[] {VALID_TERM_FREQ,
                new GenericUDF.DeferredJavaObject(new Integer(0)), VALID_AVG_DOC_LEN,
                VALID_NUM_DOCS, VALID_NUM_DOCS_WITH_TERM};

        udf.evaluate(args);
    }

    @Test(expected = HiveException.class)
    public void testAvgDocLengthIsNegative() throws Exception {
        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[] {VALID_TERM_FREQ,
                VALID_DOC_LEN, new GenericUDF.DeferredJavaObject(new Double(-10)), VALID_NUM_DOCS,
                VALID_NUM_DOCS_WITH_TERM};

        udf.evaluate(args);
    }

    @Test(expected = HiveException.class)
    public void testAvgDocLengthIsZero() throws Exception {
        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[] {VALID_TERM_FREQ,
                VALID_DOC_LEN, new GenericUDF.DeferredJavaObject(new Double(0.0)), VALID_NUM_DOCS,
                VALID_NUM_DOCS_WITH_TERM};

        udf.evaluate(args);
    }

    @Test(expected = HiveException.class)
    public void testNumDocsIsLessThanOne() throws Exception {
        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args = new GenericUDF.DeferredObject[] {VALID_TERM_FREQ,
                VALID_DOC_LEN, VALID_AVG_DOC_LEN, new GenericUDF.DeferredJavaObject(new Integer(0)),
                VALID_NUM_DOCS_WITH_TERM};

        udf.evaluate(args);
    }

    @Test(expected = HiveException.class)
    public void testNumDocsWithTermIsLessThanOne() throws Exception {
        initializeUDFWithoutOptions();

        GenericUDF.DeferredObject[] args =
                new GenericUDF.DeferredObject[] {VALID_TERM_FREQ, VALID_DOC_LEN, VALID_AVG_DOC_LEN,
                        VALID_NUM_DOCS, new GenericUDF.DeferredJavaObject(new Integer(0))};

        udf.evaluate(args);
    }

    private void initializeUDFWithoutOptions() throws Exception {
        udf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector});
    }
}
