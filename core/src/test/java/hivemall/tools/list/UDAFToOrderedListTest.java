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
package hivemall.tools.list;

import hivemall.tools.list.UDAFToOrderedList.UDAFToOrderedListEvaluator;
import hivemall.tools.list.UDAFToOrderedList.UDAFToOrderedListEvaluator.QueueAggregationBuffer;

import java.util.List;

import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class UDAFToOrderedListTest {

    private UDAFToOrderedListEvaluator evaluator;
    private QueueAggregationBuffer agg;

    @Before
    public void setUp() throws Exception {
        this.evaluator = new UDAFToOrderedListEvaluator();
        this.agg = (QueueAggregationBuffer) evaluator.getNewAggregationBuffer();
    }

    @Test
    public void testNaturalOrder() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(3, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
        Assert.assertEquals("candy", res.get(2));
    }

    @Test
    public void testReverseOrder() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-reverse_order")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(3, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
        Assert.assertEquals("apple", res.get(2));
    }

    @Test
    public void testTopK() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k 2")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testReverseTopK() throws Exception {
        // = tail-k
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k 2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testTailK() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k -2")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testReverseTailK() throws Exception {
        // = top-k
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k -2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testNaturalOrderWithKey() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.7};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(3, res.size());
        Assert.assertEquals("apple", res.get(0));
        if (res.get(1) == "banana") { // duplicated key (0.7)
            Assert.assertEquals("candy", res.get(2));
        } else {
            Assert.assertEquals("banana", res.get(2));
        }
    }

    @Test
    public void testReverseOrderWithKey() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-reverse_order")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.7};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(3, res.size());
        if (res.get(0) == "banana") { // duplicated key (0.7)
            Assert.assertEquals("candy", res.get(1));
        } else {
            Assert.assertEquals("banana", res.get(1));
        }
        Assert.assertEquals("apple", res.get(2));
    }

    @Test
    public void testTopKWithKey() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k 2")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testReverseTopKWithKey() throws Exception {
        // = tail-k
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k 2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testTailKWithKey() throws Exception {
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k -2")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testReverseTailKWithKey() throws Exception {
        // = top-k
        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k -2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        List<Object> res = evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

}
