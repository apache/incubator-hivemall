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

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
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
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(3, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
        Assert.assertEquals("candy", res.get(2));
    }

    @Test
    public void testIntegerNaturalOrder() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector};

        final Integer[] values = new Integer[] {3, -1, 4, 2, 5};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(5, res.size());
        Assert.assertEquals(-1, res.get(0));
        Assert.assertEquals(2, res.get(1));
        Assert.assertEquals(3, res.get(2));
        Assert.assertEquals(4, res.get(3));
        Assert.assertEquals(5, res.get(4));
    }

    @Test
    public void testDoubleNaturalOrder() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector};

        final Double[] values = new Double[] {3.1d, -1.1d, 4.1d, 2.1d, 5.1d};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(5, res.size());
        Assert.assertEquals(-1.1d, res.get(0));
        Assert.assertEquals(2.1d, res.get(1));
        Assert.assertEquals(3.1d, res.get(2));
        Assert.assertEquals(4.1d, res.get(3));
        Assert.assertEquals(5.1d, res.get(4));
    }

    @Test
    public void testReverseOrder() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-reverse_order")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(3, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
        Assert.assertEquals("apple", res.get(2));
    }

    @Test
    public void testTopK() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k 2")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testTop2IntNuturalOrder() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k 2")};

        final Integer[] values = new Integer[] {3, -1, 4, 4, 2, 5};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals(5, res.get(0));
        Assert.assertEquals(4, res.get(1));
    }

    @Test
    public void testReverseTopK() throws Exception {
        // = tail-k
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testTailK() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-k -2")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testReverseTailK() throws Exception {
        // = top-k
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k -2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testNaturalOrderWithKey() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.7};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

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
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-reverse_order")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.7};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        @SuppressWarnings("unchecked")
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
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
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

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testReverseTopKWithKey() throws Exception {
        // = tail-k
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testTailKWithKey() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
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

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testReverseTailKWithKey() throws Exception {
        // = top-k
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k -2 -reverse")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(2, res.size());
        Assert.assertEquals("candy", res.get(0));
        Assert.assertEquals("banana", res.get(1));
    }

    @Test
    public void testNullOnly() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector};

        final String[] values = new String[] {null, null, null};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertNull(res);
    }

    @Test
    public void testNullMixed() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaDoubleObjectInspector};

        final String[] values = new String[] {"banana", "apple", null, "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i]});
        }

        @SuppressWarnings("unchecked")
        List<Object> res = (List<Object>) evaluator.terminate(agg);

        Assert.assertEquals(3, res.size());
        Assert.assertEquals("apple", res.get(0));
        Assert.assertEquals("banana", res.get(1));
        Assert.assertEquals("candy", res.get(2));
    }

    @Test
    public void testKVMapOption() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -kv_map")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals("candy", map.get(0.8d));
        Assert.assertEquals("banana", map.get(0.7d));
    }

    @Test
    public void testVKMapOption() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -vk_map")};

        final String[] values = new String[] {"banana", "apple", "candy"};
        final double[] keys = new double[] {0.7, 0.5, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals(0.8d, map.get("candy"));
        Assert.assertEquals(0.7d, map.get("banana"));
    }

    @Test
    public void testVKMapOptionBananaOverlap() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -vk_map")};

        final String[] values = new String[] {"banana", "banana", "candy"};
        final double[] keys = new double[] {0.7, 0.8, 0.81};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals(0.81d, map.get("candy"));
        Assert.assertEquals(0.8d, map.get("banana"));
    }

    @Test
    public void testVKMapOptionBananaOverlap2() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -vk_map")};

        final String[] values = new String[] {"banana", "banana", "candy"};
        final double[] keys = new double[] {0.8, 0.8, 0.7};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(1, map.size());

        Assert.assertEquals(0.8d, map.get("banana"));
    }

    @Test
    public void testVKMapOptionReverseOrderTop2() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k -2 -vk_map")};

        final String[] values = new String[] {"banana", "apple", "banana"};
        final double[] keys = new double[] {0.7, 0.6, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals(0.6d, map.get("apple"));
        Assert.assertEquals(0.7d, map.get("banana"));
    }

    @Test
    public void testVKMapOptionNaturalOrder() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-vk_map")};

        final String[] values = new String[] {"banana", "apple", "banana"};
        final double[] keys = new double[] {0.7, 0.6, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals(0.6d, map.get("apple"));
        Assert.assertEquals(0.7d, map.get("banana"));
    }

    @Test
    public void testVKMapOptionReverseOrder() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-reverse -vk_map")};

        final String[] values = new String[] {"banana", "apple", "banana"};
        final double[] keys = new double[] {0.7, 0.6, 0.8};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals(0.6d, map.get("apple"));
        Assert.assertEquals(0.8d, map.get("banana"));
    }

    @Test
    public void testVKMapOptionBananaOverlapReverseOrder() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k -2 -vk_map")};

        final String[] values = new String[] {"banana", "banana", "candy"};
        final double[] keys = new double[] {0.9, 0.8, 0.7};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals(0.7d, map.get("candy"));
        Assert.assertEquals(0.8d, map.get("banana"));
    }

    @Test
    public void testVKMapTop2() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -vk_map")};

        final int[] keys = new int[] {5, 3, 4, 2, 3};
        final String[] values = new String[] {"apple", "banana", "candy", "donut", "egg"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals(5, map.get("apple"));
        Assert.assertEquals(4, map.get("candy"));
    }

    @Test
    public void testKVMapTop2() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -kv_map")};

        final int[] keys = new int[] {5, 3, 4, 2, 3};
        final String[] values = new String[] {"apple", "banana", "candy", "donut", "egg"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(2, map.size());

        Assert.assertEquals("apple", map.get(5));
        Assert.assertEquals("candy", map.get(4));
    }

    @Test
    public void testTop4Dedup() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 4 -dedup -kv_map")};

        final int[] keys = new int[] {5, 3, 4, 1, 2, 4};
        final String[] values = new String[] {"apple", "banana", "candy", "donut", "egg", "candy"}; // 4:candy is duplicating

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(4, map.size());

        Assert.assertEquals("apple", map.get(5));
        Assert.assertEquals("candy", map.get(4));
        Assert.assertEquals("banana", map.get(3));
        Assert.assertEquals("egg", map.get(2));
        Assert.assertNull(map.get(1));
    }


    @Test
    public void testTop4NoDedup() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 4 -kv_map")};

        final int[] keys = new int[] {5, 3, 4, 1, 2, 4};
        final String[] values = new String[] {"apple", "banana", "candy", "donut", "egg", "candy"}; // 4:candy is duplicating

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < values.length; i++) {
            evaluator.iterate(agg, new Object[] {values[i], keys[i]});
        }

        Object result = evaluator.terminate(agg);

        Assert.assertEquals(LinkedHashMap.class, result.getClass());
        Map<?, ?> map = (Map<?, ?>) result;
        Assert.assertEquals(3, map.size());

        Assert.assertEquals("apple", map.get(5));
        Assert.assertEquals("candy", map.get(4));
        Assert.assertEquals("banana", map.get(3));
        Assert.assertNull(map.get(2));
        Assert.assertNull(map.get(1));
    }

    @Test(expected = UDFArgumentException.class)
    public void testKVandVKFail() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -kv_map -vk_map")};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
    }

    @Test(expected = UDFArgumentException.class)
    public void testKVMapReturnWithoutValue() throws Exception {
        ObjectInspector[] inputOIs =
                new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        ObjectInspectorUtils.getConstantObjectInspector(
                            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                            "-k 2 -kv_map")};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
    }

}
