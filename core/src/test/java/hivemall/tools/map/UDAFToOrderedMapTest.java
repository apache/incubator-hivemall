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
package hivemall.tools.map;

import hivemall.tools.map.UDAFToOrderedMap.AscendingMapEvaluator;
import hivemall.tools.map.UDAFToOrderedMap.DescendingMapEvaluator;

import java.util.SortedMap;

import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class UDAFToOrderedMapTest {

    @Test
    public void testNaturalOrder() throws Exception {
        AscendingMapEvaluator evaluator = new AscendingMapEvaluator(0);
        AscendingMapEvaluator.MapAggregationBuffer agg = (AscendingMapEvaluator.MapAggregationBuffer) evaluator.getNewAggregationBuffer();

        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.javaStringObjectInspector};

        final double[] keys = new double[] {0.7, 0.5, 0.8};
        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < keys.length; i++) {
            evaluator.iterate(agg, new Object[] {keys[i], values[i]});
        }

        SortedMap<Object, Object> res = (SortedMap<Object, Object>) evaluator.terminate(agg);
        Object[] sortedValues = res.values().toArray();

        Assert.assertEquals(3, sortedValues.length);
        Assert.assertEquals("apple", sortedValues[0]);
        Assert.assertEquals("banana", sortedValues[1]);
        Assert.assertEquals("candy", sortedValues[2]);

        evaluator.close();
    }

    @Test
    public void testReverseOrder() throws Exception {
        DescendingMapEvaluator evaluator = new DescendingMapEvaluator(0);
        DescendingMapEvaluator.MapAggregationBuffer agg = (DescendingMapEvaluator.MapAggregationBuffer) evaluator.getNewAggregationBuffer();

        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaBooleanObjectInspector};

        final double[] keys = new double[] {0.7, 0.5, 0.8};
        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < keys.length; i++) {
            evaluator.iterate(agg, new Object[] {keys[i], values[i]});
        }

        SortedMap<Object, Object> res = (SortedMap<Object, Object>) evaluator.terminate(agg);
        Object[] sortedValues = res.values().toArray();

        Assert.assertEquals(3, sortedValues.length);
        Assert.assertEquals("candy", sortedValues[0]);
        Assert.assertEquals("banana", sortedValues[1]);
        Assert.assertEquals("apple", sortedValues[2]);

        evaluator.close();
    }

    @Test
    public void testTopK() throws Exception {
        int size = 2;
        DescendingMapEvaluator evaluator = new DescendingMapEvaluator(size);
        DescendingMapEvaluator.MapAggregationBuffer agg = (DescendingMapEvaluator.MapAggregationBuffer) evaluator.getNewAggregationBuffer();

        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaIntObjectInspector};

        final double[] keys = new double[] {0.7, 0.5, 0.8};
        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < keys.length; i++) {
            evaluator.iterate(agg, new Object[] {keys[i], values[i], size});
        }

        SortedMap<Object, Object> res = (SortedMap<Object, Object>) evaluator.terminate(agg);
        Object[] sortedValues = res.values().toArray();

        Assert.assertEquals(size, sortedValues.length);
        Assert.assertEquals("candy", sortedValues[0]);
        Assert.assertEquals("banana", sortedValues[1]);

        evaluator.close();
    }

    @Test
    public void testTailK() throws Exception {
        int size = -2;
        AscendingMapEvaluator evaluator = new AscendingMapEvaluator(Math.abs(size));
        AscendingMapEvaluator.MapAggregationBuffer agg = (AscendingMapEvaluator.MapAggregationBuffer) evaluator.getNewAggregationBuffer();

        ObjectInspector[] inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaIntObjectInspector};

        final double[] keys = new double[] {0.7, 0.5, 0.8};
        final String[] values = new String[] {"banana", "apple", "candy"};

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < keys.length; i++) {
            evaluator.iterate(agg, new Object[] {keys[i], values[i], size});
        }

        SortedMap<Object, Object> res = (SortedMap<Object, Object>) evaluator.terminate(agg);
        Object[] sortedValues = res.values().toArray();

        Assert.assertEquals(Math.abs(size), sortedValues.length);
        Assert.assertEquals("apple", sortedValues[0]);
        Assert.assertEquals("banana", sortedValues[1]);

        evaluator.close();
    }

}
