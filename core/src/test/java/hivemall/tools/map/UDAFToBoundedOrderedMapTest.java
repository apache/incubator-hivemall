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

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.SortedMap;

@SuppressWarnings("deprecation")
public class UDAFToBoundedOrderedMapTest {
    UDAFToBoundedOrderedMap udaf;
    GenericUDAFEvaluator evaluator;
    ObjectInspector[] inputOIs;
    UDAFToBoundedOrderedMap.BoundedOrderedMapEvaluator.BoundedMapAggregationBuffer agg;

    @Before
    public void setUp() throws Exception {
        this.udaf = new UDAFToBoundedOrderedMap();

        inputOIs = new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.DOUBLE),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.STRING),
                PrimitiveObjectInspectorFactory.getPrimitiveJavaObjectInspector(PrimitiveObjectInspector.PrimitiveCategory.INT)};
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidMapSize() throws Exception {
        evaluator = new UDAFToBoundedOrderedMap.BoundedOrderedMapEvaluator();
        agg = (UDAFToBoundedOrderedMap.BoundedOrderedMapEvaluator.BoundedMapAggregationBuffer) evaluator.getNewAggregationBuffer();

        // should be sorted by scores in a descending order
        final double[] keys = new double[] {0.7, 0.5, 0.8};
        final String[] values = new String[] {"banana", "apple", "candy"};
        int size = 0;

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < keys.length; i++) {
            evaluator.iterate(agg, new Object[] {keys[i], values[i], size});
        }
    }


    @Test
    public void testNaturalOrder() throws Exception {
        evaluator = new UDAFToBoundedOrderedMap.BoundedOrderedMapEvaluator();
        agg = (UDAFToBoundedOrderedMap.BoundedOrderedMapEvaluator.BoundedMapAggregationBuffer) evaluator.getNewAggregationBuffer();

        // should be sorted by scores in a descending order
        final double[] keys = new double[] {0.7, 0.5, 0.8};
        final String[] values = new String[] {"banana", "apple", "candy"};
        int size = 2;

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < keys.length; i++) {
            evaluator.iterate(agg, new Object[] {keys[i], values[i], size});
        }

        SortedMap<Object, Object> res = (SortedMap<Object, Object>) agg.get();
        Object[] sortedValues = res.values().toArray();

        Assert.assertEquals(size, sortedValues.length);
        Assert.assertEquals("apple", sortedValues[0]);
        Assert.assertEquals("banana", sortedValues[1]);
    }

    @Test
    public void testReverseOrder() throws Exception {
        evaluator = new UDAFToBoundedOrderedMap.BoundedReverseOrderedMapEvaluator();
        agg = (UDAFToBoundedOrderedMap.BoundedOrderedMapEvaluator.BoundedMapAggregationBuffer) evaluator.getNewAggregationBuffer();

        // should be sorted by scores in a descending order
        final double[] keys = new double[] {0.7, 0.5, 0.8};
        final String[] values = new String[] {"banana", "apple", "candy"};
        int size = 2;

        evaluator.init(GenericUDAFEvaluator.Mode.PARTIAL1, inputOIs);
        evaluator.reset(agg);

        for (int i = 0; i < keys.length; i++) {
            evaluator.iterate(agg, new Object[] {keys[i], values[i], size});
        }

        SortedMap<Object, Object> res = (SortedMap<Object, Object>) agg.get();
        Object[] sortedValues = res.values().toArray();

        Assert.assertEquals(size, sortedValues.length);
        Assert.assertEquals("candy", sortedValues[0]);
        Assert.assertEquals("banana", sortedValues[1]);
    }

}
