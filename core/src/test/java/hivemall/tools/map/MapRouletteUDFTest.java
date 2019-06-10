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

import hivemall.TestUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;
import java.io.IOException;
import java.util.*;

/**
 * Unit test for {@link hivemall.tools.map.MapRouletteUDF}
 */
public class MapRouletteUDFTest {

    /**
     * Tom, Jerry, Amy, Wong, Zhao joined a roulette. Jerry has 0.2 weight to win. Zhao's weight is
     * highest, he has more chance to win. During data processing ,Tom 's weight was Lost. Algorithm
     * treat Tom 's weight as average. After 1000000 times of roulette, Zhao wins the most. Jerry
     * wins less than Zhao but more than the other.
     *
     * @throws HiveException fmp.initialize may throws UDFArgumentException when checking parameter,
     *         org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector#getMap(java.lang.Object)
     *         may throw Hive Exception
     */
    @Test
    public void testRoulette() throws HiveException {
        MapRouletteUDF fmp = new MapRouletteUDF();
        fmp.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)});
        Map<Object, Integer> solve = new HashMap<>();
        solve.put("Tom", 0);
        solve.put("Jerry", 0);
        solve.put("Amy", 0);
        solve.put("Wong", 0);
        solve.put("Zhao", 0);
        int T = 1000000;
        while (T-- > 0) {
            Map<Object, Double> m = new HashMap<>();
            m.put("Tom", null);
            m.put("Jerry", 0.2);
            m.put("Amy", 0.1);
            m.put("Wong", 0.1);
            m.put("Zhao", 0.5);
            GenericUDF.DeferredObject[] arguments =
                    new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(m)};
            Object key = fmp.evaluate(arguments);
            solve.put(key, solve.get(key) + 1);
        }
        List<Map.Entry<Object, Integer>> solveList = new ArrayList<>(solve.entrySet());
        Collections.sort(solveList, new KvComparator());
        Object highestSolve = solveList.get(solveList.size() - 1).getKey();
        Assert.assertEquals(highestSolve.toString(), "Zhao");
        Object secondarySolve = solveList.get(solveList.size() - 2).getKey();
        Assert.assertEquals(secondarySolve.toString(), "Jerry");
    }

    private static class KvComparator implements Comparator<Map.Entry<Object, Integer>> {

        @Override
        public int compare(Map.Entry<Object, Integer> o1, Map.Entry<Object, Integer> o2) {
            return o1.getValue().compareTo(o2.getValue());
        }
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        Map<Object, Double> m = new HashMap<>();
        m.put("Tom", 0.1);
        m.put("Jerry", 0.2);
        m.put("Amy", 0.1);
        m.put("Wong", 0.1);
        m.put("Zhao", null);

        TestUtils.testGenericUDFSerialization(MapRouletteUDF.class,
            new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)},
            new Object[] {m});
        byte[] serialized = TestUtils.serializeObjectByKryo(new MapRouletteUDFTest());
        TestUtils.deserializeObjectByKryo(serialized, MapRouletteUDFTest.class);
    }

    @Test
    public void testEmptyMapAndAllNullMap() throws HiveException {
        MapRouletteUDF udf = new MapRouletteUDF();
        Map<Object, Double> m = new HashMap<>();
        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)});
        GenericUDF.DeferredObject[] arguments =
                new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(m)};
        Assert.assertNull(udf.evaluate(arguments));
        m.put(null, null);
        arguments = new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(m)};
        Assert.assertNull(udf.evaluate(arguments));
    }

    @Test
    public void testOnlyOne() throws HiveException {
        MapRouletteUDF udf = new MapRouletteUDF();
        Map<Object, Double> m = new HashMap<>();
        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)});
        m.put("One", 324.6);
        GenericUDF.DeferredObject[] arguments =
                new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(m)};
        Assert.assertEquals(udf.evaluate(arguments), "One");
    }

    @Test
    public void testString() throws HiveException {
        MapRouletteUDF udf = new MapRouletteUDF();
        Map<Object, String> m = new HashMap<>();
        udf.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            PrimitiveObjectInspectorFactory.javaStringObjectInspector)});
        m.put("One", "0.7");
        GenericUDF.DeferredObject[] arguments =
                new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(m)};
        Assert.assertEquals(udf.evaluate(arguments), "One");
    }
}
