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
package hivemall.knn.similarity;

import hivemall.utils.lang.mutable.MutableInt;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

public class DIMSUMMapperUDTFTest {
    private static final boolean DEBUG = false;

    @Test
    public void testIntFeature() throws HiveException {
        DIMSUMMapperUDTF udtf = new DIMSUMMapperUDTF();

        ObjectInspector[] argOIs = new ObjectInspector[] {
            ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector),
            ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
            ObjectInspectorUtils.getConstantObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                "-threshold 0.5 -disable_symmetric_output -int_feature")};

        final MutableInt count = new MutableInt(0);
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                count.addValue(1);
            }
        };
        udtf.setCollector(collector);
        udtf.initialize(argOIs);

        final int[] itemIDs = new int[] {1, 2, 3};
        final double[][] R = new double[][] {{1, 2, 3}, {1, 2, 3}};

        int numUsers = R.length;
        int numItems = itemIDs.length;

        List<String> user1 = new ArrayList<String>();
        for (int j = 0; j < numItems; j++) {
            double r = R[0][j];
            if (r != 0.d) {
                user1.add(itemIDs[j] + ":" + r);
            }
        }

        List<String> user2 = new ArrayList<String>();
        for (int j = 0; j < numItems; j++) {
            double r = R[1][j];
            if (r != 0.d) {
                user2.add(itemIDs[j] + ":" + r);
            }
        }

        Map<Integer, Double> norms = new HashMap<Integer, Double>();
        for (int j = 0; j < numItems; j++) {
            double sim = 0.d;
            for (int i = 0; i < numUsers; i++) {
                sim += R[i][j] * R[i][j];
            }
            norms.put(itemIDs[j], Math.sqrt(sim));
        }

        // at most 3 emits (<1, 2>, <1, 3>, <2, 3>) per row
        udtf.process(new Object[]{ user1, norms });
        udtf.process(new Object[]{ user2, norms });
        Assert.assertTrue(count.getValue() <= 3 * 2);
    }

    @Test
    public void testStringFeature() throws HiveException {
        DIMSUMMapperUDTF udtf = new DIMSUMMapperUDTF();

        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorFactory.getStandardMapObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-threshold 0.5")};

        final MutableInt count = new MutableInt(0);
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                count.addValue(1);
            }
        };
        udtf.setCollector(collector);
        udtf.initialize(argOIs);

        final String[] itemIDs = new String[] {"i1", "i2", "i3"};
        final double[][] R = new double[][] {{1, 2, 3}, {1, 2, 3}};

        int numUsers = R.length;
        int numItems = itemIDs.length;

        List<String> user1 = new ArrayList<String>();
        for (int j = 0; j < numItems; j++) {
            double r = R[0][j];
            if (r != 0.d) {
                user1.add(itemIDs[j] + ":" + r);
            }
        }

        List<String> user2 = new ArrayList<String>();
        for (int j = 0; j < numItems; j++) {
            double r = R[1][j];
            if (r != 0.d) {
                user2.add(itemIDs[j] + ":" + r);
            }
        }

        Map<String, Double> norms = new HashMap<String, Double>();
        for (int j = 0; j < numItems; j++) {
            double sim = 0.d;
            for (int i = 0; i < numUsers; i++) {
                sim += R[i][j] * R[i][j];
            }
            norms.put(itemIDs[j], Math.sqrt(sim));
        }

        // at most 6 emits (<i1, i2>, <i2, i1>, <i1, i3>, <i3, i2>, <i2, i3>, <i3, i2>) per row
        udtf.process(new Object[]{ user1, norms });
        udtf.process(new Object[]{ user2, norms });
        Assert.assertTrue(count.getValue() <= 6 * 2);
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }
}
