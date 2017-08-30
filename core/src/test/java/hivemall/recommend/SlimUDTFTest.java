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
package hivemall.recommend;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class SlimUDTFTest {
    @Test
    public void testAllSamples() throws HiveException {
        SlimUDTF slim = new SlimUDTF();
        ObjectInspector itemIOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector itemJOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;

        ObjectInspector itemIRatesOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);
        ObjectInspector itemJRatesOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);
        ObjectInspector topKRatesOfIOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector));
        ObjectInspector optionArgumentOI = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-l2 0.01 -l1 0.01");

        ObjectInspector[] argOIs = {itemIOI, itemIRatesOI, topKRatesOfIOI, itemJOI, itemJRatesOI,
                optionArgumentOI};

        slim.initialize(argOIs);
        int numUser = 4;
        int numItem = 5;

        double[][] data = { {1., 4., 0., 0., 0.}, {0., 3., 0., 1., 2.}, {2., 2., 0., 0., 3.},
                {0., 1., 1., 0., 0.},};

        for (int iter = 0; iter < 20; iter++) {
            for (int i = 0; i < numItem; i++) {
                Map<Integer, Double> Ri = new HashMap<>();
                for (int u = 0; u < numUser; u++) {
                    if (data[u][i] != 0.) {
                        Ri.put(u, data[u][i]);
                    }
                }

                // most similar data
                Map<Integer, Map<Integer, Double>> knnRatesOfI = new HashMap<>();
                for (int u = 0; u < numUser; u++) {
                    Map<Integer, Double> Ru = new HashMap<>();
                    for (int k = 0; k < numItem; k++) {
                        if (k == i)
                            continue;
                        Ru.put(k, data[u][k]);
                    }
                    knnRatesOfI.put(u, Ru);
                }

                for (int j = 0; j < numItem; j++) {
                    if (i == j)
                        continue;
                    Map<Integer, Double> Rj = new HashMap<>();
                    for (int u = 0; u < numUser; u++) {
                        if (data[u][j] != 0.) {
                            Rj.put(u, data[u][j]);
                        }
                    }

                    Object[] args = {i, Ri, knnRatesOfI, j, Rj};
                    slim.process(args);
                }
            }
        }
    }

    @Test(expected = HiveException.class)
    public void testInvalidL1() throws Exception {
        SlimUDTF slim = new SlimUDTF();
        ObjectInspector itemIOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector itemJOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;

        ObjectInspector itemIRatesOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);
        ObjectInspector itemJRatesOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            PrimitiveObjectInspectorFactory.javaDoubleObjectInspector);
        ObjectInspector knnOfIOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector));
        ObjectInspector argumentOI = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-l1 2.");

        ObjectInspector[] argOIs = {itemIOI, itemIRatesOI, knnOfIOI, itemJOI, itemJRatesOI,
                argumentOI};

        slim.initialize(argOIs);
    }
}
