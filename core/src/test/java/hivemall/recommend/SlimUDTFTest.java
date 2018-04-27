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

import java.util.HashMap;
import java.util.Map;

import hivemall.TestUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Test;

public class SlimUDTFTest {
    @Test
    public void testAllSamples() throws HiveException {
        SlimUDTF slim = new SlimUDTF();
        ObjectInspector itemIOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;
        ObjectInspector itemJOI = PrimitiveObjectInspectorFactory.javaIntObjectInspector;

        ObjectInspector itemIRatesOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            PrimitiveObjectInspectorFactory.javaFloatObjectInspector);
        ObjectInspector itemJRatesOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            PrimitiveObjectInspectorFactory.javaFloatObjectInspector);
        ObjectInspector topKRatesOfIOI = ObjectInspectorFactory.getStandardMapObjectInspector(
            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
            ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                PrimitiveObjectInspectorFactory.javaFloatObjectInspector));
        ObjectInspector optionArgumentOI = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-l2 0.01 -l1 0.01");

        ObjectInspector[] argOIs =
                {itemIOI, itemIRatesOI, topKRatesOfIOI, itemJOI, itemJRatesOI, optionArgumentOI};

        slim.initialize(argOIs);
        int numUser = 4;
        int numItem = 5;

        float[][] data = {{1.f, 4.f, 0.f, 0.f, 0.f}, {0.f, 3.f, 0.f, 1.f, 2.f},
                {2.f, 2.f, 0.f, 0.f, 3.f}, {0.f, 1.f, 1.f, 0.f, 0.f}};

        for (int i = 0; i < numItem; i++) {
            Map<Integer, Float> Ri = new HashMap<>();
            for (int u = 0; u < numUser; u++) {
                if (data[u][i] != 0.) {
                    Ri.put(u, data[u][i]);
                }
            }

            // most similar data
            Map<Integer, Map<Integer, Float>> knnRatesOfI = new HashMap<>();
            for (int u = 0; u < numUser; u++) {
                Map<Integer, Float> Ru = new HashMap<>();
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
                Map<Integer, Float> Rj = new HashMap<>();
                for (int u = 0; u < numUser; u++) {
                    if (data[u][j] != 0.) {
                        Rj.put(u, data[u][j]);
                    }
                }

                Object[] args = {i, Ri, knnRatesOfI, j, Rj};
                slim.process(args);
            }
        }
        slim.finalizeTraining();
    }

    @Test
    public void testSerialization() throws HiveException {
        int numUser = 4;
        int numItem = 5;

        float[][] data = {{1.f, 4.f, 0.f, 0.f, 0.f}, {0.f, 3.f, 0.f, 1.f, 2.f},
                {2.f, 2.f, 0.f, 0.f, 3.f}, {0.f, 1.f, 1.f, 0.f, 0.f}};

        Object[][] rows = new Object[numItem * (numItem - 1)][5];
        int ri = 0;

        for (int i = 0; i < numItem; i++) {
            Map<Integer, Float> Ri = new HashMap<>();
            for (int u = 0; u < numUser; u++) {
                if (data[u][i] != 0.) {
                    Ri.put(u, data[u][i]);
                }
            }

            // most similar data
            Map<Integer, Map<Integer, Float>> knnRatesOfI = new HashMap<>();
            for (int u = 0; u < numUser; u++) {
                Map<Integer, Float> Ru = new HashMap<>();
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
                Map<Integer, Float> Rj = new HashMap<>();
                for (int u = 0; u < numUser; u++) {
                    if (data[u][j] != 0.) {
                        Rj.put(u, data[u][j]);
                    }
                }

                rows[ri][0] = i;
                rows[ri][1] = Ri;
                rows[ri][2] = knnRatesOfI;
                rows[ri][3] = j;
                rows[ri][4] = Rj;
                ri += 1;
            }
        }

        TestUtils.testGenericUDTFSerialization(SlimUDTF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    ObjectInspectorFactory.getStandardMapObjectInspector(
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        PrimitiveObjectInspectorFactory.javaFloatObjectInspector),
                    ObjectInspectorFactory.getStandardMapObjectInspector(
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        ObjectInspectorFactory.getStandardMapObjectInspector(
                            PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                            PrimitiveObjectInspectorFactory.javaFloatObjectInspector)),
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    ObjectInspectorFactory.getStandardMapObjectInspector(
                        PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                        PrimitiveObjectInspectorFactory.javaFloatObjectInspector),
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-l2 0.01 -l1 0.01")},
            rows);
    }

}
