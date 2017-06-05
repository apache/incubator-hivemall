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

import hivemall.mf.BPRMatrixFactorizationUDTFTest;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.StringUtils;
import hivemall.utils.lang.mutable.MutableInt;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class DIMSUMMapperUDTFTest {
    private static final boolean DEBUG = false;

    DIMSUMMapperUDTF udtf;

    double[][] R;
    int numUsers, numItems;

    @Before
    public void setUp() throws HiveException {
        this.udtf = new DIMSUMMapperUDTF();

        this.R = new double[][] { {1, 2, 3}, {1, 2, 3}};
        this.numUsers = R.length;
        this.numItems = R[0].length;
    }

    @Test
    public void testIntFeature() throws HiveException {
        final MutableInt count = new MutableInt(0);
        final Map<Integer, Map<Integer, Double>> sims = new HashMap<Integer, Map<Integer, Double>>();
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                Object[] row = (Object[]) input;

                Assert.assertTrue(row.length == 3);

                int j = HiveUtils.asJavaInt(row[0]);
                int k = HiveUtils.asJavaInt(row[1]);

                Map<Integer, Double> sims_j = sims.get(j);
                if (sims_j == null) {
                    sims_j = new HashMap<Integer, Double>();
                    sims.put(j, sims_j);
                }
                Double sims_jk = sims_j.get(k);
                if (sims_jk == null) {
                    sims_jk = 0.d;
                    count.addValue(1);
                }
                sims_j.put(k, sims_jk + HiveUtils.asJavaDouble(row[2]));
            }
        };
        udtf.setCollector(collector);

        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorFactory.getStandardMapObjectInspector(
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-threshold 0 -disable_symmetric_output -int_feature")};
        // if threshold = 0, output is exact cosine similarity

        udtf.initialize(argOIs);

        final Integer[] itemIDs = new Integer[] {1, 2, 3};

        final List<String> user1 = new ArrayList<String>();
        convertRowToFeatures(0, user1, itemIDs);

        final List<String> user2 = new ArrayList<String>();
        convertRowToFeatures(1, user2, itemIDs);

        final Map<Integer, Double> norms = new HashMap<Integer, Double>();
        computeColumnNorms(norms, itemIDs);

        udtf.process(new Object[] {user1, norms});
        udtf.process(new Object[] {user2, norms});

        udtf.close();

        for (Integer j : sims.keySet()) {
            Map<Integer, Double> e = sims.get(j);
            for (Integer k : e.keySet()) {
                double s = e.get(k).doubleValue();
                println("(" + j + ", " + k + ") = " + s);
                Assert.assertEquals(1.d, s, 1e-6);
            }
        }

        // <1, 2>, <1, 3>, <2, 3>
        Assert.assertTrue(count.getValue() == 3);
    }

    @Test
    public void testStringFeature() throws HiveException {
        final MutableInt count = new MutableInt(0);
        final Map<String, Map<String, Double>> sims = new HashMap<String, Map<String, Double>>();
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                Object[] row = (Object[]) input;

                Assert.assertTrue(row.length == 3);

                String j = row[0].toString();
                String k = row[1].toString();

                Map<String, Double> sims_j = sims.get(j);
                if (sims_j == null) {
                    sims_j = new HashMap<String, Double>();
                    sims.put(j, sims_j);
                }
                Double sims_jk = sims_j.get(k);
                if (sims_jk == null) {
                    sims_jk = 0.d;
                    count.addValue(1);
                }
                sims_j.put(k, sims_jk + HiveUtils.asJavaDouble(row[2]));
            }
        };
        udtf.setCollector(collector);

        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorFactory.getStandardMapObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-threshold 0")};

        udtf.initialize(argOIs);

        final String[] itemIDs = new String[] {"i1", "i2", "i3"};

        final List<String> user1 = new ArrayList<String>();
        convertRowToFeatures(0, user1, itemIDs);

        final List<String> user2 = new ArrayList<String>();
        convertRowToFeatures(1, user2, itemIDs);

        final Map<String, Double> norms = new HashMap<String, Double>();
        computeColumnNorms(norms, itemIDs);

        udtf.process(new Object[] {user1, norms});
        udtf.process(new Object[] {user2, norms});

        udtf.close();

        for (String j : sims.keySet()) {
            Map<String, Double> e = sims.get(j);
            for (String k : e.keySet()) {
                double s = e.get(k).doubleValue();
                println("(" + j + ", " + k + ") = " + s);
                Assert.assertEquals(1.d, s, 1e-6);
            }
        }

        // <i1, i2>, <i2, i1>, <i1, i3>, <i3, i2>, <i2, i3>, <i3, i2>
        Assert.assertTrue(count.getValue() == 6);
    }

    private void convertRowToFeatures(int i, @Nonnull List<String> dst, @Nonnull Object[] itemIDs) {
        for (int j = 0; j < numItems; j++) {
            double r = R[i][j];
            if (r != 0.d) {
                dst.add(itemIDs[j] + ":" + r);
            }
        }
    }

    private <T> void computeColumnNorms(@Nonnull Map<T, Double> dst, @Nonnull T[] itemIDs) {
        for (int j = 0; j < numItems; j++) {
            double norm = 0.d;
            for (int i = 0; i < numUsers; i++) {
                norm += R[i][j] * R[i][j];
            }
            dst.put(itemIDs[j], Math.sqrt(norm));
        }
    }

    @Test
    public void testML100k() throws HiveException, IOException {
        final Map<String, List<String>> users = new HashMap<String, List<String>>();
        final Map<String, Double> norms = new HashMap<String, Double>();

        BufferedReader buf = readFile("ml1k.train.gz");
        String line;
        while ((line = buf.readLine()) != null) {
            String[] cols = StringUtils.split(line, ' ');

            // find this user's list of ratings
            String userID = cols[0];
            List<String> ratings = users.get(userID);
            if (ratings == null) {
                ratings = new ArrayList<String>();
                users.put(userID, ratings);
            }

            // store observed item-rate pairs to the list
            String itemID = cols[1];
            double rate = Double.valueOf(cols[2]).doubleValue();
            ratings.add(itemID + ":" + rate);

            // accumulate to compute L2 norm of each column
            Double norm = norms.get(itemID);
            if (norm == null) {
                norm = 0.d;
            }
            norm += rate * rate;
            norms.put(itemID, norm);
        }

        // compute L2 norm of each column
        for (Map.Entry<String, Double> e : norms.entrySet()) {
            norms.put(e.getKey(), Math.sqrt(e.getValue().doubleValue()));
        }

        final MutableInt count = new MutableInt(0);
        final Map<String, Map<String, Double>> sims = new HashMap<String, Map<String, Double>>();

        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                Object[] row = (Object[]) input;

                Assert.assertTrue(row.length == 3);

                String j = row[0].toString();
                String k = row[1].toString();

                Map<String, Double> sims_j = sims.get(j);
                if (sims_j == null) {
                    sims_j = new HashMap<String, Double>();
                    sims.put(j, sims_j);
                }
                Double sims_jk = sims_j.get(k);
                if (sims_jk == null) {
                    sims_jk = 0.d;
                    count.addValue(1);
                }
                sims_j.put(k, sims_jk + HiveUtils.asJavaDouble(row[2]));
            }
        };
        udtf.setCollector(collector);

        // Case I: Set zero to `threshold`
        // this computes exact cosine similarity
        ObjectInspector[] argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorFactory.getStandardMapObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-threshold 0 -disable_symmetric_output")};

        udtf.initialize(argOIs);
        for (List<String> user : users.values()) {
            udtf.process(new Object[] {user, norms});
        }
        udtf.close();

        int maxCount = count.getValue();

        // reset counter and similarities
        count.setValue(0);
        sims.clear();

        // Case II: Set (almost) max value to `threshold`
        // this skips a bunch of operations with high probability
        argOIs = new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                ObjectInspectorFactory.getStandardMapObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-threshold 0.999999 -disable_symmetric_output")};

        udtf.initialize(argOIs);
        for (List<String> user : users.values()) {
            udtf.process(new Object[] {user, norms});
        }
        udtf.close();

        Assert.assertTrue("Approximated one MUST reduce the number of operations",
            count.getValue() < maxCount);
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        // use MF's resource file
        InputStream is = BPRMatrixFactorizationUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

    private static void println(String msg) {
        if (DEBUG) {
            System.out.println(msg);
        }
    }
}
