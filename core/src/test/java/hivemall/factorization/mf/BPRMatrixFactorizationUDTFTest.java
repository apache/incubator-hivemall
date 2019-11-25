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
package hivemall.factorization.mf;

import hivemall.TestUtils;
import hivemall.factorization.mf.BPRMatrixFactorizationUDTF;
import hivemall.utils.lang.StringUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.exec.MapredContextAccessor;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.junit.Assert;
import org.junit.Test;

public class BPRMatrixFactorizationUDTFTest {

    @Test
    public void testMovielens1k() throws HiveException, IOException {
        final int iterations = 50;
        BPRMatrixFactorizationUDTF bpr = new BPRMatrixFactorizationUDTF();

        ObjectInspector intOI = PrimitiveObjectInspectorFactory.writableIntObjectInspector;
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            new String("-factor 10 -iter " + iterations));
        ObjectInspector[] argOIs = new ObjectInspector[] {intOI, intOI, intOI, param};

        MapredContext mapredContext = MapredContextAccessor.create(true, null);
        bpr.configure(mapredContext);
        bpr.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {}
        });
        bpr.initialize(argOIs);

        final IntWritable user = new IntWritable();
        final IntWritable posItem = new IntWritable();
        final IntWritable negItem = new IntWritable();
        final Object[] args = new Object[] {user, posItem, negItem};

        BufferedReader train = readFile("ml1k.train.gz");
        String line;
        while ((line = train.readLine()) != null) {
            parseLine(line, user, posItem, negItem);
            bpr.process(args);
        }
        bpr.close();
        int finishedIter = bpr.cvState.getCurrentIteration();
        Assert.assertTrue("finishedIter: " + finishedIter, finishedIter < iterations);
    }

    @Test
    public void testMovielens1kBoldDriver() throws HiveException, IOException {
        final int iterations = 50;
        BPRMatrixFactorizationUDTF bpr = new BPRMatrixFactorizationUDTF();

        ObjectInspector intOI = PrimitiveObjectInspectorFactory.writableIntObjectInspector;
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector,
            new String("-boldDriver -factor 10 -iter " + iterations));
        ObjectInspector[] argOIs = new ObjectInspector[] {intOI, intOI, intOI, param};

        MapredContext mapredContext = MapredContextAccessor.create(true, null);
        bpr.configure(mapredContext);
        bpr.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {}
        });
        bpr.initialize(argOIs);

        final IntWritable user = new IntWritable();
        final IntWritable posItem = new IntWritable();
        final IntWritable negItem = new IntWritable();
        final Object[] args = new Object[] {user, posItem, negItem};

        BufferedReader train = readFile("ml1k.train.gz");
        String line;
        while ((line = train.readLine()) != null) {
            parseLine(line, user, posItem, negItem);
            bpr.process(args);
        }
        bpr.close();
        int finishedIter = bpr.cvState.getCurrentIteration();
        Assert.assertTrue("finishedIter: " + finishedIter, finishedIter < iterations);
    }

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(BPRMatrixFactorizationUDTF.class,
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-factor 10 -iter 1")},
            new Object[][] {{0, 0, 1}});
    }

    @Nonnull
    private static BufferedReader readFile(@Nonnull String fileName) throws IOException {
        InputStream is = BPRMatrixFactorizationUDTFTest.class.getResourceAsStream(fileName);
        if (fileName.endsWith(".gz")) {
            is = new GZIPInputStream(is);
        }
        return new BufferedReader(new InputStreamReader(is));
    }

    private static void parseLine(@Nonnull String line, @Nonnull IntWritable user,
            @Nonnull IntWritable posItem, @Nonnull IntWritable negItem) {
        String[] cols = StringUtils.split(line, ' ');
        Assert.assertEquals(3, cols.length);
        user.set(Integer.parseInt(cols[0]));
        posItem.set(Integer.parseInt(cols[1]));
        negItem.set(Integer.parseInt(cols[2]));
    }


}
