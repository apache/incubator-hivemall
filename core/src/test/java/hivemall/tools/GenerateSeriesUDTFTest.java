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
package hivemall.tools;

import hivemall.TestUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.junit.Assert;
import org.junit.Test;

public class GenerateSeriesUDTFTest {

    @Test
    public void testTwoConstArgs() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(new ObjectInspector[] {
                PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
                    TypeInfoFactory.intTypeInfo, new IntWritable(1)),
                PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
                    TypeInfoFactory.intTypeInfo, new IntWritable(3))});

        final List<IntWritable> actual = new ArrayList<>();

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {
                Object[] row = (Object[]) args;
                IntWritable row0 = (IntWritable) row[0];
                actual.add(new IntWritable(row0.get()));
            }
        });

        udtf.process(new Object[] {new IntWritable(1), new IntWritable(3)});

        List<IntWritable> expected =
                Arrays.asList(new IntWritable(1), new IntWritable(2), new IntWritable(3));
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void testTwoIntArgs() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.writableIntObjectInspector});

        final List<IntWritable> actual = new ArrayList<>();

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {
                Object[] row = (Object[]) args;
                IntWritable row0 = (IntWritable) row[0];
                actual.add(new IntWritable(row0.get()));
            }
        });

        udtf.process(new Object[] {1, new IntWritable(3)});

        List<IntWritable> expected =
                Arrays.asList(new IntWritable(1), new IntWritable(2), new IntWritable(3));
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void testTwoLongArgs() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.writableLongObjectInspector});

        final List<LongWritable> actual = new ArrayList<>();

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {
                Object[] row = (Object[]) args;
                LongWritable row0 = (LongWritable) row[0];
                actual.add(new LongWritable(row0.get()));
            }
        });

        udtf.process(new Object[] {1, new LongWritable(3)});

        List<LongWritable> expected =
                Arrays.asList(new LongWritable(1), new LongWritable(2), new LongWritable(3));
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void testThreeIntArgs() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaLongObjectInspector});

        final List<IntWritable> actual = new ArrayList<>();

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {
                Object[] row = (Object[]) args;
                IntWritable row0 = (IntWritable) row[0];
                actual.add(new IntWritable(row0.get()));
            }
        });

        udtf.process(new Object[] {1, new IntWritable(7), 3L});

        List<IntWritable> expected =
                Arrays.asList(new IntWritable(1), new IntWritable(4), new IntWritable(7));
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void testThreeLongArgs() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaLongObjectInspector,
                    PrimitiveObjectInspectorFactory.writableLongObjectInspector,
                    PrimitiveObjectInspectorFactory.javaLongObjectInspector});

        final List<LongWritable> actual = new ArrayList<>();

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {
                Object[] row = (Object[]) args;
                LongWritable row0 = (LongWritable) row[0];
                actual.add(new LongWritable(row0.get()));
            }
        });

        udtf.process(new Object[] {1L, new LongWritable(7), 3L});

        List<LongWritable> expected =
                Arrays.asList(new LongWritable(1), new LongWritable(4), new LongWritable(7));
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void testNegativeStepInt() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaLongObjectInspector});

        final List<IntWritable> actual = new ArrayList<>();

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {
                Object[] row = (Object[]) args;
                IntWritable row0 = (IntWritable) row[0];
                actual.add(new IntWritable(row0.get()));
            }
        });

        udtf.process(new Object[] {5, new IntWritable(1), -2L});

        List<IntWritable> expected =
                Arrays.asList(new IntWritable(5), new IntWritable(3), new IntWritable(1));
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void testNegativeStepLong() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaLongObjectInspector,
                    PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                    PrimitiveObjectInspectorFactory.javaIntObjectInspector});

        final List<LongWritable> actual = new ArrayList<>();

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {
                Object[] row = (Object[]) args;
                LongWritable row0 = (LongWritable) row[0];
                actual.add(new LongWritable(row0.get()));
            }
        });

        udtf.process(new Object[] {5L, new IntWritable(1), -2});

        List<LongWritable> expected =
                Arrays.asList(new LongWritable(5), new LongWritable(3), new LongWritable(1));
        Assert.assertEquals(expected, actual);
    }

    @Test
    public void testSerialization() throws HiveException {
        GenerateSeriesUDTF udtf = new GenerateSeriesUDTF();

        udtf.initialize(
            new ObjectInspector[] {PrimitiveObjectInspectorFactory.javaIntObjectInspector,
                    PrimitiveObjectInspectorFactory.writableIntObjectInspector});

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object args) throws HiveException {}
        });

        udtf.process(new Object[] {1, new IntWritable(3)});

        byte[] serialized = TestUtils.serializeObjectByKryo(udtf);
        TestUtils.deserializeObjectByKryo(serialized, GenerateSeriesUDTF.class);
    }

}
