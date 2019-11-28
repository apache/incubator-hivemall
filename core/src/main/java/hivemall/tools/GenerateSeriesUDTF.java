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

import hivemall.utils.hadoop.HiveUtils;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Writable;

// @formatter:off
@Description(name = "generate_series",
        value = "_FUNC_(const int|bigint start, const int|bigint end) - "
                + "Generate a series of values, from start to end. " + 
                "A similar function to PostgreSQL's [generate_serics](https://www.postgresql.org/docs/current/static/functions-srf.html)",
        extended = "SELECT generate_series(2,4);\n" + 
                "\n" + 
                " 2\n" + 
                " 3\n" + 
                " 4\n" + 
                "\n" + 
                "SELECT generate_series(5,1,-2);\n" + 
                "\n" + 
                " 5\n" + 
                " 3\n" + 
                " 1\n" + 
                "\n" + 
                "SELECT generate_series(4,3);\n" + 
                "\n" + 
                " (no return)\n" + 
                "\n" + 
                "SELECT date_add(current_date(),value),value from (SELECT generate_series(1,3)) t;\n" + 
                "\n" + 
                " 2018-04-21      1\n" + 
                " 2018-04-22      2\n" + 
                " 2018-04-23      3\n" + 
                "\n" + 
                "WITH input as (\n" + 
                " SELECT 1 as c1, 10 as c2, 3 as step\n" + 
                " UNION ALL\n" + 
                " SELECT 10, 2, -3\n" + 
                ")\n" + 
                "SELECT generate_series(c1, c2, step) as series\n" + 
                "FROM input;\n" + 
                "\n" + 
                " 1\n" + 
                " 4\n" + 
                " 7\n" + 
                " 10\n" + 
                " 10\n" + 
                " 7\n" + 
                " 4")
// @formatter:on
public final class GenerateSeriesUDTF extends GenericUDTF {

    private PrimitiveObjectInspector startOI, endOI;
    @Nullable
    private PrimitiveObjectInspector stepOI;

    @Nonnull
    private final Writable[] row = new Writable[1];
    private boolean returnLong;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentException(
                "Expected number of arguments is 2 or 3: " + argOIs.length);
        }
        this.startOI = HiveUtils.asIntegerOI(argOIs, 0);
        this.endOI = HiveUtils.asIntegerOI(argOIs, 1);

        if (argOIs.length == 3) {
            this.stepOI = HiveUtils.asIntegerOI(argOIs, 2);
        }

        this.returnLong = HiveUtils.isBigIntOI(startOI) || HiveUtils.isBigIntOI(endOI);

        List<String> fieldNames = new ArrayList<>(1);
        fieldNames.add("value");
        List<ObjectInspector> fieldOIs = new ArrayList<>(1);
        if (returnLong) {
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
        } else {
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        }
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (returnLong) {
            generateLongSeries(args);
        } else {
            generateIntSeries(args);
        }
    }

    private void generateLongSeries(@Nonnull final Object[] args) throws HiveException {
        final long start, end;
        long step = 1L;
        switch (args.length) {
            case 3:
                step = PrimitiveObjectInspectorUtils.getLong(args[2], stepOI);
                if (step == 0) {
                    throw new UDFArgumentException("Step MUST NOT be zero");
                }
                // fall through
            case 2:
                start = PrimitiveObjectInspectorUtils.getLong(args[0], startOI);
                end = PrimitiveObjectInspectorUtils.getLong(args[1], endOI);
                break;
            default:
                throw new UDFArgumentException("Expected number of arguments: " + args.length);
        }

        final LongWritable row0 = new LongWritable();
        row[0] = row0;
        if (step > 0) {
            for (long i = start; i <= end; i += step) {
                row0.set(i);
                forward(row);
            }
        } else {
            for (long i = start; i >= end; i += step) {
                row0.set(i);
                forward(row);
            }
        }
    }

    private void generateIntSeries(@Nonnull final Object[] args) throws HiveException {
        final int start, end;
        int step = 1;
        switch (args.length) {
            case 3:
                step = PrimitiveObjectInspectorUtils.getInt(args[2], stepOI);
                if (step == 0) {
                    throw new UDFArgumentException("Step MUST NOT be zero");
                }
                // fall through
            case 2:
                start = PrimitiveObjectInspectorUtils.getInt(args[0], startOI);
                end = PrimitiveObjectInspectorUtils.getInt(args[1], endOI);
                break;
            default:
                throw new UDFArgumentException("Expected number of arguments: " + args.length);
        }

        final IntWritable row0 = new IntWritable();
        row[0] = row0;
        if (step > 0) {
            for (int i = start; i <= end; i += step) {
                row0.set(i);
                forward(row);
            }
        } else {
            for (int i = start; i >= end; i += step) {
                row0.set(i);
                forward(row);
            }
        }
    }

    @Override
    public void close() throws HiveException {}

}
