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
package hivemall.tools.array;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.StringUtils;

import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnegative;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.IntWritable;

// @formatter:off
@Description(name = "arange",
value = "_FUNC_([int start=0, ] int stop, [int step=1]) - Return evenly spaced values within a given interval",
extended = "SELECT arange(5), arange(1, 5), arange(1, 5, 1), arange(0, 5, 1);\n" + 
        "> [0,1,2,3,4]     [1,2,3,4]       [1,2,3,4]       [0,1,2,3,4]\n" + 
        "\n" + 
        "SELECT arange(1, 6, 2);\n" + 
        "> 1, 3, 5\n" + 
        "\n" + 
        "SELECT arange(-1, -6, 2);\n" + 
        "-1, -3, -5")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class ArangeUDF extends GenericUDF {

    @Nullable
    private PrimitiveObjectInspector startOI;
    private PrimitiveObjectInspector stopOI;
    @Nullable
    private PrimitiveObjectInspector stepOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        switch (argOIs.length) {
            case 1:
                if (!HiveUtils.isIntegerOI(argOIs[0])) {
                    throw new UDFArgumentException(
                        "arange(int stop) expects integer for the 1st argument: "
                                + argOIs[0].getTypeName());
                }
                this.stopOI = HiveUtils.asIntegerOI(argOIs[0]);
                break;
            case 3:
                if (!HiveUtils.isIntegerOI(argOIs[2])) {
                    throw new UDFArgumentException(
                        "arange(int start, int stop, int step) expects integer for the 3rd argument: "
                                + argOIs[2].getTypeName());
                }
                this.stepOI = HiveUtils.asIntegerOI(argOIs[2]);
                // fall through
            case 2:
                if (!HiveUtils.isIntegerOI(argOIs[0])) {
                    throw new UDFArgumentException(
                        "arange(int start, int stop) expects integer for the 1st argument: "
                                + argOIs[0].getTypeName());
                }
                this.startOI = HiveUtils.asIntegerOI(argOIs[0]);
                if (!HiveUtils.isIntegerOI(argOIs[1])) {
                    throw new UDFArgumentException(
                        "arange(int start, int stop) expects integer for the 2nd argument: "
                                + argOIs[1].getTypeName());
                }
                this.stopOI = HiveUtils.asIntegerOI(argOIs[1]);
                break;
            default:
                throw new UDFArgumentException(
                    "arange([int start=0, ] int stop, [int step=1]) takes 1~3 arguments: "
                            + argOIs.length);
        }

        return ObjectInspectorFactory.getStandardListObjectInspector(
            ObjectInspectorUtils.getStandardObjectInspector(
                PrimitiveObjectInspectorFactory.writableIntObjectInspector));
    }

    @Nullable
    @Override
    public List<IntWritable> evaluate(DeferredObject[] arguments) throws HiveException {
        int start = 0, step = 1;
        final int stop;
        switch (arguments.length) {
            case 1: {
                Object arg0 = arguments[0].get();
                if (arg0 == null) {
                    return null;
                }
                stop = PrimitiveObjectInspectorUtils.getInt(arg0, stopOI);
                break;
            }
            case 3: {
                Object arg2 = arguments[2].get();
                if (arg2 == null) {
                    return null;
                }
                step = PrimitiveObjectInspectorUtils.getInt(arg2, stepOI);
                // fall through
            }
            case 2: {
                Object arg0 = arguments[0].get();
                if (arg0 == null) {
                    return null;
                }
                start = PrimitiveObjectInspectorUtils.getInt(arg0, startOI);
                Object arg1 = arguments[1].get();
                if (arg1 == null) {
                    return null;
                }
                stop = PrimitiveObjectInspectorUtils.getInt(arg1, stopOI);
                break;
            }
            default:
                throw new UDFArgumentException(
                    "arange([int start=0, ] int stop, [int step=1]) takes 1~3 arguments: "
                            + arguments.length);
        }

        return Arrays.asList(range(start, stop, step));
    }

    /**
     * Return evenly spaced values within a given interval.
     *
     * @param start inclusive index of the start
     * @param stop exclusive index of the end
     * @param step positive interval value
     */
    private static IntWritable[] range(final int start, final int stop, @Nonnegative final int step)
            throws UDFArgumentException {
        if (step <= 0) {
            throw new UDFArgumentException("Invalid step value: " + step);
        }

        final IntWritable[] r;
        final int diff = stop - start;
        if (diff < 0) {
            final int count = ArrayUtils.divideAndRoundUp(-diff, step);
            r = new IntWritable[count];
            for (int i = 0, value = start; i < r.length; i++, value -= step) {
                r[i] = new IntWritable(value);
            }
        } else {
            final int count = ArrayUtils.divideAndRoundUp(diff, step);
            r = new IntWritable[count];
            for (int i = 0, value = start; i < r.length; i++, value += step) {
                r[i] = new IntWritable(value);
            }
        }
        return r;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "arange(" + StringUtils.join(children, ',') + ')';
    }

}
