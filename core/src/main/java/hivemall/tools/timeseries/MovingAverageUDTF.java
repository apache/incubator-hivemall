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
package hivemall.tools.timeseries;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.stats.MovingAverage;

import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Writable;

// @formatter:off
@Description(name = "moving_avg",
        value = "_FUNC_(NUMBER value, const int windowSize)"
                + " - Returns moving average of a time series using a given window",
        extended = "SELECT moving_avg(x, 3) FROM (SELECT explode(array(1.0,2.0,3.0,4.0,5.0,6.0,7.0)) as x) series;\n" +
                " 1.0\n" + 
                " 1.5\n" + 
                " 2.0\n" + 
                " 3.0\n" + 
                " 4.0\n" + 
                " 5.0\n" + 
                " 6.0")
// @formatter:on
@UDFType(deterministic = false, stateful = true)
public final class MovingAverageUDTF extends GenericUDTF {

    private PrimitiveObjectInspector valueOI;

    private MovingAverage movingAvg;

    private Writable[] forwardObjs;
    private DoubleWritable result;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentException(
                "Two argument is expected for moving_avg(NUMBER value, const int windowSize): "
                        + argOIs.length);
        }
        this.valueOI = HiveUtils.asNumberOI(argOIs[0]);

        int windowSize = HiveUtils.getConstInt(argOIs[1]);
        this.movingAvg = new MovingAverage(windowSize);

        this.result = new DoubleWritable();
        this.forwardObjs = new Writable[] {result};

        List<String> fieldNames = Arrays.asList("avg");
        List<ObjectInspector> fieldOIs = Arrays.<ObjectInspector>asList(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        double x = HiveUtils.getDouble(args[0], valueOI);

        double avg = movingAvg.add(x);
        result.set(avg);

        forward(forwardObjs);
    }

    @Override
    public void close() throws HiveException {}

}
