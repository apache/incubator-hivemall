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
package hivemall.ftvec.scaling;

import static hivemall.utils.hadoop.WritableUtils.val;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.Text;

/**
 * Min-Max normalization
 * 
 * @see <a href="http://en.wikipedia.org/wiki/Feature_scaling">Feature_scaling</a>
 */
@Description(name = "rescale",
        value = "_FUNC_(value, min, max) - Returns rescaled value by min-max normalization")
@UDFType(deterministic = true, stateful = false)
public final class RescaleUDF extends UDF {

    public FloatWritable evaluate(final Float value, final Float min, final Float max)
            throws HiveException {
        if(value == null) {
            return null;
        }

        if(min == null) throw new HiveException("min should not be null");
        if(max == null) throw new HiveException("max should not be null");

        return val(min_max_normalization(value, min, max));
    }

    public FloatWritable evaluate(final Double value, final Double min, final Double max)
            throws HiveException {
        return evaluate(value.floatValue(), double2Float(min), double2Float(max));
    }

    public Text evaluate(final String s, final Double min, final Double max) throws HiveException {
        return evaluate(s, double2Float(min), double2Float(max));
    }

    public Text evaluate(final String s, final Float min, final Float max)
            throws HiveException, IllegalArgumentException {
        String[] fv = s.split(":");
        if (fv.length != 2) {
            throw new IllegalArgumentException(String.format("Invalid feature value " +
                    "representation: %s", s));
        }
        float v;
        try {
            v = Float.parseFloat(fv[1]);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(String.format("Invalid feature value " +
                    "representation: %s, %s can't parse to float.", s, fv[1]));
        }

        FloatWritable scaled_v = evaluate(v, min, max);
        String ret = fv[0] + ':' + scaled_v.get();
        return val(ret);
    }

    private static float min_max_normalization(final float value,
                                               final float min, final float max) {
        if (min == max) {
            return 0.5f;
        }
        return (value - min) / (max - min);
    }

    private Float double2Float(final Double value) {
        if (null == value) {
            return null;
        } else {
            return value.floatValue();
        }
    }

}
