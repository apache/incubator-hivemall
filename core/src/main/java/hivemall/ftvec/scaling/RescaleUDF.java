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

import javax.annotation.CheckForNull;
import javax.annotation.Nullable;

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

    @Nullable
    public FloatWritable evaluate(@Nullable final Double value, @CheckForNull final Double min,
            @CheckForNull final Double max) throws HiveException {
        return evaluate(double2Float(value), double2Float(min), double2Float(max));
    }

    @Nullable
    public FloatWritable evaluate(@Nullable final Float value, @CheckForNull final Float min,
            @CheckForNull final Float max) throws HiveException {
        if (value == null) {
            return null;
        }

        if (min == null)
            throw new HiveException("min should not be null");
        if (max == null)
            throw new HiveException("max should not be null");

        return val(min_max_normalization(value, min, max));
    }

    @Nullable
    public Text evaluate(@Nullable final String s, @CheckForNull final Double min,
            @CheckForNull final Double max) throws HiveException {
        return evaluate(s, double2Float(min), double2Float(max));
    }

    @Nullable
    public Text evaluate(@Nullable final String s, @CheckForNull final Float min,
            @CheckForNull final Float max) throws HiveException {
        if (s == null) {
            return null;
        }

        if (min == null)
            throw new HiveException("min should not be null");
        if (max == null)
            throw new HiveException("max should not be null");

        final String[] fv = s.split(":");
        if (fv.length != 2) {
            throw new HiveException(
                String.format("Invalid feature value " + "representation: %s", s));
        }
        float v;
        try {
            v = Float.parseFloat(fv[1]);
        } catch (NumberFormatException e) {
            throw new HiveException(String.format(
                "Invalid feature value " + "representation: %s, %s can't parse to float.", s,
                fv[1]));
        }

        float scaled_v = min_max_normalization(v, min.floatValue(), max.floatValue());
        String ret = fv[0] + ':' + scaled_v;
        return val(ret);
    }

    private static float min_max_normalization(final float value, final float min, final float max)
            throws HiveException {
        if (min > max) {
            throw new HiveException(
                "min value `" + min + "` SHOULD be less than max value `" + max + '`');
        }
        if (min == max) {
            return 0.5f;
        }
        if (value < min) {
            return 0.f;
        }
        if (value > max) {
            return 1.f;
        }
        return (value - min) / (max - min);
    }

    @Nullable
    private static Float double2Float(@Nullable final Double value) {
        if (value == null) {
            return null;
        } else {
            return Float.valueOf(value.floatValue());
        }
    }

}
