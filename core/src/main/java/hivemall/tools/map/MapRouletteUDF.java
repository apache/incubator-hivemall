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
package hivemall.tools.map;

import static hivemall.utils.lang.StringUtils.join;

import hivemall.utils.hadoop.HiveUtils;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Random;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

import com.clearspring.analytics.util.Preconditions;

/**
 * The map_roulette returns a map key based on weighted random sampling of map values.
 */
@Description(name = "map_roulette",
        value = "_FUNC_(Map<K, number> map [, (const) int/bigint seed])"
                + " - Returns a map key based on weighted random sampling of map values."
                + " Average of values is used for null values")
@UDFType(deterministic = false, stateful = false) // it is false because it return value base on probability
public final class MapRouletteUDF extends GenericUDF {

    private transient MapObjectInspector mapOI;
    private transient PrimitiveObjectInspector valueOI;
    @Nullable
    private transient PrimitiveObjectInspector seedOI;

    @Nullable
    private transient Random _rand;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1 && argOIs.length != 2) {
            throw new UDFArgumentLengthException(
                "Expected exactly one argument for map_roulette: " + argOIs.length);
        }
        if (argOIs[0].getCategory() != ObjectInspector.Category.MAP) {
            throw new UDFArgumentTypeException(0,
                "Only map type argument is accepted but got " + argOIs[0].getTypeName());
        }

        this.mapOI = HiveUtils.asMapOI(argOIs[0]);
        this.valueOI = HiveUtils.asNumberOI(mapOI.getMapValueObjectInspector());

        if (argOIs.length == 2) {
            ObjectInspector argOI1 = argOIs[1];
            if (HiveUtils.isIntegerOI(argOI1) == false) {
                throw new UDFArgumentException(
                    "The second argument of map_roulette must be integer type: "
                            + argOI1.getTypeName());
            }
            if (ObjectInspectorUtils.isConstantObjectInspector(argOI1)) {
                long seed = HiveUtils.getAsConstLong(argOI1);
                this._rand = new Random(seed); // fixed seed
            } else {
                this.seedOI = HiveUtils.asLongCompatibleOI(argOI1);
            }
        } else {
            this._rand = new Random(); // random seed
        }

        return mapOI.getMapKeyObjectInspector();
    }

    @Nullable
    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        Random rand = _rand;
        if (rand == null) {
            Object arg1 = arguments[1].get();
            if (arg1 == null) {
                rand = new Random();
            } else {
                long seed = HiveUtils.getLong(arg1, seedOI);
                rand = new Random(seed);
            }
        }

        Map<Object, Double> input = getObjectDoubleMap(arguments[0], mapOI, valueOI);
        if (input == null) {
            return null;
        }

        return rouletteWheelSelection(input, rand);
    }

    @Nullable
    private static Map<Object, Double> getObjectDoubleMap(@Nonnull final DeferredObject argument,
            @Nonnull final MapObjectInspector mapOI,
            @Nonnull final PrimitiveObjectInspector valueOI) throws HiveException {
        final Map<?, ?> m = mapOI.getMap(argument.get());
        if (m == null) {
            return null;
        }
        final int size = m.size();
        if (size == 0) {
            return null;
        }

        final Map<Object, Double> result = new HashMap<>(size);
        double sum = 0.d;
        int cnt = 0;
        for (Map.Entry<?, ?> entry : m.entrySet()) {
            Object key = entry.getKey();
            if (key == null) {
                continue;
            }
            Object value = entry.getValue();
            if (value == null) {
                continue;
            }
            final double v = PrimitiveObjectInspectorUtils.convertPrimitiveToDouble(value, valueOI);
            if (v < 0) {
                throw new UDFArgumentException(
                    "Map value must be greather than or equals to zero: " + entry.getValue());
            }

            result.put(key, Double.valueOf(v));
            sum += v;
            cnt++;
        }

        if (result.isEmpty()) {
            return null;
        }

        if (result.size() < m.size()) {
            // fillna with the avg value
            final Double avg = Double.valueOf(sum / cnt);
            for (Map.Entry<?, ?> entry : m.entrySet()) {
                Object key = entry.getKey();
                if (key == null) {
                    continue;
                }
                if (entry.getValue() == null) {
                    result.put(key, avg);
                }
            }
        }

        return result;
    }

    /**
     * Roulette Wheel Selection.
     * 
     * See https://www.obitko.com/tutorials/genetic-algorithms/selection.php
     */
    @Nullable
    private static Object rouletteWheelSelection(@Nonnull final Map<Object, Double> m,
            @Nonnull final Random rnd) {
        Preconditions.checkArgument(m.isEmpty() == false);

        // 1. calculate sum
        double sum = 0.d;
        for (Double v : m.values()) {
            sum += v.doubleValue();
        }

        // 2. Generate random number from interval r=[0,sum)
        double r = rnd.nextDouble() * sum;

        // 3. Go through the population and sum weight from 0 - sum s.
        //    When the sum s is greater then r, stop and return the element.
        Object k = null;
        double s = 0.d;
        for (Map.Entry<Object, Double> e : m.entrySet()) {
            k = e.getKey();
            double v = e.getValue().doubleValue();
            s += v;
            if (s > r) {
                break;
            }
        }

        return Objects.requireNonNull(k);
    }

    @Override
    public String getDisplayString(String[] children) {
        return "map_roulette(" + join(children, ',') + ")";
    }

}
