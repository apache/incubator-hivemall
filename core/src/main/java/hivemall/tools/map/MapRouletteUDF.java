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
import hivemall.utils.lang.Preconditions;

import java.util.HashMap;
import java.util.Map;
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

/**
 * The map_roulette returns a map key based on weighted random sampling of map values.
 */
// @formatter:off
@Description(name = "map_roulette",
        value = "_FUNC_(Map<K, number> map [, (const) int/bigint seed])"
                + " - Returns a map key based on weighted random sampling of map values."
                + " Average of values is used for null values",
        extended = "-- `map_roulette(map<key, number> [, integer seed])` returns key by weighted random selection\n" + 
                "SELECT \n" + 
                "  map_roulette(to_map(a, b)) -- 25% Tom, 21% Zhang, 54% Wang\n" + 
                "FROM ( -- see https://issues.apache.org/jira/browse/HIVE-17406\n" + 
                "  select 'Wang' as a, 54 as b\n" + 
                "  union all\n" + 
                "  select 'Zhang' as a, 21 as b\n" + 
                "  union all\n" + 
                "  select 'Tom' as a, 25 as b\n" + 
                ") tmp;\n" + 
                "> Wang\n" + 
                "\n" + 
                "-- Weight random selection with using filling nulls with the average value\n" + 
                "SELECT\n" + 
                "  map_roulette(map(1, 0.5, 'Wang', null)), -- 50% Wang, 50% 1\n" + 
                "  map_roulette(map(1, 0.5, 'Wang', null, 'Zhang', null)) -- 1/3 Wang, 1/3 1, 1/3 Zhang\n" + 
                ";\n" + 
                "\n" + 
                "-- NULL will be returned if every key is null\n" + 
                "SELECT \n" + 
                "  map_roulette(map()),\n" + 
                "  map_roulette(map(null, null, null, null));\n" + 
                "> NULL    NULL\n" + 
                "\n" + 
                "-- Return NULL if all weights are zero\n" + 
                "SELECT\n" + 
                "  map_roulette(map(1, 0)),\n" + 
                "  map_roulette(map(1, 0, '5', 0))\n" + 
                ";\n" + 
                "> NULL    NULL\n" + 
                "\n" + 
                "-- map_roulette does not support non-numeric weights or negative weights.\n" + 
                "SELECT map_roulette(map('Wong', 'A string', 'Zhao', 2));\n" + 
                "> HiveException: Error evaluating map_roulette(map('Wong':'A string','Zhao':2))\n" + 
                "SELECT map_roulette(map('Wong', 'A string', 'Zhao', 2));\n" + 
                "> UDFArgumentException: Map value must be greather than or equals to zero: -2")
// @formatter:on
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
        this.valueOI = HiveUtils.asDoubleCompatibleOI(mapOI.getMapValueObjectInspector());

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
            if (v < 0.d) {
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
        double s = 0.d;
        for (Map.Entry<Object, Double> e : m.entrySet()) {
            Object k = e.getKey();
            double v = e.getValue().doubleValue();
            s += v;
            if (s > r) {
                return k;
            }
        }

        return null;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "map_roulette(" + join(children, ',') + ")";
    }

}
