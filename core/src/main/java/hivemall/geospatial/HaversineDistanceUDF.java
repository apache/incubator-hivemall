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
package hivemall.geospatial;

import hivemall.utils.geospatial.GeoSpatialUtils;
import hivemall.utils.hadoop.HiveUtils;

import java.util.Arrays;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

/**
 * A UDF to return Haversine distance between given two points
 * 
 * @link http://www.movable-type.co.uk/scripts/latlong.html
 * @link https://en.wikipedia.org/wiki/Haversine_formula
 * @link https://rosettacode.org/wiki/Haversine_formula
 */
@Description(name = "haversine_distance",
        value = "_FUNC_(double lat1, double lon1, double lat2, double lon2, [const boolean mile=false])::double"
                + " - return distance between two locations in km [or miles] using `haversine` formula",
        extended = "Usage: select latlon_distance(lat1, lon1, lat2, lon2) from ...")
@UDFType(deterministic = true, stateful = false)
public final class HaversineDistanceUDF extends GenericUDF {

    private PrimitiveObjectInspector lat1OI, lon1OI;
    private PrimitiveObjectInspector lat2OI, lon2OI;

    private boolean inMiles;
    private DoubleWritable result;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 4 && argOIs.length != 5) {
            throw new UDFArgumentException("_FUNC_ takes 4 or 5 arguments: " + argOIs.length);
        }
        this.lat1OI = HiveUtils.asDoubleCompatibleOI(argOIs[0]);
        this.lon1OI = HiveUtils.asDoubleCompatibleOI(argOIs[1]);
        this.lat2OI = HiveUtils.asDoubleCompatibleOI(argOIs[2]);
        this.lon2OI = HiveUtils.asDoubleCompatibleOI(argOIs[3]);
        this.inMiles = (argOIs.length == 5) && HiveUtils.getConstBoolean(argOIs[4]);

        this.result = new DoubleWritable();
        return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
    }

    @Override
    public DoubleWritable evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        Object arg1 = arguments[1].get();
        Object arg2 = arguments[2].get();
        Object arg3 = arguments[3].get();

        if (arg0 == null || arg1 == null || arg2 == null || arg3 == null) {
            return null;
        }
        double lat1 = PrimitiveObjectInspectorUtils.getDouble(arg0, lat1OI);
        double lon1 = PrimitiveObjectInspectorUtils.getDouble(arg1, lon1OI);
        double lat2 = PrimitiveObjectInspectorUtils.getDouble(arg2, lat2OI);
        double lon2 = PrimitiveObjectInspectorUtils.getDouble(arg3, lon2OI);

        final double distance;
        try {
            distance = GeoSpatialUtils.haversineDistance(lat1, lon1, lat2, lon2);
        } catch (IllegalArgumentException ex) {
            throw new UDFArgumentException(ex);
        }

        if (inMiles) {
            double miles = distance / 1.609344d;
            result.set(miles);
        } else {
            result.set(distance);
        }

        return result;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "haversine_distance(" + Arrays.toString(children) + ")";
    }

}
