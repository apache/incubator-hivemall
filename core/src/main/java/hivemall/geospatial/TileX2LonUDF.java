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
import hivemall.utils.lang.Preconditions;

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
 * @link http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
 */
@Description(name = "tilex2lon",
        value = "_FUNC_(int x, int zoom)::double - Returns longitude of the given tile x and zoom level")
@UDFType(deterministic = true, stateful = false)
public final class TileX2LonUDF extends GenericUDF {

    private PrimitiveObjectInspector xOI;
    private PrimitiveObjectInspector zoomOI;

    private DoubleWritable result;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentException("_FUNC_ takes exactly 2 arguments: " + argOIs.length);
        }
        this.xOI = HiveUtils.asIntegerOI(argOIs[0]);
        this.zoomOI = HiveUtils.asIntegerOI(argOIs[1]);

        this.result = new DoubleWritable();
        return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
    }

    @Override
    public DoubleWritable evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        Object arg1 = arguments[1].get();

        if (arg0 == null) {
            return null;
        }
        if (arg1 == null) {
            throw new UDFArgumentException("zoom level should not be null");
        }

        int x = PrimitiveObjectInspectorUtils.getInt(arg0, xOI);
        int zoom = PrimitiveObjectInspectorUtils.getInt(arg1, zoomOI);
        Preconditions.checkArgument(zoom >= 0, "Invalid zoom level", UDFArgumentException.class);

        final double lon;
        try {
            lon = GeoSpatialUtils.tilex2lon(x, zoom);
        } catch (IllegalArgumentException ex) {
            throw new UDFArgumentException(ex);
        }

        result.set(lon);
        return result;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "tilex2lon(" + Arrays.toString(children) + ")";
    }

}
