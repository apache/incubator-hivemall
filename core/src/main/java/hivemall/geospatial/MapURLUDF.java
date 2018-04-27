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

import hivemall.UDFWithOptions;
import hivemall.utils.geospatial.GeoSpatialUtils;
import hivemall.utils.hadoop.HiveUtils;

import java.util.Arrays;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.Text;

@Description(name = "map_url",
        value = "_FUNC_(double lat, double lon, int zoom [, const string option]) - Returns a URL string",
        extended = "OpenStreetMap: http://tile.openstreetmap.org/${zoom}/${xtile}/${ytile}.png\n"
                + "Google Maps: https://www.google.com/maps/@${lat},${lon},${zoom}z")
@UDFType(deterministic = true, stateful = false)
public final class MapURLUDF extends UDFWithOptions {

    private PrimitiveObjectInspector latOI;
    private PrimitiveObjectInspector lonOI;
    private PrimitiveObjectInspector zoomOI;

    private MapType type = MapType.openstreetmap;
    private Text result;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("t", "type", true,
            "Map type [default: openstreetmap|osm, googlemaps|google]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(String optionValue) throws UDFArgumentException {
        CommandLine cl = parseOptions(optionValue);

        this.type = MapType.resolve(cl.getOptionValue("type", "openstreetmap"));
        return cl;
    }

    public enum MapType {
        openstreetmap, googlemaps;

        @Nonnull
        public static final MapType resolve(@Nonnull String type) throws UDFArgumentException {
            if ("openstreetmap".equalsIgnoreCase(type) || "osm".equalsIgnoreCase(type)) {
                return openstreetmap;
            } else if ("googlemaps".equalsIgnoreCase(type) || "google".equalsIgnoreCase(type)) {
                return googlemaps;
            } else {
                throw new UDFArgumentException("Illegal map type: " + type);
            }
        }
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 3 && argOIs.length != 4) {
            throw new UDFArgumentException("_FUNC_ takes 3 or 4 arguments: " + argOIs.length);
        }
        if (argOIs.length == 4) {
            String opts = HiveUtils.getConstString(argOIs[3]);
            processOptions(opts);
        }

        this.latOI = HiveUtils.asDoubleCompatibleOI(argOIs[0]);
        this.lonOI = HiveUtils.asDoubleCompatibleOI(argOIs[1]);
        this.zoomOI = HiveUtils.asIntegerOI(argOIs[2]);

        this.result = new Text();
        return PrimitiveObjectInspectorFactory.writableStringObjectInspector;
    }

    @Override
    public Text evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        Object arg1 = arguments[1].get();
        Object arg2 = arguments[2].get();

        if (arg0 == null || arg1 == null) {
            return null;
        }
        if (arg2 == null) {
            throw new UDFArgumentException("zoom level is null");
        }

        double lat = PrimitiveObjectInspectorUtils.getDouble(arg0, latOI);
        double lon = PrimitiveObjectInspectorUtils.getDouble(arg1, lonOI);
        int zoom = PrimitiveObjectInspectorUtils.getInt(arg2, zoomOI);

        result.set(toMapURL(lat, lon, zoom, type));
        return result;
    }

    @Nonnull
    private static String toMapURL(double lat, double lon, int zoom, @Nonnull MapType type)
            throws UDFArgumentException {
        if (type == MapType.openstreetmap) {// http://tile.openstreetmap.org/${zoom}/${xtile}/${ytile}.png            
            if (zoom < 0 || zoom > 19) {
                throw new UDFArgumentException(
                    "Illegal zoom level. Supported zoom level for openstreetmap is [0,19]: "
                            + zoom);
            }
            final int xtile, ytile;
            try {
                xtile = GeoSpatialUtils.lon2tilex(lon, zoom);
                ytile = GeoSpatialUtils.lat2tiley(lat, zoom);
            } catch (IllegalArgumentException ex) {
                throw new UDFArgumentException(ex);
            }
            return "http://tile.openstreetmap.org/" + Integer.toString(zoom) + '/'
                    + Integer.toString(xtile) + '/' + Integer.toString(ytile) + ".png";
        } else if (type == MapType.googlemaps) {// https://www.google.com/maps/@${lat},${lon},${zoom}z
            if (zoom < 0 || zoom > 21) {
                throw new UDFArgumentException(
                    "Illegal zoom level. Supported zoom level for Google Maps is [0,21]: " + zoom);
            }
            return "https://www.google.com/maps/@" + Double.toString(lat) + ','
                    + Double.toString(lon) + ',' + Integer.toString(zoom) + 'z';
        } else {
            throw new UDFArgumentException("Unexpected map type: " + type);
        }
    }

    @Override
    public String getDisplayString(String[] children) {
        return "map_url(" + Arrays.toString(children) + ")";
    }

}
