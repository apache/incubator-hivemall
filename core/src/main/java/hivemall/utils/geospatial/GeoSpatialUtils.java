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
package hivemall.utils.geospatial;

import static hivemall.utils.math.MathUtils.sec;
import static java.lang.Math.PI;
import static java.lang.Math.floor;
import static java.lang.Math.log;
import static java.lang.Math.tan;

import javax.annotation.Nonnegative;

public final class GeoSpatialUtils {

    public static final double MAX_LATITUDE = 85.0511d;
    public static final double MIN_LATITUDE = -85.0511d;

    private GeoSpatialUtils() {}

    public static int lon2tile(final double lon, @Nonnegative final int zoom) {
        if (lon < -180.d || lon > 180.d) {
            throw new IllegalArgumentException("Longitude must be in range [-180,+180]: " + lon);
        }
        return (int) floor((lon + 180.d) / 360.d * (1 << zoom));
    }

    public static int lat2tile(final double lat, @Nonnegative final int zoom) {
        if (lat < MIN_LATITUDE || lat > MAX_LATITUDE) {
            throw new IllegalArgumentException("Latitude must be in range [-85.0511,+85.0511]: "
                    + lat + "\nSee http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames");
        }
        double lat_rad = Math.toRadians(lat);
        int n = 1 << zoom;
        return (int) floor((1.d - log(tan(lat_rad) + sec(lat_rad)) / PI) / 2.d * n);
    }

    /**
     * @link https://en.wikipedia.org/wiki/Tiled_web_map#Tile_numbering_schemes
     */
    public static int tile(final double lat, final double lon, @Nonnegative final int zoom) {
        int xtile = lon2tile(lon, zoom);
        int ytile = lat2tile(lat, zoom);
        int n = 1 << zoom; // 2^z
        return xtile + (n * ytile);
    }

    public static int tiles(final int zoom) {
        return 1 << (zoom * 2); // 2^2z
    }

}
