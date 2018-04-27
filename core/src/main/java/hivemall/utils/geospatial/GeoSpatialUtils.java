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
import static java.lang.Math.atan;
import static java.lang.Math.atan2;
import static java.lang.Math.cos;
import static java.lang.Math.floor;
import static java.lang.Math.log;
import static java.lang.Math.pow;
import static java.lang.Math.sin;
import static java.lang.Math.sinh;
import static java.lang.Math.sqrt;
import static java.lang.Math.tan;
import static java.lang.Math.toDegrees;
import static java.lang.Math.toRadians;

import javax.annotation.Nonnegative;

public final class GeoSpatialUtils {

    public static final double MAX_LATITUDE = 85.0511d;
    public static final double MIN_LATITUDE = -85.0511d;

    private GeoSpatialUtils() {}

    public static int lon2tilex(final double lon, @Nonnegative final int zoom) {
        if (lon < -180.d || lon > 180.d) {
            throw new IllegalArgumentException("Longitude must be in range [-180,+180]: " + lon);
        }
        return (int) floor((lon + 180.d) / 360.d * (1 << zoom));
    }

    public static int lat2tiley(final double lat, @Nonnegative final int zoom) {
        if (lat < MIN_LATITUDE || lat > MAX_LATITUDE) {
            throw new IllegalArgumentException("Latitude must be in range [-85.0511,+85.0511]: "
                    + lat + "\nSee http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames");
        }
        double lat_rad = toRadians(lat);
        int n = 1 << zoom;
        return (int) floor((1.d - log(tan(lat_rad) + sec(lat_rad)) / PI) / 2.d * n);
    }

    public static double tilex2lon(final int x, @Nonnegative final int zoom) {
        return x / pow(2.d, zoom) * 360.d - 180.d;
    }

    public static double tiley2lat(final int y, @Nonnegative final int zoom) {
        double n = PI - (2.d * PI * y) / pow(2.d, zoom);
        return toDegrees(atan(sinh(n)));
    }

    /**
     * @link https://en.wikipedia.org/wiki/Tiled_web_map#Tile_numbering_schemes
     */
    public static long tile(final double lat, final double lon, @Nonnegative final int zoom) {
        int xtile = lon2tilex(lon, zoom);
        int ytile = lat2tiley(lat, zoom);
        long n = 1L << zoom; // 2^z
        return xtile + (n * ytile);
    }

    public static int tiles(final int zoom) {
        return 1 << (zoom * 2); // 2^2z
    }

    /**
     * Return a Haversine distance in Kilometers between two points.
     * 
     * @link http://www.movable-type.co.uk/scripts/latlong.html
     * @link http://rosettacode.org/wiki/Haversine_formula#Java
     * @return distance between two points in Kilometers
     */
    public static double haversineDistance(final double lat1, final double lon1, final double lat2,
            final double lon2) {
        double R = 6371.0d; // Radius of the earth in Km
        double dLat = toRadians(lat2 - lat1); // deg2rad below
        double dLon = toRadians(lon2 - lon1);
        double sinDLat = sin(dLat / 2.d);
        double sinDLon = sin(dLon / 2.d);
        double a =
                sinDLat * sinDLat + cos(toRadians(lat1)) * cos(toRadians(lat2)) * sinDLon * sinDLon;
        double c = 2.d * atan2(sqrt(a), sqrt(1.d - a));
        return R * c; // Distance in Km
    }

}
