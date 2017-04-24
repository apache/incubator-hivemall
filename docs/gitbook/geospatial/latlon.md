<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

This page introduces Geo-spatial functions that treats latitude and longitude.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.1 or later.

# Tile number function

`tile(double lat, double lon, int zoom)` returns a tile number in `xtile(lon,zoom) + ytile(lat,zoom) * 2^n`. The tile number is in range `[0,2^2z]`.

Formulas to convert latitude and longitude into tile x,y coordinates are as follows:

{% math %}
\begin{aligned}
x &= \left\lfloor \frac{lon + 180}{360} \cdot 2^z \right\rfloor \\ \\
y &=
    \left\lfloor
        \left(
            1 - \frac{
                \ln \left(
                    \tan \left(
                        lat \cdot \frac{\pi}{180}
                    \right) + \frac{1}{\cos \left( lat \cdot \frac{\pi}{180} \right)}
                \right)
            }{\pi}
        \right) \cdot 2^{z - 1}
    \right\rfloor
\end{aligned}
{% endmath %}

Refer [this page](http://wiki.openstreetmap.org/wiki/Slippy_map_tilenames) for detail. Zoom level is well described in [this page](http://wiki.openstreetmap.org/wiki/Zoom_levels).

### Usage

```sql
WITH data as (
  select 51.51202 as lat, 0.02435 as lon, 17 as zoom
  union all
  select 51.51202 as lat, 0.02435 as lon, 4 as zoom
  union all
  select null as lat, 0.02435 as lon, 17 as zoom
)
select 
   tile(lat, lon, zoom) as tile
from 
   data;
```

| tile |
|:--:|
|1417478152|
|88|
|NULL|

# Map URL function

`map_url(double lat, double lon, int zoom [, const string option])` function returns a tile URL of openstreetmap.com or maps.google.com.

The 4th argument takes the following optional arguments:
```sql
hive> select map_url(1,1,1,'-help');

usage: map_url(double lat, double lon, int zoom [, const string option]) -
       Returns a URL string [-help] [-t <arg>]
 -help             Show function help
 -t,--type <arg>   Map type [default: openstreetmap|osm,
                   googlemaps|google]
```

### Usage

```sql
WITH data as (
  select 51.51202 as lat, 0.02435 as lon, 17 as zoom
  union all
  select 51.51202 as lat, 0.02435 as lon, 4 as zoom
  union all
  select null, 0.02435, 17
)
select 
   map_url(lat,lon,zoom) as osm_url,
   map_url(lat,lon,zoom,'-type googlemaps') as gmap_url
from
  data;
```

|osm_url|gmap_url|
|:------:|:--------:|
|http://tile.openstreetmap.org/17/65544/43582.png | https://www.google.com/maps/@51.51202,0.02435,17z| 
|http://tile.openstreetmap.org/4/8/5.png|https://www.google.com/maps/@51.51202,0.02435,4z|
|NULL|NULL|

![http://tile.openstreetmap.org/17/65544/43582.png](http://tile.openstreetmap.org/17/65544/43582.png "http://tile.openstreetmap.org/17/65544/43582.png")