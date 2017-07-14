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
package hivemall.utils.io;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLConnection;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class HttpUtils {

    private HttpUtils() {}

    @Nonnull
    public static HttpURLConnection getHttpURLConnection(@Nonnull String urlStr)
            throws IllegalArgumentException, IOException {
        if (!urlStr.startsWith("http://") && !urlStr.startsWith("https://")) {
            throw new IllegalArgumentException("Unexpected url: " + urlStr);
        }
        URL url = new URL(urlStr);
        URLConnection conn = url.openConnection();
        return (HttpURLConnection) conn;
    }

    @Nonnull
    public static InputStream getLimitedInputStream(@Nonnull HttpURLConnection conn,
            @Nonnegative long size) throws IOException {
        InputStream is = conn.getInputStream();
        return new LimitedInputStream(is, size);
    }
}
