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
package hivemall.utils.hadoop;

import hivemall.utils.io.FastByteArrayInputStream;
import hivemall.utils.io.FastByteArrayOutputStream;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Objects;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;

import org.roaringbitmap.RoaringBitmap;

public final class SerdeUtils {

    @Nonnull
    public static byte[] serializeRoaring(@Nonnull final RoaringBitmap r) {
        r.runOptimize(); // might improve compression
        // next we create the ByteBuffer where the data will be stored
        final byte[] buf = new byte[r.serializedSizeInBytes()];
        // then we can serialize on a custom OutputStream
        try {
            r.serialize(new DataOutputStream(new FastByteArrayOutputStream(buf)));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to serialize RoaringBitmap", e);
        }
        return buf;
    }

    @Nonnull
    public static RoaringBitmap deserializeRoaring(@CheckForNull final byte[] b) {
        final RoaringBitmap bitmap = new RoaringBitmap();
        try {
            bitmap.deserialize(
                new DataInputStream(new FastByteArrayInputStream(Objects.requireNonNull(b))));
        } catch (IOException e) {
            throw new IllegalStateException("Failed to deserialize RoaringBitmap", e);
        }
        return bitmap;
    }


}
