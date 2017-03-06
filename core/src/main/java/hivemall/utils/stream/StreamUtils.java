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
package hivemall.utils.stream;

import hivemall.utils.io.DeflaterOutputStream;
import hivemall.utils.io.FastByteArrayInputStream;
import hivemall.utils.io.FastMultiByteArrayOutputStream;
import hivemall.utils.io.IOUtils;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.NoSuchElementException;
import java.util.zip.Deflater;
import java.util.zip.Inflater;
import java.util.zip.InflaterInputStream;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class StreamUtils {

    private StreamUtils() {}

    @Nonnull
    public static IntStream toCompressedIntStream(@Nonnull final int[] src) throws IOException {
        return toCompressedIntStream(src, Deflater.DEFAULT_COMPRESSION);
    }

    @Nonnull
    public static IntStream toCompressedIntStream(@Nonnull final int[] src, final int level)
            throws IOException {
        FastMultiByteArrayOutputStream bos = new FastMultiByteArrayOutputStream(16384);
        Deflater deflater = new Deflater(level, true);
        DeflaterOutputStream defos = new DeflaterOutputStream(bos, deflater, 8192);
        DataOutputStream dos = new DataOutputStream(defos);

        final int count = src.length;
        final byte[] compressed;
        try {
            for (int i = 0; i < count; i++) {
                dos.writeInt(src[i]);
            }
            defos.finish();
            compressed = bos.toByteArray_clear();
        } catch (IOException e) {
            throw e;
        } finally {
            IOUtils.closeQuietly(dos);
        }

        FastByteArrayInputStream bis = new FastByteArrayInputStream(compressed);
        InflaterInputStream infis = new InflaterInputStream(bis, new Inflater(true), 8192);
        DataInputStream dis = new DataInputStream(infis);

        return new InflateIntStream(dis, count);
    }

    static final class InflateIntStream implements IntStream {

        @Nonnull
        private final DataInputStream in;
        @Nonnegative
        private final int count;
        @Nonnegative
        private int index;

        InflateIntStream(@Nonnull DataInputStream compressed, @Nonnegative int count) {
            this.in = compressed;
            this.count = count;
            this.index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < count;
        }

        @Override
        public int next() {
            if (index < count) {// hasNext()
                final int v;
                try {
                    v = in.readInt();
                } catch (IOException e) {
                    throw new IllegalStateException("Invalid input at " + index, e);
                }
                index++;
                return v;
            }
            throw new NoSuchElementException();
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

    }

}
