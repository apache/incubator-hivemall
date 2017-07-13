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

import javax.annotation.Nonnegative;
import java.io.IOException;
import java.io.InputStream;

/**
 * Input stream which is limited to a certain length
 * Implementation is based on BoundedInputStream in Apache Commons IO and LimitedInputStream in Apache Commons FileUpload
 *
 * @link https://commons.apache.org/proper/commons-io/javadocs/api-2.5/org/apache/commons/io/input/BoundedInputStream.html
 * @link https://commons.apache.org/proper/commons-fileupload/apidocs/org/apache/commons/fileupload/util/LimitedInputStream.html
 */
public final class LimitedInputStream extends InputStream {

    private final InputStream is;
    private final long max;
    private long pos = 0L;

    public LimitedInputStream(final InputStream is, @Nonnegative final long size) {
        this.is = is;
        this.max = size;
    }

    private void proceed(@Nonnegative long bytes) throws IOException {
        this.pos += bytes;
        if (pos > max) {
            throw new IOException("Exceeded maximum size of input stream [" + max + " bytes]");
        }
    }

    @Override
    public int read() throws IOException {
        int result = is.read();
        proceed(1L);
        return result;
    }

    @Override
    public int read(final byte[] b) throws IOException {
        return read(b, 0, b.length);
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        int bytesRead = is.read(b, off, len);
        if (bytesRead > 0) {
            proceed(bytesRead);
        }
        return bytesRead;
    }

    @Override
    public long skip(long n) throws IOException {
        long skippedBytes = is.skip(n);
        proceed(skippedBytes);
        return skippedBytes;
    }

    @Override
    public int available() throws IOException {
        if (pos == max) {
            return 0;
        }
        return is.available();
    }

    @Override
    public void close() throws IOException {
        is.close();
    }

    @Override
    public synchronized void reset() throws IOException {
        is.reset();
    }

    @Override
    public synchronized void mark(int readlimit) {
        is.mark(readlimit);
    }

    @Override
    public boolean markSupported() {
        return is.markSupported();
    }

}
