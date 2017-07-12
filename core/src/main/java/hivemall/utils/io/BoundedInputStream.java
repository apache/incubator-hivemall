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

/**
 * Input stream which only supplies bytes up to a certain length
 * Implementation is based on BoundedInputStream in Apache Commons IO
 * https://commons.apache.org/proper/commons-io/javadocs/api-2.5/org/apache/commons/io/input/BoundedInputStream.html
 */
public final class BoundedInputStream extends InputStream {
    private static final int EOF = -1;

    private final InputStream is;
    private final long max;
    private long pos = 0L;
    private long mark = EOF;

    public BoundedInputStream(final InputStream is) {
        this(is, EOF); // wrapped, but unlimited input stream
    }

    public BoundedInputStream(final InputStream is, final long size) {
        this.is = is;
        this.max = size;
    }

    @Override
    public int read() throws IOException {
        if (max >= 0 && pos >= max) {
            return EOF;
        }

        int result = is.read();
        this.pos += 1;

        return result;
    }

    @Override
    public int read(final byte[] b) throws IOException {
        return read(b, 0, b.length);
    }

    @Override
    public int read(byte[] b, int off, int len) throws IOException {
        if (max >= 0 && pos >= max) {
            return EOF;
        }

        long maxRead = max >= 0 ? Math.min(len, max - pos) : len;
        int bytesRead = is.read(b, off, (int) maxRead);
        if (bytesRead == EOF) {
            return EOF;
        }
        this.pos += bytesRead;

        return bytesRead;
    }

    @Override
    public long skip(long n) throws IOException {
        long toSkip = max >= 0 ? Math.min(n, max - pos) : n;
        long skippedBytes = is.skip(toSkip);
        this.pos += skippedBytes;
        return skippedBytes;
    }

    @Override
    public int available() throws IOException {
        if (max >= 0 && pos >= max) {
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
        this.pos = mark;
    }

    @Override
    public synchronized void mark(int readlimit) {
        is.mark(readlimit);
        this.mark = pos;
    }

    @Override
    public boolean markSupported() {
        return is.markSupported();
    }

}
