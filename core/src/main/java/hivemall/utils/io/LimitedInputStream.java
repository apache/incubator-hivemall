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

import hivemall.utils.lang.Preconditions;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnegative;

/**
 * Input stream which is limited to a certain length. Implementation is based on LimitedInputStream
 * in Apache Commons FileUpload.
 *
 * @link 
 *       https://commons.apache.org/proper/commons-fileupload/apidocs/org/apache/commons/fileupload/util
 *       /LimitedInputStream.html
 */
public class LimitedInputStream extends FilterInputStream {

    protected final long max;
    protected long pos = 0L;

    public LimitedInputStream(@CheckForNull final InputStream in, @Nonnegative final long maxSize) {
        super(in);
        Preconditions.checkNotNull(in, "Base input stream must not be null");
        this.max = maxSize;
    }

    protected void raiseError() throws IOException {
        throw new IOException("Exceeded maximum size of input stream: limit = " + max
                + " bytes, but pos = " + pos);
    }

    private void proceed(@Nonnegative final long bytes) throws IOException {
        this.pos += bytes;
        if (pos > max) {
            raiseError();
        }
    }

    @Override
    public int read() throws IOException {
        final int res = super.read();
        if (res != -1) {
            proceed(1L);
        }
        return res;
    }

    @Override
    public int read(final byte[] b, final int off, final int len) throws IOException {
        final int res = super.read(b, off, len);
        if (res > 0) {
            proceed(res);
        }
        return res;
    }

    @Override
    public long skip(final long n) throws IOException {
        final long res = super.skip(n);
        if (res > 0) {
            proceed(res);
        }
        return res;
    }
}
