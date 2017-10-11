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

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;

public final class BitInputStream extends DataInputStream {

    public static final int MAX_BITS = 31;

    private int leftBits = 0;
    private int byteBuf = 0;

    public BitInputStream(InputStream in) {
        super(in);
    }

    public final boolean readBit() throws IOException {
        if (--leftBits >= 0) {
            return ((byteBuf >>> leftBits) & 1) == 1;
        }
        leftBits = 7;
        byteBuf = in.read();
        if (byteBuf == -1) {
            throw new EOFException("reached end of stream");
        }
        return ((byteBuf >>> 7) & 1) == 1;
    }

    public final int readBits(int n) throws IOException {
        int x = 0;
        while (n > leftBits) {
            n -= leftBits;
            x |= rightBits(leftBits, byteBuf) << n;
            byteBuf = in.read();
            if (byteBuf == -1) {
                throw new EOFException("reached end of stream");
            }
            leftBits = 8;
        }
        leftBits -= n;
        return x | rightBits(n, byteBuf >>> leftBits);
    }

    private static final int rightBits(final int n, final int x) {
        return x & ((1 << n) - 1);
    }
}
