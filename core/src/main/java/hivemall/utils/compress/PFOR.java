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
package hivemall.utils.compress;

import hivemall.utils.codec.BitwiseCodec;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.io.BitInputStream;
import hivemall.utils.io.FastByteArrayInputStream;
import hivemall.utils.io.FastByteArrayOutputStream;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Arrays;

public final class PFOR {

    private PFOR() {}

    public static final class CompressedSegment {

        // header
        int totalEntries_ = 0;
        final int bitwidth_; // at most 128
        final transient int maxcode_;

        // entry points
        byte[] entryStartList = new byte[12];
        int[] entryStartException = new int[12];

        // code section
        final transient BitwiseCodec codesBuf;
        final int[] codes;

        // exception section
        final IntArrayList exceptionList;
        int firstException_ = -1;
        transient int nextException_ = 0;

        public CompressedSegment(int bitwidth) {
            this.bitwidth_ = bitwidth;
            this.maxcode_ = 2 ^ bitwidth;
            this.codesBuf = new BitwiseCodec(4096);
            this.codes = null;
            this.exceptionList = new IntArrayList(64);
        }

        public CompressedSegment(int totalEntries, int bitwidth, int firstException, int[] codes,
                IntArrayList exceptionList) {
            this.totalEntries_ = totalEntries;
            this.bitwidth_ = bitwidth;
            this.maxcode_ = 2 ^ bitwidth;
            this.firstException_ = firstException;
            this.codesBuf = null;
            this.codes = codes;
            this.exceptionList = exceptionList;
        }

        public void writeTo(final DataOutputStream out) throws IOException {
            out.writeInt(totalEntries_);
            out.writeByte(bitwidth_);
            out.writeInt(firstException_);
            codesBuf.writeTo(out, true);
            int exceptions = exceptionList.size();
            out.writeShort(exceptions);
            for (int i = 0; i < exceptions; i++) {
                int c = exceptionList.get(i);
                out.writeInt(c);
            }
        }

        public static CompressedSegment readFrom(final byte[] in) throws IOException {
            FastByteArrayInputStream bis = new FastByteArrayInputStream(in);
            try (DataInputStream dis = new DataInputStream(bis)) {
                int totalEntries = dis.readInt();
                int bitwidth = dis.readByte();
                int firstException = dis.readInt();
                int len = dis.readInt();
                byte[] b = new byte[len];
                dis.readFully(b, 0, len);
                FastByteArrayInputStream codesIs = new FastByteArrayInputStream(b);
                BitInputStream codesBis = new BitInputStream(codesIs);
                int[] codes = new int[totalEntries];
                unpack(codesBis, bitwidth, codes, totalEntries);
                int exceptions = dis.readShort();
                IntArrayList exceptionList = new IntArrayList(exceptions);
                for (int i = 0; i < exceptions; i++) {
                    int c = dis.readInt();
                    exceptionList.add(c);
                }
                return new CompressedSegment(totalEntries, bitwidth, firstException, codes,
                    exceptionList);
            }
        }

    }

    public static byte[] compress(final int[] input) {
        int n = input.length;
        int bitwidth = estimateAdequateBitWidth(input, n);
        CompressedSegment segment = new CompressedSegment(bitwidth);
        compress(input, n, segment);
        FastByteArrayOutputStream bos = new FastByteArrayOutputStream(4096);
        try (DataOutputStream dos = new DataOutputStream(bos)) {
            segment.writeTo(dos);
            dos.flush();
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return bos.toByteArray();
    }

    public static byte[] compress(final int[] input, final int bitwidth) {
        CompressedSegment segment = new CompressedSegment(bitwidth);
        compress(input, input.length, segment);
        FastByteArrayOutputStream bos = new FastByteArrayOutputStream(4096);
        try (DataOutputStream dos = new DataOutputStream(bos)) {
            segment.writeTo(dos);
            dos.flush();
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return bos.toByteArray();
    }

    public static int compress(final int[] input, final int n, final CompressedSegment segment) {
        final int[] miss = new int[n];
        final int[] data = new int[n];
        final int maxcode = segment.maxcode_;
        int prev = segment.nextException_;

        // LOOP1: find exceptions
        int j = 0;
        for (int i = 0; i < n; i++) {
            int val = input[i];
            data[i] = val;
            miss[j] = i;
            j += (val > maxcode) ? 1 : 0; // can't eliminate if-then-else control hazard
        }
        // LOOP2: create patch-list        
        final IntArrayList exceptionList = segment.exceptionList;
        if (j > 0) {
            segment.firstException_ = miss[0];
            prev = miss[j - 1]; // last-patch
            for (int i = 0; i < j; i++) {
                int cur = miss[i];
                exceptionList.add(input[cur]);
                data[prev] = cur - prev - 1; // difference of the two exception
                prev = cur;
            }
        }

        pack(segment, data, n); // bit-pack the values

        segment.totalEntries_ += n;
        segment.nextException_ = prev;
        return j; // # of exceptions
    }

    private static void pack(final CompressedSegment segment, final int[] data, final int n) {
        final BitwiseCodec codes = segment.codesBuf;
        final int nbits = segment.bitwidth_;
        for (int i = 0; i < n; i++) {
            codes.putBits(data[i], nbits);
        }
    }

    public static int[] decompressAsInts(final byte[] in) {
        final CompressedSegment segment;
        try {
            segment = CompressedSegment.readFrom(in);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return decompressAsInts(segment.totalEntries_, segment);
    }

    public static int[] decompressAsInts(final int n, final CompressedSegment segment) {
        final int[] code = segment.codes;

        // LOOP1: decode regardless
        final int[] output = Arrays.copyOf(code, n);

        // LOOP2: patch it up
        int firstException = segment.firstException_;
        if (firstException != -1) {
            final IntArrayList exceptionList = segment.exceptionList;
            int cur = code[firstException]; // REVIEWME  exceptionList was empty
            for (int i = 0, next; cur < n; i++, cur = next) {
                output[cur] = exceptionList.get(i);
                next = cur + code[cur] + 1;
            }
            segment.nextException_ = cur - n;
        }

        return output;
    }

    public static int finegrainedDecompress(final int x, final CompressedSegment segment) {
        final int[] code = segment.codes;
        int entryIdx = x >> 7;
        int i = segment.entryStartList[entryIdx] + (x & ~127);
        int j = segment.entryStartException[entryIdx];
        while (i > x) {
            i += code[i];
            j--;
        }
        if (i == x) {
            return segment.exceptionList.get(j);
        } else {
            return code[x];
        }
    }

    private static void unpack(final BitInputStream codes, final int bitwidth, final int[] code,
            final int n) throws IOException {
        for (int i = 0; i < n; i++) {
            code[i] = codes.readBits(bitwidth);
        }
    }

    public final static int estimateAdequateBitWidth(final int[] v, final int s) {
        final int[] sorted = Arrays.copyOf(v, s);
        Arrays.sort(sorted, 0, s);

        float minEst = Float.MAX_VALUE;
        int bitwidth = 1;
        for (int b = 1; b < 16; b++) {
            float erate = exceptionRate(sorted, s, b);
            float nrate = 1f - erate;
            float est = (nrate * b) + (erate * 16);
            if (est < minEst) {
                minEst = est;
                bitwidth = b;
            }
        }
        return bitwidth;
    }

    private static float exceptionRate(final int[] v, final int s, final int b) {
        int lenb = pforAnalyzeBits(v, s, b);
        return (float) (s - lenb) / s;
    }

    /**
     * calculate the longest stretch of value starts, such that the difference between first and
     * last is representable in "b" bits.
     */
    private static int pforAnalyzeBits(final int[] v, final int n, final int b) {
        assert (v != null);
        assert (n >= 0) : n;
        assert (b > 0) : b;

        int len = 0, range = 1 << b;
        for (int lo = 0, hi = 0; hi < n; hi++) {
            if (v[hi] - v[lo] >= range) {
                if (hi - lo >= len) {
                    len = hi - lo;
                }
                while (v[hi] - v[++lo] >= range);
            }
        }
        return len + 1;
    }

}
