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
package hivemall.sketch.bloom;

import static hivemall.utils.math.MathUtils.LOG2;

import hivemall.utils.io.Base91InputStream;
import hivemall.utils.io.Base91OutputStream;
import hivemall.utils.io.FastByteArrayInputStream;
import hivemall.utils.io.FastByteArrayOutputStream;

import java.io.DataInput;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.bloom.BloomFilter;
import org.apache.hadoop.util.bloom.DynamicBloomFilter;
import org.apache.hadoop.util.bloom.Filter;
import org.apache.hadoop.util.hash.Hash;

public final class BloomFilterUtils {

    public static final int DEFAULT_BLOOM_FILTER_SIZE = 1024 * 1024;
    public static final float DEFAULT_ERROR_RATE = 0.005f;
    public static final int NUM_HASHES = 5;

    @Nonnull
    public static BloomFilter newBloomFilter(@Nonnegative final int expectedNumberOfElements) {
        return newBloomFilter(expectedNumberOfElements, DEFAULT_ERROR_RATE);
    }

    @Nonnull
    public static BloomFilter newBloomFilter(@Nonnegative final int expectedNumberOfElements,
            @Nonnegative final float errorRate) {
        // k = ceil(-log_2(false prob.))
        int nbHash = Math.max(2, (int) Math.ceil(-(Math.log(errorRate) / LOG2)));
        return newBloomFilter(expectedNumberOfElements, errorRate, nbHash);
    }

    @Nonnull
    public static BloomFilter newBloomFilter(@Nonnegative final int expectedNumberOfElements,
            @Nonnegative final float errorRate, @Nonnegative final int nbHash) {
        // vector size should be `-kn / (ln(1 - c^(1/k)))` bits for
        // single key, where `k` is the number of hash functions,
        // `n` is the number of keys and `c` is the desired max error rate.
        int vectorSize = (int) Math.ceil((-nbHash * expectedNumberOfElements)
                / Math.log(1.d - Math.pow(errorRate, 1.d / nbHash)));
        return new BloomFilter(vectorSize, nbHash, Hash.MURMUR_HASH);
    }

    @Nonnull
    public static DynamicBloomFilter newDynamicBloomFilter() {
        return newDynamicBloomFilter(DEFAULT_BLOOM_FILTER_SIZE, DEFAULT_ERROR_RATE, NUM_HASHES);
    }

    @Nonnull
    public static DynamicBloomFilter newDynamicBloomFilter(
            @Nonnegative final int expectedNumberOfElements) {
        return newDynamicBloomFilter(expectedNumberOfElements, DEFAULT_ERROR_RATE);
    }

    @Nonnull
    public static DynamicBloomFilter newDynamicBloomFilter(
            @Nonnegative final int expectedNumberOfElements, @Nonnegative final float errorRate) {
        // k = ceil(-log_2(false prob.))
        int nbHash = Math.max(2, (int) Math.ceil(-(Math.log(errorRate) / LOG2)));
        return newDynamicBloomFilter(expectedNumberOfElements, errorRate, nbHash);
    }

    @Nonnull
    public static DynamicBloomFilter newDynamicBloomFilter(
            @Nonnegative final int expectedNumberOfElements, @Nonnegative final float errorRate,
            @Nonnegative final int nbHash) {
        int vectorSize = (int) Math.ceil((-nbHash * expectedNumberOfElements)
                / Math.log(1.d - Math.pow(errorRate, 1.d / nbHash)));
        return new DynamicBloomFilter(vectorSize, nbHash, Hash.MURMUR_HASH,
            expectedNumberOfElements);
    }

    @Nonnull
    public static byte[] serialize(@Nonnull final Filter filter) throws IOException {
        FastByteArrayOutputStream bos = new FastByteArrayOutputStream();
        Base91OutputStream base91 = new Base91OutputStream(bos);
        DataOutputStream out = new DataOutputStream(base91);
        filter.write(out);
        out.flush();
        base91.finish();
        return bos.toByteArray();
    }

    @Nonnull
    public static Text serialize(@Nonnull final Filter filter, @Nonnull final Text dst)
            throws IOException {
        FastByteArrayOutputStream bos = new FastByteArrayOutputStream();
        Base91OutputStream base91 = new Base91OutputStream(bos);
        DataOutputStream out = new DataOutputStream(base91);
        filter.write(out);
        out.flush();
        base91.finish();
        dst.set(bos.getInternalArray(), 0, bos.size());
        return dst;
    }

    @Nonnull
    public static <F extends Filter> F deserialize(@Nonnull final Text in, @Nonnull final F dst)
            throws IOException {
        return deserialize(in.getBytes(), 0, in.getLength(), dst);
    }

    @Nonnull
    public static <F extends Filter> F deserialize(@Nonnull final byte[] buf, @Nonnull final F dst)
            throws IOException {
        return deserialize(buf, 0, buf.length, dst);
    }

    @Nonnull
    public static <F extends Filter> F deserialize(@Nonnull final byte[] buf,
            @Nonnegative final int offset, @Nonnegative final int len, @Nonnull final F dst)
            throws IOException {
        FastByteArrayInputStream fis = new FastByteArrayInputStream(buf, offset, len);
        DataInput in = new DataInputStream(new Base91InputStream(fis));
        dst.readFields(in);
        return dst;
    }

}
