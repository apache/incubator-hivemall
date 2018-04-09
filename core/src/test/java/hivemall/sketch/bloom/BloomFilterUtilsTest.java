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

import java.io.IOException;
import java.util.Random;

import org.apache.hadoop.util.bloom.DynamicBloomFilter;
import org.apache.hadoop.util.bloom.Key;
import org.junit.Assert;
import org.junit.Test;

public class BloomFilterUtilsTest {

    @Test
    public void testDynamicBloomFilter() {
        DynamicBloomFilter dbf = BloomFilterUtils.newDynamicBloomFilter(300000);
        final Key key = new Key();

        final Random rnd1 = new Random(43L);
        for (int i = 0; i < 1000000; i++) {
            double d = rnd1.nextGaussian();
            String s = Double.toHexString(d);
            key.set(s.getBytes(), 1.0);
            dbf.add(key);
        }

        final Random rnd2 = new Random(43L);
        for (int i = 0; i < 1000000; i++) {
            double d = rnd2.nextGaussian();
            String s = Double.toHexString(d);
            key.set(s.getBytes(), 1.0);
            Assert.assertTrue(dbf.membershipTest(key));
        }
    }

    @Test
    public void testDynamicBloomFilterSerde() throws IOException {
        final Key key = new Key();

        DynamicBloomFilter dbf1 = BloomFilterUtils.newDynamicBloomFilter(300000);
        final Random rnd1 = new Random(43L);
        for (int i = 0; i < 1000000; i++) {
            double d = rnd1.nextGaussian();
            String s = Double.toHexString(d);
            key.set(s.getBytes(), 1.0);
            dbf1.add(key);
        }

        DynamicBloomFilter dbf2 = BloomFilterUtils.deserialize(BloomFilterUtils.serialize(dbf1),
            new DynamicBloomFilter());
        final Random rnd2 = new Random(43L);
        for (int i = 0; i < 1000000; i++) {
            double d = rnd2.nextGaussian();
            String s = Double.toHexString(d);
            key.set(s.getBytes(), 1.0);
            Assert.assertTrue(dbf2.membershipTest(key));
        }
    }


}
