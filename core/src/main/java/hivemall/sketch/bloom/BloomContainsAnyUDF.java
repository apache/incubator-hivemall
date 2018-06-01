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
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.bloom.DynamicBloomFilter;
import org.apache.hadoop.util.bloom.Filter;
import org.apache.hadoop.util.bloom.Key;

@Description(name = "bloom_contains_any",
        value = "_FUNC_(string bloom, string key) - Returns true if the bloom filter contains any of the given key")
@UDFType(deterministic = true, stateful = false)
public final class BloomContainsAnyUDF extends UDF {

    @Nonnull
    private final Key key = new Key();

    @Nullable
    private Text prevBfStr;
    @Nullable
    private Filter prevBf;

    @Nullable
    public Boolean evaluate(@Nullable Text bloomStr, @Nullable List<Text> keys)
            throws HiveException {
        if (bloomStr == null) {
            return null;
        }
        if (keys == null) {
            return Boolean.FALSE;
        }

        final Filter bloom = getFilter(bloomStr);

        for (Text keyStr : keys) {
            if (keyStr == null) {
                continue;
            }
            key.set(keyStr.getBytes(), 1.0d);
            if (bloom.membershipTest(key)) {
                return Boolean.TRUE;
            }
        }

        return Boolean.FALSE;
    }

    @Nonnull
    private Filter getFilter(@Nonnull final Text bloomStr) throws HiveException {
        final Filter bloom;
        if (prevBf != null && prevBfStr.equals(bloomStr)) {
            bloom = prevBf;
        } else {
            try {
                bloom = BloomFilterUtils.deserialize(bloomStr, new DynamicBloomFilter());
            } catch (IOException e) {
                throw new HiveException(e);
            }
            this.prevBfStr = bloomStr;
            this.prevBf = bloom;
        }
        return bloom;
    }

}
