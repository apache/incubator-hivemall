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

@Description(name = "bloom_contains",
        value = "_FUNC_(string bloom, string key) - Returns true if the bloom filter contains the given key")
@UDFType(deterministic = true, stateful = false)
public final class BloomContainsUDF extends UDF {

    @Nonnull
    private final Key key = new Key();

    @Nullable
    private Text prevKey;
    @Nullable
    private Filter prevFilter;

    @Nullable
    public Boolean evaluate(@Nullable Text bloomStr, @Nullable Text keyStr) throws HiveException {
        if (bloomStr == null || key == null) {
            return null;
        }

        final Filter bloom;
        if (prevFilter != null && prevKey.equals(keyStr)) {
            bloom = prevFilter;
        } else {
            try {
                bloom = BloomFilterUtils.deserialize(bloomStr, new DynamicBloomFilter());
            } catch (IOException e) {
                throw new HiveException(e);
            }
            this.prevKey = keyStr;
            this.prevFilter = bloom;
            key.set(keyStr.getBytes(), 1.0d);
        }

        return Boolean.valueOf(bloom.membershipTest(key));
    }

}
