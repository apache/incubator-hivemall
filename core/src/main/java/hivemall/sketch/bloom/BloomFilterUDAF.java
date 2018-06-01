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
import org.apache.hadoop.hive.ql.exec.UDAF;
import org.apache.hadoop.hive.ql.exec.UDAFEvaluator;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.bloom.DynamicBloomFilter;
import org.apache.hadoop.util.bloom.Filter;
import org.apache.hadoop.util.bloom.Key;

@Description(name = "bloom_filter",
        value = "_FUNC_(string key) - Constructs a BloomFilter by aggregating a set of keys")
@SuppressWarnings("deprecation")
public final class BloomFilterUDAF extends UDAF {

    public static class Evaluator implements UDAFEvaluator {

        private Filter filter;
        private Key key;

        @Override
        public void init() {
            this.filter = BloomFilterUtils.newDynamicBloomFilter();
            this.key = new Key();
        }

        public boolean iterate(@Nullable Text keyStr) {
            if (keyStr == null) {
                return true;
            }
            key.set(keyStr.getBytes(), 1.0d);

            filter.add(key);

            return true;
        }

        @Nonnull
        public Text terminatePartial() throws HiveException {
            try {
                return BloomFilterUtils.serialize(filter, new Text());
            } catch (IOException e) {
                throw new HiveException(e);
            }
        }

        public boolean merge(@Nonnull Text partial) throws HiveException {
            final DynamicBloomFilter other;
            try {
                other = BloomFilterUtils.deserialize(partial, new DynamicBloomFilter());
            } catch (IOException e) {
                throw new HiveException(e);
            }

            if (filter == null) {
                this.filter = other;
            } else {
                filter.or(other);
            }
            return true;
        }

        @Nullable
        public Text terminate() throws HiveException {
            if (filter == null) {
                return null;
            }

            try {
                return BloomFilterUtils.serialize(filter, new Text());
            } catch (IOException e) {
                throw new HiveException(e);
            }
        }

    }
}
