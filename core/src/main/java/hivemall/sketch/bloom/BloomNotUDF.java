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

import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.bloom.DynamicBloomFilter;
import org.apache.hadoop.util.bloom.Filter;

@Description(name = "bloom_not",
        value = "_FUNC_(string bloom) - Returns the logical NOT of a bloom filters",
        extended = "SELECT bloom_not(bf) FROM xxx;")
@UDFType(deterministic = true, stateful = false)
public final class BloomNotUDF extends UDF {

    @Nullable
    public Text evaluate(@Nullable Text bloomStr) throws HiveException {
        if (bloomStr == null) {
            return null;
        }

        final Filter bloom;
        try {
            bloom = BloomFilterUtils.deserialize(bloomStr, new DynamicBloomFilter());
        } catch (IOException e) {
            throw new HiveException(e);
        }

        bloom.not();

        try {
            return BloomFilterUtils.serialize(bloom, new Text());
        } catch (IOException e) {
            throw new HiveException(e);
        }
    }
}
