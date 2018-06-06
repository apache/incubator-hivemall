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

//@formatter:off
@Description(name = "bloom_contains",
        value = "_FUNC_(string bloom, string key) or _FUNC_(string bloom, array<string> keys)"
                + " - Returns true if the bloom filter contains all the given key(s). Returns false if key is null.",
        extended = "WITH satisfied_movies as (\n" + 
                "  SELECT bloom(movieid) as movies\n" + 
                "  FROM (\n" + 
                "    SELECT movieid\n" + 
                "    FROM ratings\n" + 
                "    GROUP BY movieid\n" + 
                "    HAVING avg(rating) >= 4.0\n" + 
                "  ) t\n" + 
                ")\n" + 
                "SELECT\n" + 
                "  l.rating,\n" + 
                "  count(distinct l.userid) as cnt\n" + 
                "FROM\n" + 
                "  ratings l \n" + 
                "  CROSS JOIN satisfied_movies r\n" + 
                "WHERE\n" + 
                "  bloom_contains(r.movies, l.movieid) -- includes false positive\n" + 
                "GROUP BY \n" + 
                "  l.rating;\n" + 
                "\n" + 
                "l.rating        cnt\n" + 
                "1       1296\n" + 
                "2       2770\n" + 
                "3       5008\n" + 
                "4       5824\n" + 
                "5       5925")
//@formatter:on
@UDFType(deterministic = true, stateful = false)
public final class BloomContainsUDF extends UDF {

    @Nonnull
    private final Key key = new Key();

    @Nullable
    private Text prevBfStr;
    @Nullable
    private Filter prevBf;

    @Nullable
    public Boolean evaluate(@Nullable Text bloomStr, @Nullable Text keyStr) throws HiveException {
        if (bloomStr == null) {
            return null;
        }
        if (keyStr == null) {
            return Boolean.FALSE;
        }

        Filter bloom = getFilter(bloomStr);
        key.set(keyStr.copyBytes(), 1.0d);
        return Boolean.valueOf(bloom.membershipTest(key));
    }

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
            key.set(keyStr.copyBytes(), 1.0d);
            if (bloom.membershipTest(key) == false) {
                return Boolean.FALSE;
            }
        }

        return Boolean.TRUE;
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
            this.prevBfStr = new Text(bloomStr);
            this.prevBf = bloom;
        }
        return bloom;
    }

}
