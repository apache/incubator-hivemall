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
package hivemall.knn.similarity;

import static hivemall.utils.hadoop.WritableUtils.val;

import hivemall.knn.distance.HammingDistanceUDF;

import java.math.BigInteger;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.FloatWritable;

//@formatter:off
@Description(name = "jaccard_similarity",
        value = "_FUNC_(A, B [,int k]) - Returns Jaccard similarity coefficient of A and B",
        extended = "WITH docs as (\n" + 
                "  select 1 as docid, array('apple:1.0', 'orange:2.0', 'banana:1.0', 'kuwi:0') as features\n" + 
                "  union all\n" + 
                "  select 2 as docid, array('apple:1.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features\n" + 
                "  union all\n" + 
                "  select 3 as docid, array('apple:2.0', 'orange:0', 'banana:2.0', 'kuwi:1.0') as features\n" + 
                ") \n" + 
                "select\n" + 
                "  l.docid as doc1,\n" + 
                "  r.docid as doc2,\n" + 
                "  jaccard_similarity(l.features, r.features) as similarity\n" + 
                "from \n" + 
                "  docs l\n" + 
                "  CROSS JOIN docs r\n" + 
                "where\n" + 
                "  l.docid != r.docid\n" + 
                "order by \n" + 
                "  doc1 asc,\n" + 
                "  similarity desc;\n" + 
                "\n" + 
                "doc1    doc2    similarity\n" + 
                "1       2       0.14285715\n" + 
                "1       3       0.0\n" + 
                "2       3       0.6\n" + 
                "2       1       0.14285715\n" + 
                "3       2       0.6\n" + 
                "3       1       0.0")
//@formatter:on
@UDFType(deterministic = true, stateful = false)
public final class JaccardIndexUDF extends UDF {

    private final Set<Object> union = new HashSet<Object>();
    private final Set<Object> intersect = new HashSet<Object>();

    public FloatWritable evaluate(long a, long b) {
        return evaluate(a, b, 128);
    }

    public FloatWritable evaluate(long a, long b, int k) {
        int countMatches = k - HammingDistanceUDF.hammingDistance(a, b);
        float jaccard = countMatches / (float) k;
        return val(2.f * (jaccard - 0.5f));
    }

    public FloatWritable evaluate(String a, String b) {
        return evaluate(a, b, 128);
    }

    public FloatWritable evaluate(String a, String b, int k) {
        BigInteger ai = new BigInteger(a);
        BigInteger bi = new BigInteger(b);
        int countMatches = k - HammingDistanceUDF.hammingDistance(ai, bi);
        float jaccard = countMatches / (float) k;
        return val(2.f * (jaccard - 0.5f));
    }

    public FloatWritable evaluate(final List<String> a, final List<String> b) {
        if (a == null && b == null) {
            return new FloatWritable(1.f);
        } else if (a == null || b == null) {
            return new FloatWritable(0.f);
        }
        final int asize = a.size();
        final int bsize = b.size();
        if (asize == 0 && bsize == 0) {
            return new FloatWritable(1.f);
        } else if (asize == 0 || bsize == 0) {
            return new FloatWritable(0.f);
        }

        union.addAll(a);
        union.addAll(b);
        float unionSize = union.size();
        union.clear();

        intersect.addAll(a);
        intersect.retainAll(b);
        float intersectSize = intersect.size();
        intersect.clear();

        return new FloatWritable(intersectSize / unionSize);
    }

}
