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
package hivemall.knn.distance;

import java.math.BigInteger;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.FloatWritable;

//@formatter:off
@Description(name = "jaccard_distance",
        value = "_FUNC_(integer A, integer B [,int k=128]) - Returns Jaccard distance between A and B",
        extended = "select \n" + 
                "  jaccard_distance(0,3) as c1, \n" + 
                "  jaccard_distance(\"0\",\"3\") as c2, -- 0=0x00, 0=0x11\n" + 
                "  jaccard_distance(0,4) as c3\n" + 
                ";\n" + 
                "\n" + 
                "c1      c2      c3\n" + 
                "0.03125 0.03125 0.015625")
@UDFType(deterministic = true, stateful = false)
//@formatter:on
public final class JaccardDistanceUDF extends UDF {

    private final Set<Object> union = new HashSet<Object>();
    private final Set<Object> intersect = new HashSet<Object>();

    public FloatWritable evaluate(long a, long b) {
        return evaluate(a, b, 128);
    }

    public FloatWritable evaluate(long a, long b, int k) {
        int countMatches = k - HammingDistanceUDF.hammingDistance(a, b);
        float jaccard = countMatches / (float) k;
        float pseudoJaccard = 2.f * (jaccard - 0.5f); // pseudo Jaccard
        return new FloatWritable(1.f - pseudoJaccard);
    }

    public FloatWritable evaluate(String a, String b) {
        return evaluate(a, b, 128);
    }

    public FloatWritable evaluate(String a, String b, int k) {
        BigInteger ai = new BigInteger(a);
        BigInteger bi = new BigInteger(b);
        int countMatches = k - HammingDistanceUDF.hammingDistance(ai, bi);
        float jaccard = countMatches / (float) k;
        float pseudoJaccard = 2.f * (jaccard - 0.5f); // pseudo Jaccard
        return new FloatWritable(1.f - pseudoJaccard);
    }

    public FloatWritable evaluate(final List<String> a, final List<String> b) {
        if (a == null && b == null) {
            return new FloatWritable(0.f);
        } else if (a == null || b == null) {
            return new FloatWritable(1.f);
        }
        final int asize = a.size();
        final int bsize = b.size();
        if (asize == 0 && bsize == 0) {
            return new FloatWritable(0.f);
        } else if (asize == 0 || bsize == 0) {
            return new FloatWritable(1.f);
        }

        union.addAll(a);
        union.addAll(b);
        float unionSize = union.size();
        union.clear();

        intersect.addAll(a);
        intersect.retainAll(b);
        float intersectSize = intersect.size();
        intersect.clear();

        float j = intersectSize / unionSize;
        return new FloatWritable(1.f - j);
    }

}
