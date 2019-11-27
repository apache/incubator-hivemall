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

import hivemall.knn.similarity.CosineSimilarityUDF;
import hivemall.utils.hadoop.HiveUtils;

import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.FloatWritable;

/**
 * @link http://en.wikipedia.org/wiki/Cosine_similarity
 */
//@formatter:off
@Description(name = "cosine_distance",
        value = "_FUNC_(ftvec1, ftvec2) - Returns a cosine distance of the given two vectors",
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
                "  cosine_distance(l.features, r.features) as distance,\n" + 
                "  distance2similarity(cosine_distance(l.features, r.features)) as similarity\n" + 
                "from \n" + 
                "  docs l\n" + 
                "  CROSS JOIN docs r\n" + 
                "where\n" + 
                "  l.docid != r.docid\n" + 
                "order by \n" + 
                "  doc1 asc,\n" + 
                "  distance asc;\n" + 
                "\n" + 
                "doc1    doc2    distance        similarity\n" + 
                "1       3       0.45566893      0.6869694\n" + 
                "1       2       0.5     0.6666667\n" + 
                "2       3       0.04742068      0.95472616\n" + 
                "2       1       0.5     0.6666667\n" + 
                "3       2       0.04742068      0.95472616\n" + 
                "3       1       0.45566893      0.6869694")
@UDFType(deterministic = true, stateful = false)
//@formatter:on
public final class CosineDistanceUDF extends GenericUDF {

    private ListObjectInspector arg0ListOI, arg1ListOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentException("cosine_distance takes 2 arguments");
        }
        this.arg0ListOI = HiveUtils.asListOI(argOIs[0]);
        this.arg1ListOI = HiveUtils.asListOI(argOIs[1]);

        return PrimitiveObjectInspectorFactory.writableFloatObjectInspector;
    }

    @Override
    public FloatWritable evaluate(DeferredObject[] arguments) throws HiveException {
        List<String> ftvec1 = HiveUtils.asStringList(arguments[0], arg0ListOI);
        List<String> ftvec2 = HiveUtils.asStringList(arguments[1], arg1ListOI);
        float d = 1.f - CosineSimilarityUDF.cosineSimilarity(ftvec1, ftvec2);
        return new FloatWritable(d);
    }

    @Deprecated
    public FloatWritable evaluate(List<String> ftvec1, List<String> ftvec2) {
        float d = 1.f - CosineSimilarityUDF.cosineSimilarity(ftvec1, ftvec2);
        return new FloatWritable(d);
    }

    @Override
    public String getDisplayString(String[] children) {
        return "cosine_distance(" + Arrays.toString(children) + ")";
    }

}
