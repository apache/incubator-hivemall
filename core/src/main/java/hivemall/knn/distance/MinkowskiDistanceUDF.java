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

import hivemall.model.FeatureValue;
import hivemall.utils.hadoop.HiveUtils;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.FloatWritable;

//@formatter:off
@Description(name = "minkowski_distance",
        value = "_FUNC_(list x, list y, double p) - Returns sum(|x - y|^p)^(1/p)", 
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
                "  minkowski_distance(l.features, r.features, 1) as distance1, -- p=1 (manhattan_distance)\n" + 
                "  minkowski_distance(l.features, r.features, 2) as distance2, -- p=2 (euclid_distance)\n" + 
                "  minkowski_distance(l.features, r.features, 3) as distance3, -- p=3\n" + 
                "  manhattan_distance(l.features, r.features) as manhattan_distance,\n" + 
                "  euclid_distance(l.features, r.features) as euclid_distance\n" + 
                "from \n" + 
                "  docs l\n" + 
                "  CROSS JOIN docs r\n" + 
                "where\n" + 
                "  l.docid != r.docid\n" + 
                "order by \n" + 
                "  doc1 asc,\n" + 
                "  distance1 asc;\n" + 
                "\n" + 
                "doc1    doc2    distance1       distance2       distance3       manhattan_distance      euclid_distance\n" + 
                "1       2       4.0     2.4494898       2.1544347       4.0     2.4494898\n" + 
                "1       3       5.0     2.6457512       2.2239802       5.0     2.6457512\n" + 
                "2       3       1.0     1.0     1.0     1.0     1.0\n" + 
                "2       1       4.0     2.4494898       2.1544347       4.0     2.4494898\n" + 
                "3       2       1.0     1.0     1.0     1.0     1.0\n" + 
                "3       1       5.0     2.6457512       2.2239802       5.0     2.6457512")
@UDFType(deterministic = true, stateful = false)
//@formatter:on
public final class MinkowskiDistanceUDF extends GenericUDF {

    private ListObjectInspector arg0ListOI, arg1ListOI;
    private double order_p;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 3) {
            throw new UDFArgumentException("minkowski_distance takes 3 arguments");
        }
        this.arg0ListOI = HiveUtils.asListOI(argOIs, 0);
        this.arg1ListOI = HiveUtils.asListOI(argOIs, 1);
        this.order_p = HiveUtils.getAsConstDouble(argOIs[2]);

        return PrimitiveObjectInspectorFactory.writableFloatObjectInspector;
    }

    @Override
    public FloatWritable evaluate(DeferredObject[] arguments) throws HiveException {
        List<String> ftvec1 = HiveUtils.asStringList(arguments[0], arg0ListOI);
        List<String> ftvec2 = HiveUtils.asStringList(arguments[1], arg1ListOI);
        float d = (float) minkowskiDistance(ftvec1, ftvec2, order_p);
        return new FloatWritable(d);
    }

    public static double minkowskiDistance(final List<String> ftvec1, final List<String> ftvec2,
            final double orderP) {
        final FeatureValue probe = new FeatureValue();
        final Map<String, Float> map = new HashMap<String, Float>(ftvec1.size() * 2 + 1);
        for (String ft : ftvec1) {
            if (ft == null) {
                continue;
            }
            FeatureValue.parseFeatureAsString(ft, probe);
            float v1 = probe.getValueAsFloat();
            String f1 = probe.getFeature();
            map.put(f1, v1);
        }
        double d = 0.d;
        for (String ft : ftvec2) {
            if (ft == null) {
                continue;
            }
            FeatureValue.parseFeatureAsString(ft, probe);
            String f2 = probe.getFeature();
            float v2f = probe.getValueAsFloat();
            Float v1 = map.remove(f2);
            if (v1 == null) {
                d += Math.abs(v2f);
            } else {
                float v1f = v1.floatValue();
                d += Math.pow(Math.abs(v1f - v2f), orderP);
            }
        }
        for (Map.Entry<String, Float> e : map.entrySet()) {
            float v1f = e.getValue();
            d += Math.pow(Math.abs(v1f), orderP);
        }
        return Math.pow(d, 1.d / orderP);
    }

    @Override
    public String getDisplayString(String[] children) {
        return "minkowski_distance(" + Arrays.toString(children) + ")";
    }

}
