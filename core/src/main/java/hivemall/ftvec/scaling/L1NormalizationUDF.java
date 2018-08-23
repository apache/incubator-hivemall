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
package hivemall.ftvec.scaling;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.Text;

import java.util.Arrays;
import java.util.List;

@Description(name = "l1_normalize", value = "_FUNC_(ftvec string) - Returned a L1 normalized value")
@UDFType(deterministic = true, stateful = false)
public final class L1NormalizationUDF extends UDF {

    public List<Text> evaluate(final List<Text> ftvecs) throws HiveException {
        if (ftvecs == null) {
            return null;
        }
        double absoluteSum = 0.d;
        final int numFeatures = ftvecs.size();
        final String[] features = new String[numFeatures];
        final float[] weights = new float[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            Text ftvec = ftvecs.get(i);
            if (ftvec == null) {
                continue;
            }
            String s = ftvec.toString();
            final String[] ft = s.split(":");
            final int ftlen = ft.length;
            if (ftlen == 1) {
                features[i] = ft[0];
                weights[i] = 1.f;
                absoluteSum += 1.d;
            } else if (ftlen == 2) {
                features[i] = ft[0];
                float v = Float.parseFloat(ft[1]);
                weights[i] = v;
                absoluteSum += Math.abs(v);
            } else if (ftlen == 3) {
                features[i] = ft[0] + ':' + ft[1];
                float v = Float.parseFloat(ft[2]);
                weights[i] = v;
                absoluteSum += Math.abs(v);
            } else {
                throw new HiveException("Invalid feature value representation: " + s);
            }
        }
        final float norm = (float) absoluteSum;
        final Text[] t = new Text[numFeatures];
        if (norm == 0.f) {
            for (int i = 0; i < numFeatures; i++) {
                String f = features[i];
                t[i] = new Text(f + ':' + 0.f);
            }
        } else {
            for (int i = 0; i < numFeatures; i++) {
                String f = features[i];
                float v = weights[i] / norm;
                t[i] = new Text(f + ':' + v);
            }
        }
        return Arrays.asList(t);
    }

}
