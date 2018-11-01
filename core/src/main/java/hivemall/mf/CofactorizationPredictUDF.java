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
package hivemall.mf;

import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.io.FloatWritable;

@Description(name = "cofactor_predict",
        value = "_FUNC_(array<float> theta, array<float> beta) - Returns the prediction value")
@UDFType(deterministic = true, stateful = false)
public final class CofactorizationPredictUDF extends UDF {

    private static final double DEFAULT_RESULT = 0.d;

    @Nonnull
    public DoubleWritable evaluate(@Nullable List<FloatWritable> Pu, @Nullable List<FloatWritable> Qi) throws HiveException {
        if (Pu == null || Qi == null) {
            return new DoubleWritable(DEFAULT_RESULT);
        }

        final int PuSize = Pu.size();
        final int QiSize = Qi.size();
        // workaround for TD
        if (PuSize == 0) {
            return new DoubleWritable(DEFAULT_RESULT);
        } else if (QiSize == 0) {
            return new DoubleWritable(DEFAULT_RESULT);
        }

        if (QiSize != PuSize) {
            throw new HiveException("|Pu| " + PuSize + " was not equal to |Qi| " + QiSize);
        }

        double ret = DEFAULT_RESULT;
        for (int k = 0; k < PuSize; k++) {
            FloatWritable Pu_k = Pu.get(k);
            if (Pu_k == null) {
                continue;
            }
            FloatWritable Qi_k = Qi.get(k);
            if (Qi_k == null) {
                continue;
            }
            ret += Pu_k.get() * Qi_k.get();
        }
        return new DoubleWritable(ret);
    }
}
