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
package hivemall.factorization.mf;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Preconditions;

import javax.annotation.Nullable;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

@Description(name = "mf_predict",
        value = "_FUNC_(array<double> Pu, array<double> Qi[, double Bu, double Bi[, double mu]]) - Returns the prediction value")
@UDFType(deterministic = true, stateful = false)
public final class MFPredictionUDF extends GenericUDF {

    private ListObjectInspector puOI, qiOI;
    private PrimitiveObjectInspector puElemOI, qiElemOI;

    @Nullable
    private PrimitiveObjectInspector buOI, biOI, muOI;

    private DoubleWritable result;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 2 || argOIs.length > 5) {
            throw new UDFArgumentException("mf_predict takes 2~5 arguments: " + argOIs.length);
        }

        this.puOI = HiveUtils.asListOI(argOIs, 0);
        this.puElemOI = HiveUtils.asFloatingPointOI(puOI.getListElementObjectInspector());
        this.qiOI = HiveUtils.asListOI(argOIs, 1);
        this.qiElemOI = HiveUtils.asFloatingPointOI(qiOI.getListElementObjectInspector());

        switch (argOIs.length) {
            case 3:
                this.muOI = HiveUtils.asNumberOI(argOIs, 2);
                break;
            case 4:
                this.buOI = HiveUtils.asNumberOI(argOIs, 2);
                this.biOI = HiveUtils.asNumberOI(argOIs, 3);
                break;
            case 5:
                this.buOI = HiveUtils.asNumberOI(argOIs, 2);
                this.biOI = HiveUtils.asNumberOI(argOIs, 3);
                this.muOI = HiveUtils.asNumberOI(argOIs, 4);
                break;
            default:
                break;
        }

        this.result = new DoubleWritable();
        return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
    }

    @Override
    public Object evaluate(DeferredObject[] args) throws HiveException {
        Preconditions.checkArgument(args.length >= 2 && args.length <= 5, args.length);

        @Nullable
        double[] pu = HiveUtils.asDoubleArray(args[0].get(), puOI, puElemOI);
        @Nullable
        double[] qi = HiveUtils.asDoubleArray(args[1].get(), qiOI, qiElemOI);

        double mu = 0.d, bu = 0.d, bi = 0.d;
        switch (args.length) {
            case 3: {
                Object arg2 = args[2].get();
                if (arg2 != null) {
                    mu = PrimitiveObjectInspectorUtils.getDouble(arg2, muOI);
                }
                break;
            }
            case 4: {
                Object arg2 = args[2].get();
                if (arg2 != null) {
                    bu = PrimitiveObjectInspectorUtils.getDouble(arg2, buOI);
                }
                Object arg3 = args[3].get();
                if (arg3 != null) {
                    bi = PrimitiveObjectInspectorUtils.getDouble(arg3, biOI);
                }
                break;
            }
            case 5: {
                Object arg2 = args[2].get();
                if (arg2 != null) {
                    bu = PrimitiveObjectInspectorUtils.getDouble(arg2, buOI);
                }
                Object arg3 = args[3].get();
                if (arg3 != null) {
                    bi = PrimitiveObjectInspectorUtils.getDouble(arg3, biOI);
                }
                Object arg4 = args[4].get();
                if (arg4 != null) {
                    mu = PrimitiveObjectInspectorUtils.getDouble(arg4, muOI);
                }
                break;
            }
            default:
                break;
        }

        double predicted = mfPredict(pu, qi, bu, bi, mu);
        result.set(predicted);
        return result;
    }

    private static double mfPredict(@Nullable final double[] Pu, @Nullable final double[] Qi,
            final double Bu, final double Bi, final double mu) throws UDFArgumentException {
        if (Pu == null) {
            if (Qi == null) {
                return mu;
            } else {
                return mu + Bi;
            }
        } else if (Qi == null) {
            return mu + Bu;
        }
        // workaround for TD
        if (Pu.length == 0) {
            if (Qi.length == 0) {
                return mu;
            } else {
                return mu + Bi;
            }
        } else if (Qi.length == 0) {
            return mu + Bu;
        }

        if (Pu.length != Qi.length) {
            throw new UDFArgumentException(
                "|Pu| " + Pu.length + " was not equal to |Qi| " + Qi.length);
        }

        double ret = mu + Bu + Bi;
        for (int k = 0, size = Pu.length; k < size; k++) {
            double pu_k = Pu[k];
            double qi_k = Qi[k];
            ret += pu_k * qi_k;
        }
        return ret;
    }

    @Override
    public String getDisplayString(String[] args) {
        return "mf_predict(" + StringUtils.join(args, ',') + ')';
    }

}
