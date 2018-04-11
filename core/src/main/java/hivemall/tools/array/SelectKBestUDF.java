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
package hivemall.tools.array;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Preconditions;

import java.io.IOException;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@Description(name = "select_k_best",
        value = "_FUNC_(array<number> array, const array<number> importance, const int k)"
                + " - Returns selected top-k elements as array<double>")
@UDFType(deterministic = true, stateful = false)
public final class SelectKBestUDF extends GenericUDF {

    private ListObjectInspector featuresOI;
    private PrimitiveObjectInspector featureOI;
    private ListObjectInspector importanceListOI;
    private PrimitiveObjectInspector importanceElemOI;

    private int _k;
    private List<DoubleWritable> _result;
    private int[] _topKIndices;

    @Override
    public ObjectInspector initialize(ObjectInspector[] OIs) throws UDFArgumentException {
        if (OIs.length != 3) {
            throw new UDFArgumentLengthException("Specify three arguments: " + OIs.length);
        }

        if (!HiveUtils.isNumberListOI(OIs[0])) {
            throw new UDFArgumentTypeException(0,
                "Only array<number> type argument is acceptable but " + OIs[0].getTypeName()
                        + " was passed as `features`");
        }
        if (!HiveUtils.isNumberListOI(OIs[1])) {
            throw new UDFArgumentTypeException(1,
                "Only array<number> type argument is acceptable but " + OIs[1].getTypeName()
                        + " was passed as `importance_list`");
        }
        if (!HiveUtils.isIntegerOI(OIs[2])) {
            throw new UDFArgumentTypeException(2, "Only int type argument is acceptable but "
                    + OIs[2].getTypeName() + " was passed as `k`");
        }

        this.featuresOI = HiveUtils.asListOI(OIs[0]);
        this.featureOI = HiveUtils.asDoubleCompatibleOI(featuresOI.getListElementObjectInspector());
        this.importanceListOI = HiveUtils.asListOI(OIs[1]);
        this.importanceElemOI =
                HiveUtils.asDoubleCompatibleOI(importanceListOI.getListElementObjectInspector());

        this._k = HiveUtils.getConstInt(OIs[2]);
        Preconditions.checkArgument(_k >= 1, UDFArgumentException.class);
        final List<DoubleWritable> result = new ArrayList<>(_k);
        for (int i = 0; i < _k; i++) {
            result.add(new DoubleWritable());
        }
        this._result = result;

        return ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
    }

    @Override
    public List<DoubleWritable> evaluate(DeferredObject[] dObj) throws HiveException {
        final double[] features = HiveUtils.asDoubleArray(dObj[0].get(), featuresOI, featureOI);
        final double[] importanceList =
                HiveUtils.asDoubleArray(dObj[1].get(), importanceListOI, importanceElemOI);

        Preconditions.checkNotNull(features, UDFArgumentException.class);
        Preconditions.checkNotNull(importanceList, UDFArgumentException.class);
        Preconditions.checkArgument(features.length == importanceList.length,
            UDFArgumentException.class);
        Preconditions.checkArgument(features.length >= _k, UDFArgumentException.class);

        int[] topKIndices = _topKIndices;
        if (topKIndices == null) {
            final List<Map.Entry<Integer, Double>> list =
                    new ArrayList<Map.Entry<Integer, Double>>();
            for (int i = 0; i < importanceList.length; i++) {
                list.add(new AbstractMap.SimpleEntry<Integer, Double>(i, importanceList[i]));
            }
            Collections.sort(list, new Comparator<Map.Entry<Integer, Double>>() {
                @Override
                public int compare(Map.Entry<Integer, Double> o1, Map.Entry<Integer, Double> o2) {
                    return o1.getValue() > o2.getValue() ? -1 : 1;
                }
            });

            topKIndices = new int[_k];
            for (int i = 0; i < topKIndices.length; i++) {
                topKIndices[i] = list.get(i).getKey();
            }
            this._topKIndices = topKIndices;
        }

        final List<DoubleWritable> result = _result;
        for (int i = 0; i < topKIndices.length; i++) {
            int idx = topKIndices[i];
            DoubleWritable d = result.get(i);
            double f = features[idx];
            d.set(f);
        }
        return result;
    }

    @Override
    public void close() throws IOException {
        // help GC
        this._result = null;
        this._topKIndices = null;
    }

    @Override
    public String getDisplayString(String[] children) {
        final StringBuilder sb = new StringBuilder();
        sb.append("select_k_best");
        sb.append("(");
        if (children.length > 0) {
            sb.append(children[0]);
            for (int i = 1; i < children.length; i++) {
                sb.append(", ");
                sb.append(children[i]);
            }
        }
        sb.append(")");
        return sb.toString();
    }
}
