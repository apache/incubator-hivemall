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
package hivemall.ftvec.binning;

import hivemall.annotations.VisibleForTesting;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

// @formatter:off
@Description(name = "feature_binning",
        value = "_FUNC_(array<features::string> features, map<string, array<number>> quantiles_map)"
                + " - returns a binned feature vector as an array<features::string>\n"
                + "_FUNC_(number weight, array<number> quantiles) - returns bin ID as int",
                extended = "WITH extracted as (\n" + 
                        "  select \n" + 
                        "    extract_feature(feature) as index,\n" + 
                        "    extract_weight(feature) as value\n" + 
                        "  from\n" + 
                        "    input l\n" + 
                        "    LATERAL VIEW explode(features) r as feature\n" + 
                        "),\n" + 
                        "mapping as (\n" + 
                        "  select\n" + 
                        "    index, \n" + 
                        "    build_bins(value, 5, true) as quantiles -- 5 bins with auto bin shrinking\n" + 
                        "  from\n" + 
                        "    extracted\n" + 
                        "  group by\n" + 
                        "    index\n" + 
                        "),\n" + 
                        "bins as (\n" + 
                        "   select \n" + 
                        "    to_map(index, quantiles) as quantiles \n" + 
                        "   from\n" + 
                        "    mapping\n" + 
                        ")\n" + 
                        "select\n" + 
                        "  l.features as original,\n" + 
                        "  feature_binning(l.features, r.quantiles) as features\n" + 
                        "from\n" + 
                        "  input l\n" + 
                        "  cross join bins r\n\n" +
                        "> [\"name#Jacob\",\"gender#Male\",\"age:20.0\"] [\"name#Jacob\",\"gender#Male\",\"age:2\"]\n" +
                        "> [\"name#Isabella\",\"gender#Female\",\"age:20.0\"]    [\"name#Isabella\",\"gender#Female\",\"age:2\"]")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class FeatureBinningUDF extends GenericUDF {
    private boolean multiple = true;

    private ListObjectInspector featuresOI;
    private StringObjectInspector featureOI;
    private MapObjectInspector quantilesMapOI;
    private StringObjectInspector keyOI;
    private ListObjectInspector quantilesOI;
    private PrimitiveObjectInspector quantileOI;
    private PrimitiveObjectInspector weightOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentLengthException("Specify two arguments :" + argOIs.length);
        }

        if (HiveUtils.isListOI(argOIs[0]) && HiveUtils.isMapOI(argOIs[1])) {
            // feature_binning(array<features::string> features, map<string, array<number>> quantiles_map)

            if (!HiveUtils.isStringOI(
                ((ListObjectInspector) argOIs[0]).getListElementObjectInspector())) {
                throw new UDFArgumentTypeException(0,
                    "Only array<string> type argument can be accepted but "
                            + argOIs[0].getTypeName() + " was passed as `features`");
            }
            featuresOI = HiveUtils.asListOI(argOIs[0]);
            featureOI = HiveUtils.asStringOI(featuresOI.getListElementObjectInspector());

            quantilesMapOI = HiveUtils.asMapOI(argOIs[1]);
            if (!HiveUtils.isStringOI(quantilesMapOI.getMapKeyObjectInspector())
                    || !HiveUtils.isListOI(quantilesMapOI.getMapValueObjectInspector())
                    || !HiveUtils.isNumberOI(
                        ((ListObjectInspector) quantilesMapOI.getMapValueObjectInspector()).getListElementObjectInspector())) {
                throw new UDFArgumentTypeException(1,
                    "Only map<string, array<number>> type argument can be accepted but "
                            + argOIs[1].getTypeName() + " was passed as `quantiles_map`");
            }
            keyOI = HiveUtils.asStringOI(quantilesMapOI.getMapKeyObjectInspector());
            quantilesOI = HiveUtils.asListOI(quantilesMapOI.getMapValueObjectInspector());
            quantileOI =
                    HiveUtils.asDoubleCompatibleOI(quantilesOI.getListElementObjectInspector());

            multiple = true;

            return ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        } else if (HiveUtils.isPrimitiveOI(argOIs[0]) && HiveUtils.isListOI(argOIs[1])) {
            // feature_binning(number weight, array<number> quantiles)

            weightOI = HiveUtils.asDoubleCompatibleOI(argOIs[0]);

            quantilesOI = HiveUtils.asListOI(argOIs[1]);
            if (!HiveUtils.isNumberOI(quantilesOI.getListElementObjectInspector())) {
                throw new UDFArgumentTypeException(1,
                    "Only array<number> type argument can be accepted but "
                            + argOIs[1].getTypeName() + " was passed as `quantiles`");
            }
            quantileOI =
                    HiveUtils.asDoubleCompatibleOI(quantilesOI.getListElementObjectInspector());

            multiple = false;

            return PrimitiveObjectInspectorFactory.writableIntObjectInspector;
        } else {
            throw new UDFArgumentTypeException(0,
                "Only <array<features::string>, map<string, array<number>>> "
                        + "or <number, array<number>> type arguments can be accepted but <"
                        + argOIs[0].getTypeName() + ", " + argOIs[1].getTypeName()
                        + "> was passed.");
        }
    }

    private transient Map<String, double[]> quantilesMap;
    private transient double[] quantilesArray;

    @Override
    public Object evaluate(DeferredObject[] args) throws HiveException {
        final Object arg0 = args[0].get();
        if (arg0 == null) {
            return null;
        }
        final Object arg1 = args[1].get();
        if (arg1 == null) {
            throw new UDFArgumentException(
                "The second argument (i.e., quantiles) MUST be non-null value");
        }

        if (multiple) {
            if (quantilesMap == null) {
                final Map<?, ?> map = quantilesMapOI.getMap(arg1);
                quantilesMap = new HashMap<String, double[]>(map.size() * 2);
                for (Map.Entry<?, ?> e : map.entrySet()) {
                    String k = keyOI.getPrimitiveJavaObject(e.getKey());
                    double[] v = HiveUtils.asDoubleArray(e.getValue(), quantilesOI, quantileOI);
                    quantilesMap.put(k, v);
                }
            }

            final List<?> features = featuresOI.getList(arg0);
            final List<Text> result = new ArrayList<Text>();
            for (Object f : features) {
                final String entry = featureOI.getPrimitiveJavaObject(f);

                final int pos = entry.indexOf(':');
                if (pos < 0) { // categorical
                    result.add(new Text(entry));
                } else { // quantitative
                    final String k = entry.substring(0, pos);
                    String v = entry.substring(pos + 1);
                    final double[] bins = quantilesMap.get(k);
                    if (bins != null) { // binning
                        v = String.valueOf(findBin(bins, Double.parseDouble(v)));
                    }
                    result.add(new Text(k + ':' + v));
                }
            }
            return result;
        } else {
            if (quantilesArray == null) {
                quantilesArray = HiveUtils.asDoubleArray(arg1, quantilesOI, quantileOI);
            }

            return new IntWritable(
                findBin(quantilesArray, PrimitiveObjectInspectorUtils.getDouble(arg0, weightOI)));
        }
    }

    @VisibleForTesting
    static int findBin(@Nonnull final double[] quantiles, final double value) throws HiveException {
        if (quantiles.length < 3) {
            throw new HiveException(
                "Length of `quantiles` should be greater than or equal to three but "
                        + quantiles.length + ".");
        }

        final int pos = Arrays.binarySearch(quantiles, value);
        return (pos < 0) ? ~pos - 1 : (pos == 0) ? 0 : pos - 1;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "feature_binning(" + StringUtils.join(children, ',') + ')';
    }
}
