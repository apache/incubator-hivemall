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
package hivemall.ftvec.selection;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.SizeOf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFParameterInfo;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.LongObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

@Description(name = "snr", value = "_FUNC_(array<number> features, array<int> one-hot class label)"
        + " - Returns Signal Noise Ratio for each feature as array<double>")
public class SignalNoiseRatioUDAF extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(GenericUDAFParameterInfo info)
            throws SemanticException {
        final ObjectInspector[] OIs = info.getParameterObjectInspectors();

        if (OIs.length != 2) {
            throw new UDFArgumentLengthException("Specify two arguments: " + OIs.length);
        }
        if (!HiveUtils.isNumberListOI(OIs[0])) {
            throw new UDFArgumentTypeException(0,
                "Only array<number> type argument is acceptable but " + OIs[0].getTypeName()
                        + " was passed as `features`");
        }
        if (!HiveUtils.isListOI(OIs[1]) || !HiveUtils.isIntegerOI(
            ((ListObjectInspector) OIs[1]).getListElementObjectInspector())) {
            throw new UDFArgumentTypeException(1, "Only array<int> type argument is acceptable but "
                    + OIs[1].getTypeName() + " was passed as `labels`");
        }

        return new SignalNoiseRatioUDAFEvaluator();
    }

    static class SignalNoiseRatioUDAFEvaluator extends GenericUDAFEvaluator {
        // PARTIAL1 and COMPLETE
        private ListObjectInspector featuresOI;
        private PrimitiveObjectInspector featureOI;
        private ListObjectInspector labelsOI;
        private PrimitiveObjectInspector labelOI;

        // PARTIAL2 and FINAL
        private StructObjectInspector structOI;
        private StructField countsField, meansField, variancesField;
        private ListObjectInspector countsOI;
        private LongObjectInspector countOI;
        private ListObjectInspector meansOI;
        private ListObjectInspector meanListOI;
        private DoubleObjectInspector meanElemOI;
        private ListObjectInspector variancesOI;
        private ListObjectInspector varianceListOI;
        private DoubleObjectInspector varianceElemOI;

        @AggregationType(estimable = true)
        static class SignalNoiseRatioAggregationBuffer extends AbstractAggregationBuffer {
            long[] counts;
            double[][] means;
            double[][] variances;

            @Override
            public int estimate() {
                return counts == null ? 0
                        : SizeOf.LONG * counts.length
                                + SizeOf.DOUBLE * means.length * means[0].length
                                + SizeOf.DOUBLE * variances.length * variances[0].length;
            }

            public void init(int nClasses, int nFeatures) {
                this.counts = new long[nClasses];
                this.means = new double[nClasses][nFeatures];
                this.variances = new double[nClasses][nFeatures];
            }

            public void reset() {
                if (counts != null) {
                    Arrays.fill(counts, 0);
                    for (double[] mean : means) {
                        Arrays.fill(mean, 0.d);
                    }
                    for (double[] variance : variances) {
                        Arrays.fill(variance, 0.d);
                    }
                }
            }
        }

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] OIs) throws HiveException {
            super.init(mode, OIs);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.featuresOI = HiveUtils.asListOI(OIs[0]);
                this.featureOI =
                        HiveUtils.asDoubleCompatibleOI(featuresOI.getListElementObjectInspector());
                this.labelsOI = HiveUtils.asListOI(OIs[1]);
                this.labelOI = HiveUtils.asIntegerOI(labelsOI.getListElementObjectInspector());
            } else {// from partial aggregation
                this.structOI = (StructObjectInspector) OIs[0];
                this.countsField = structOI.getStructFieldRef("counts");
                this.countsOI = HiveUtils.asListOI(countsField.getFieldObjectInspector());
                this.countOI = HiveUtils.asLongOI(countsOI.getListElementObjectInspector());
                this.meansField = structOI.getStructFieldRef("means");
                this.meansOI = HiveUtils.asListOI(meansField.getFieldObjectInspector());
                this.meanListOI = HiveUtils.asListOI(meansOI.getListElementObjectInspector());
                this.meanElemOI = HiveUtils.asDoubleOI(meanListOI.getListElementObjectInspector());
                this.variancesField = structOI.getStructFieldRef("variances");
                this.variancesOI = HiveUtils.asListOI(variancesField.getFieldObjectInspector());
                this.varianceListOI =
                        HiveUtils.asListOI(variancesOI.getListElementObjectInspector());
                this.varianceElemOI =
                        HiveUtils.asDoubleOI(varianceListOI.getListElementObjectInspector());
            }

            // initialize output
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
                fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableLongObjectInspector));
                fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.writableDoubleObjectInspector)));
                fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.writableDoubleObjectInspector)));
                return ObjectInspectorFactory.getStandardStructObjectInspector(
                    Arrays.asList("counts", "means", "variances"), fieldOIs);
            } else {// terminate
                return ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
            }
        }

        @Override
        public AbstractAggregationBuffer getNewAggregationBuffer() throws HiveException {
            SignalNoiseRatioAggregationBuffer myAgg = new SignalNoiseRatioAggregationBuffer();
            reset(myAgg);
            return myAgg;
        }

        @Override
        public void reset(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            SignalNoiseRatioAggregationBuffer myAgg = (SignalNoiseRatioAggregationBuffer) agg;
            myAgg.reset();
        }

        @Override
        public void iterate(@SuppressWarnings("deprecation") AggregationBuffer agg,
                Object[] parameters) throws HiveException {
            final Object featuresObj = parameters[0];
            final Object labelsObj = parameters[1];

            Preconditions.checkNotNull(featuresObj);
            Preconditions.checkNotNull(labelsObj);

            final SignalNoiseRatioAggregationBuffer myAgg = (SignalNoiseRatioAggregationBuffer) agg;

            final List<?> labels = labelsOI.getList(labelsObj);
            final int nClasses = labels.size();
            Preconditions.checkArgument(nClasses >= 2, UDFArgumentException.class);

            final List<?> features = featuresOI.getList(featuresObj);
            final int nFeatures = features.size();
            Preconditions.checkArgument(nFeatures >= 1, UDFArgumentException.class);

            if (myAgg.counts == null) {
                myAgg.init(nClasses, nFeatures);
            } else {
                Preconditions.checkArgument(nClasses == myAgg.counts.length,
                    UDFArgumentException.class);
                Preconditions.checkArgument(nFeatures == myAgg.means[0].length,
                    UDFArgumentException.class);
            }

            // incrementally calculates means and variance
            // http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
            final int clazz = hotIndex(labels, labelOI);
            final long n = myAgg.counts[clazz];
            myAgg.counts[clazz]++;
            for (int i = 0; i < nFeatures; i++) {
                final double x =
                        PrimitiveObjectInspectorUtils.getDouble(features.get(i), featureOI);
                final double meanN = myAgg.means[clazz][i];
                final double varianceN = myAgg.variances[clazz][i];
                myAgg.means[clazz][i] = (n * meanN + x) / (n + 1.d);
                myAgg.variances[clazz][i] =
                        (n * varianceN + (x - meanN) * (x - myAgg.means[clazz][i])) / (n + 1.d);
            }
        }

        private static int hotIndex(@Nonnull List<?> labels, PrimitiveObjectInspector labelOI)
                throws UDFArgumentException {
            final int nClasses = labels.size();

            int clazz = -1;
            for (int i = 0; i < nClasses; i++) {
                final int label = PrimitiveObjectInspectorUtils.getInt(labels.get(i), labelOI);
                if (label == 1) {// assumes one hot encoding 
                    if (clazz != -1) {
                        throw new UDFArgumentException(
                            "Specify one-hot vectorized array. Multiple hot elements found.");
                    }
                    clazz = i;
                } else {
                    if (label != 0) {
                        throw new UDFArgumentException(
                            "Assumed one-hot encoding (0/1) but found an invalid label: " + label);
                    }
                }
            }
            if (clazz == -1) {
                throw new UDFArgumentException(
                    "Specify one-hot vectorized array for label. Hot element not found.");
            }
            return clazz;
        }

        @Override
        public void merge(@SuppressWarnings("deprecation") AggregationBuffer agg, Object other)
                throws HiveException {
            if (other == null) {
                return;
            }

            final SignalNoiseRatioAggregationBuffer myAgg = (SignalNoiseRatioAggregationBuffer) agg;

            final List<?> counts =
                    countsOI.getList(structOI.getStructFieldData(other, countsField));
            final List<?> means = meansOI.getList(structOI.getStructFieldData(other, meansField));
            final List<?> variances =
                    variancesOI.getList(structOI.getStructFieldData(other, variancesField));

            final int nClasses = counts.size();
            final int nFeatures = meanListOI.getListLength(means.get(0));
            if (myAgg.counts == null) {
                myAgg.init(nClasses, nFeatures);
            }

            for (int i = 0; i < nClasses; i++) {
                final long n = myAgg.counts[i];
                final long cnt = PrimitiveObjectInspectorUtils.getLong(counts.get(i), countOI);

                // no need to merge class `i`
                if (cnt == 0) {
                    continue;
                }

                final List<?> mean = meanListOI.getList(means.get(i));
                final List<?> variance = varianceListOI.getList(variances.get(i));

                myAgg.counts[i] += cnt;
                for (int j = 0; j < nFeatures; j++) {
                    final double meanN = myAgg.means[i][j];
                    final double meanM =
                            PrimitiveObjectInspectorUtils.getDouble(mean.get(j), meanElemOI);
                    final double varianceN = myAgg.variances[i][j];
                    final double varianceM = PrimitiveObjectInspectorUtils.getDouble(
                        variance.get(j), varianceElemOI);

                    if (n == 0) {// only assign `other` into `myAgg`
                        myAgg.means[i][j] = meanM;
                        myAgg.variances[i][j] = varianceM;
                    } else {
                        // merge by Chan's method
                        // http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf
                        myAgg.means[i][j] = (n * meanN + cnt * meanM) / (double) (n + cnt);
                        myAgg.variances[i][j] = (varianceN * (n - 1) + varianceM * (cnt - 1)
                                + Math.pow(meanN - meanM, 2) * n * cnt / (n + cnt)) / (n + cnt - 1);
                    }
                }
            }
        }

        @Override
        public Object terminatePartial(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            final SignalNoiseRatioAggregationBuffer myAgg = (SignalNoiseRatioAggregationBuffer) agg;

            final Object[] partialResult = new Object[3];
            partialResult[0] = WritableUtils.toWritableList(myAgg.counts);
            final List<List<DoubleWritable>> means = new ArrayList<List<DoubleWritable>>();
            for (double[] mean : myAgg.means) {
                means.add(WritableUtils.toWritableList(mean));
            }
            partialResult[1] = means;
            final List<List<DoubleWritable>> variances = new ArrayList<List<DoubleWritable>>();
            for (double[] variance : myAgg.variances) {
                variances.add(WritableUtils.toWritableList(variance));
            }
            partialResult[2] = variances;
            return partialResult;
        }

        @Override
        public Object terminate(@SuppressWarnings("deprecation") AggregationBuffer agg)
                throws HiveException {
            final SignalNoiseRatioAggregationBuffer myAgg = (SignalNoiseRatioAggregationBuffer) agg;

            final int nClasses = myAgg.counts.length;
            final int nFeatures = myAgg.means[0].length;

            // compute SNR among classes for each feature
            final double[] result = new double[nFeatures];
            final double[] sds = new double[nClasses]; // for memorization
            for (int i = 0; i < nFeatures; i++) {
                sds[0] = Math.sqrt(myAgg.variances[0][i]);
                for (int j = 1; j < nClasses; j++) {
                    sds[j] = Math.sqrt(myAgg.variances[j][i]);
                    // `ns[j] == 0` means no feature entry belongs to class `j`. Then, skip the entry.
                    if (myAgg.counts[j] == 0) {
                        continue;
                    }
                    for (int k = 0; k < j; k++) {
                        // avoid comparing between classes having only single entry
                        if (myAgg.counts[k] == 0
                                || (myAgg.counts[j] == 1 && myAgg.counts[k] == 1)) {
                            continue;
                        }

                        // SUM(snr) GROUP BY feature
                        final double snr =
                                Math.abs(myAgg.means[j][i] - myAgg.means[k][i]) / (sds[j] + sds[k]);
                        // if `NaN`(when diff between means and both sds are zero, IOW, all related values are equal),
                        // regard feature `i` as meaningless between class `j` and `k`. So, skip the entry.
                        if (!Double.isNaN(snr)) {
                            result[i] += snr; // accept `Infinity`
                        }
                    }
                }
            }

            return WritableUtils.toWritableList(result);
        }
    }
}
