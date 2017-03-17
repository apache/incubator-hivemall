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
package hivemall.evaluation;

import hivemall.utils.hadoop.HiveUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.SortedMap;
import java.util.TreeMap;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.parse.SemanticException;
import org.apache.hadoop.hive.ql.udf.generic.AbstractGenericUDAFResolver;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDAFEvaluator.AbstractAggregationBuffer;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryMap;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.WritableIntObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.io.LongWritable;

@SuppressWarnings("deprecation")
@Description(name = "auc",
        value = "_FUNC_(array rankItems | double score, array correctItems | int label "
                + "[, const int recommendSize = rankItems.size ])" + " - Returns AUC")
public final class AUCUDAF extends AbstractGenericUDAFResolver {

    @Override
    public GenericUDAFEvaluator getEvaluator(@Nonnull TypeInfo[] typeInfo) throws SemanticException {
        if (typeInfo.length != 2 && typeInfo.length != 3) {
            throw new UDFArgumentTypeException(typeInfo.length - 1,
                "_FUNC_ takes two or three arguments");
        }

        if (HiveUtils.isNumberTypeInfo(typeInfo[0]) && HiveUtils.isIntegerTypeInfo(typeInfo[1])) {
            return new ClassificationEvaluator();
        } else {
            ListTypeInfo arg1type = HiveUtils.asListTypeInfo(typeInfo[0]);
            if (!HiveUtils.isPrimitiveTypeInfo(arg1type.getListElementTypeInfo())) {
                throw new UDFArgumentTypeException(0,
                    "The first argument `array rankItems` is invalid form: " + typeInfo[0]);
            }

            ListTypeInfo arg2type = HiveUtils.asListTypeInfo(typeInfo[1]);
            if (!HiveUtils.isPrimitiveTypeInfo(arg2type.getListElementTypeInfo())) {
                throw new UDFArgumentTypeException(1,
                    "The second argument `array correctItems` is invalid form: " + typeInfo[1]);
            }

            return new RankingEvaluator();
        }
    }

    public static class ClassificationEvaluator extends GenericUDAFEvaluator {

        private PrimitiveObjectInspector scoreOI;
        private PrimitiveObjectInspector labelOI;

        private StructObjectInspector internalMergeOI;
        private StructField aField;
        private StructField maxScoreField;
        private StructField fpField;
        private StructField tpField;
        private StructField fpPrevField;
        private StructField tpPrevField;
        private StructField partialAreasField;
        private StructField fpCountsField;
        private StructField tpCountsField;
        private StructField fpPrevCountsField;
        private StructField tpPrevCountsField;

        public ClassificationEvaluator() {}

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 2 || parameters.length == 3) : parameters.length;
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.scoreOI = HiveUtils.asDoubleCompatibleOI(parameters[0]);
                this.labelOI = HiveUtils.asIntegerOI(parameters[1]);
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.aField = soi.getStructFieldRef("a");
                this.maxScoreField = soi.getStructFieldRef("maxScore");
                this.fpField = soi.getStructFieldRef("fp");
                this.tpField = soi.getStructFieldRef("tp");
                this.fpPrevField = soi.getStructFieldRef("fpPrev");
                this.tpPrevField = soi.getStructFieldRef("tpPrev");
                this.partialAreasField = soi.getStructFieldRef("partialAreas");
                this.fpCountsField = soi.getStructFieldRef("fpCounts");
                this.tpCountsField = soi.getStructFieldRef("tpCounts");
                this.fpPrevCountsField = soi.getStructFieldRef("fpPrevCounts");
                this.tpPrevCountsField = soi.getStructFieldRef("tpPrevCounts");
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI();
            } else {// terminate
                outputOI = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("a");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
            fieldNames.add("maxScore");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
            fieldNames.add("fp");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("tp");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("fpPrev");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("tpPrev");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);

            MapObjectInspector partialAreasOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
            fieldNames.add("partialAreas");
            fieldOIs.add(partialAreasOI);

            MapObjectInspector fpCountsOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("fpCounts");
            fieldOIs.add(fpCountsOI);

            MapObjectInspector tpCountsOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("tpCounts");
            fieldOIs.add(tpCountsOI);

            MapObjectInspector fpPrevCountsOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("fpPrevCounts");
            fieldOIs.add(fpPrevCountsOI);

            MapObjectInspector tpPrevCountsOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector);
            fieldNames.add("tpPrevCounts");
            fieldOIs.add(tpPrevCountsOI);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        }

        @Override
        public AggregationBuffer getNewAggregationBuffer() throws HiveException {
            AggregationBuffer myAggr = new ClassificationAUCAggregationBuffer();
            reset(myAggr);
            return myAggr;
        }

        @Override
        public void reset(AggregationBuffer agg) throws HiveException {
            ClassificationAUCAggregationBuffer myAggr = (ClassificationAUCAggregationBuffer) agg;
            myAggr.reset();
        }

        @Override
        public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
            ClassificationAUCAggregationBuffer myAggr = (ClassificationAUCAggregationBuffer) agg;

            if (parameters[0] == null) {
                return;
            }
            if (parameters[1] == null) {
                return;
            }

            double score = HiveUtils.getDouble(parameters[0], scoreOI);
            if (score < 0.0d || score > 1.0d) {
                throw new UDFArgumentException("score value MUST be in range [0,1]: " + score);
            }

            int label = PrimitiveObjectInspectorUtils.getInt(parameters[1], labelOI);
            if (label == -1) {
                label = 0;
            } else if (label != 0 && label != 1) {
                throw new UDFArgumentException("label MUST be 0/1 or -1/1: " + label);
            }

            myAggr.iterate(score, label);
        }

        @Override
        public Object terminatePartial(AggregationBuffer agg) throws HiveException {
            ClassificationAUCAggregationBuffer myAggr = (ClassificationAUCAggregationBuffer) agg;

            Object[] partialResult = new Object[11];
            partialResult[0] = new DoubleWritable(myAggr.a);
            partialResult[1] = new DoubleWritable(myAggr.maxScore);
            partialResult[2] = new LongWritable(myAggr.fp);
            partialResult[3] = new LongWritable(myAggr.tp);
            partialResult[4] = new LongWritable(myAggr.fpPrev);
            partialResult[5] = new LongWritable(myAggr.tpPrev);
            partialResult[6] = myAggr.partialAreas;
            partialResult[7] = myAggr.fpCounts;
            partialResult[8] = myAggr.tpCounts;
            partialResult[9] = myAggr.fpPrevCounts;
            partialResult[10] = myAggr.tpPrevCounts;

            return partialResult;
        }

        @Override
        public void merge(AggregationBuffer agg, Object partial) throws HiveException {
            if (partial == null) {
                return;
            }

            Object aObj = internalMergeOI.getStructFieldData(partial, aField);
            Object maxScoreObj = internalMergeOI.getStructFieldData(partial, maxScoreField);
            Object fpObj = internalMergeOI.getStructFieldData(partial, fpField);
            Object tpObj = internalMergeOI.getStructFieldData(partial, tpField);
            Object fpPrevObj = internalMergeOI.getStructFieldData(partial, fpPrevField);
            Object tpPrevObj = internalMergeOI.getStructFieldData(partial, tpPrevField);
            Object partialAreasObj = internalMergeOI.getStructFieldData(partial, partialAreasField);
            Object fpCountsObj = internalMergeOI.getStructFieldData(partial, fpCountsField);
            Object tpCountsObj = internalMergeOI.getStructFieldData(partial, tpCountsField);
            Object fpPrevCountsObj = internalMergeOI.getStructFieldData(partial, fpPrevCountsField);
            Object tpPrevCountsObj = internalMergeOI.getStructFieldData(partial, tpPrevCountsField);

            double a = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(aObj);
            double maxScore = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(maxScoreObj);
            long fp = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(fpObj);
            long tp = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(tpObj);
            long fpPrev = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(fpPrevObj);
            long tpPrev = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(tpPrevObj);

            // [workaround]
            // java.lang.ClassCastException: org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryMap
            if (partialAreasObj instanceof LazyBinaryMap) {
                partialAreasObj = ((LazyBinaryMap) partialAreasObj).getMap();
            }
            Map<Double, Double> partialAreas = (Map<Double, Double>) ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector).getMap(partialAreasObj);

            if (fpCountsObj instanceof LazyBinaryMap) {
                fpCountsObj = ((LazyBinaryMap) fpCountsObj).getMap();
            }
            Map<Double, Long> fpCounts = (Map<Double, Long>) ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector).getMap(fpCountsObj);

            if (tpCountsObj instanceof LazyBinaryMap) {
                tpCountsObj = ((LazyBinaryMap) tpCountsObj).getMap();
            }
            Map<Double, Long> tpCounts = (Map<Double, Long>) ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector).getMap(tpCountsObj);

            if (fpPrevCountsObj instanceof LazyBinaryMap) {
                fpPrevCountsObj = ((LazyBinaryMap) fpPrevCountsObj).getMap();
            }
            Map<Double, Long> fpPrevCounts = (Map<Double, Long>) ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector).getMap(fpPrevCountsObj);

            if (tpPrevCountsObj instanceof LazyBinaryMap) {
                tpPrevCountsObj = ((LazyBinaryMap) tpPrevCountsObj).getMap();
            }
            Map<Double, Long> tpPrevCounts = (Map<Double, Long>) ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector,
                PrimitiveObjectInspectorFactory.writableLongObjectInspector).getMap(tpPrevCountsObj);

            ClassificationAUCAggregationBuffer myAggr = (ClassificationAUCAggregationBuffer) agg;
            myAggr.merge(a, maxScore, fp, tp, fpPrev, tpPrev, partialAreas, fpCounts, tpCounts, fpPrevCounts, tpPrevCounts);
        }

        @Override
        public DoubleWritable terminate(AggregationBuffer agg) throws HiveException {
            ClassificationAUCAggregationBuffer myAggr = (ClassificationAUCAggregationBuffer) agg;
            double result = myAggr.get();
            return new DoubleWritable(result);
        }

    }

    public static class ClassificationAUCAggregationBuffer extends AbstractAggregationBuffer {

        double a, scorePrev, maxScore;
        long fp, tp, fpPrev, tpPrev;
        Map<Double, Double> partialAreas;
        Map<Double, Long> fpCounts, tpCounts, fpPrevCounts, tpPrevCounts;

        public ClassificationAUCAggregationBuffer() {
            super();
        }

        void reset() {
            this.a = 0.d;
            this.scorePrev = Double.POSITIVE_INFINITY;
            this.maxScore = 0.d;
            this.fp = 0;
            this.tp = 0;
            this.fpPrev = 0;
            this.tpPrev = 0;
            this.partialAreas = new HashMap<Double, Double>();
            this.fpCounts = new HashMap<Double, Long>();
            this.tpCounts = new HashMap<Double, Long>();
            this.fpPrevCounts = new HashMap<Double, Long>();
            this.tpPrevCounts = new HashMap<Double, Long>();
        }

        void merge(double o_a, double o_maxScore, long o_fp, long o_tp,
                long o_fpPrev, long o_tpPrev, Map<Double, Double> o_partialAreas,
                Map<Double, Long> o_fpCounts, Map<Double, Long> o_tpCounts,
                Map<Double, Long> o_fpPrevCounts, Map<Double, Long> o_tpPrevCounts) {

            // merge past results
            partialAreas.putAll(o_partialAreas);
            fpCounts.putAll(o_fpCounts);
            tpCounts.putAll(o_tpCounts);
            fpPrevCounts.putAll(o_fpPrevCounts);
            tpPrevCounts.putAll(o_tpPrevCounts);

            // finalize source AUC computation
            o_a += trapezoidArea(o_fp, o_fpPrev, o_tp, o_tpPrev);

            // store source results
            partialAreas.put(o_maxScore, o_a);
            fpCounts.put(o_maxScore, o_fp);
            tpCounts.put(o_maxScore, o_tp);
            fpPrevCounts.put(o_maxScore, o_fpPrev);
            tpPrevCounts.put(o_maxScore, o_tpPrev);
        }

        double get() throws HiveException {
            // store self results
            partialAreas.put(maxScore, a);
            fpCounts.put(maxScore, fp);
            tpCounts.put(maxScore, tp);
            fpPrevCounts.put(maxScore, fpPrev);
            tpPrevCounts.put(maxScore, tpPrev);

            SortedMap<Double, Double> sortedPartialAreas = new TreeMap<Double, Double>(partialAreas);

            // initialize with right-most partial result
            double firstKey = sortedPartialAreas.firstKey();
            double res = sortedPartialAreas.get(firstKey);
            long fpAccum = fpCounts.get(firstKey);
            long tpAccum = tpCounts.get(firstKey);
            long fpPrevAccum = fpPrevCounts.get(firstKey);
            long tpPrevAccum = tpPrevCounts.get(firstKey);
            sortedPartialAreas.remove(firstKey);

            // Merge from right (smaller score) to left (larger score)
            for (Map.Entry<Double, Double> e : sortedPartialAreas.entrySet()) {
                double k = e.getKey();

                // sum up partial area
                res += e.getValue();

                // adjust combined area by adding missing rectangle
                res += trapezoidArea(0, fpAccum, tpCounts.get(k), tpCounts.get(k));

                fpPrevAccum += fpCounts.get(k);
                tpPrevAccum += tpCounts.get(k);

                fpAccum += fpCounts.get(k);
                tpAccum += tpCounts.get(k);
            }

            if (tpAccum == 0 || fpAccum == 0) {
                throw new HiveException(
                    "AUC score is not defined because there is only one class in `label`.");
            }

            // finalize by adding a trapezoid based on the last tp/fp counts
            res += trapezoidArea(fpAccum, fpPrevAccum, tpAccum, tpPrevAccum);

            return res / (tpAccum * fpAccum); // scale
        }

        void iterate(double score, int label) {
            if (score != scorePrev) {
                if (scorePrev == Double.POSITIVE_INFINITY) {
                    // store maximum score for merging
                    maxScore = score;
                }
                a += trapezoidArea(fp, fpPrev, tp, tpPrev); // under (fp, tp)-(fpPrev, tpPrev)
                scorePrev = score;
                fpPrev = fp;
                tpPrev = tp;
            }
            if (label == 1) {
                tp++; // this finally will be the number of positive samples
            } else {
                fp++; // this finally will be the number of negative samples
            }
        }

        private double trapezoidArea(double x1, double x2, double y1, double y2) {
            double base = Math.abs(x1 - x2);
            double height = (y1 + y2) / 2.d;
            return base * height;
        }
    }

    public static class RankingEvaluator extends GenericUDAFEvaluator {

        private ListObjectInspector recommendListOI;
        private ListObjectInspector truthListOI;
        private WritableIntObjectInspector recommendSizeOI;

        private StructObjectInspector internalMergeOI;
        private StructField countField;
        private StructField sumField;

        public RankingEvaluator() {}

        @Override
        public ObjectInspector init(Mode mode, ObjectInspector[] parameters) throws HiveException {
            assert (parameters.length == 2 || parameters.length == 3) : parameters.length;
            super.init(mode, parameters);

            // initialize input
            if (mode == Mode.PARTIAL1 || mode == Mode.COMPLETE) {// from original data
                this.recommendListOI = (ListObjectInspector) parameters[0];
                this.truthListOI = (ListObjectInspector) parameters[1];
                if (parameters.length == 3) {
                    this.recommendSizeOI = (WritableIntObjectInspector) parameters[2];
                }
            } else {// from partial aggregation
                StructObjectInspector soi = (StructObjectInspector) parameters[0];
                this.internalMergeOI = soi;
                this.countField = soi.getStructFieldRef("count");
                this.sumField = soi.getStructFieldRef("sum");
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI();
            } else {// terminate
                outputOI = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("sum");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
            fieldNames.add("count");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableLongObjectInspector);

            return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
        }

        @Override
        public AggregationBuffer getNewAggregationBuffer() throws HiveException {
            AggregationBuffer myAggr = new RankingAUCAggregationBuffer();
            reset(myAggr);
            return myAggr;
        }

        @Override
        public void reset(AggregationBuffer agg) throws HiveException {
            RankingAUCAggregationBuffer myAggr = (RankingAUCAggregationBuffer) agg;
            myAggr.reset();
        }

        @Override
        public void iterate(AggregationBuffer agg, Object[] parameters) throws HiveException {
            RankingAUCAggregationBuffer myAggr = (RankingAUCAggregationBuffer) agg;

            List<?> recommendList = recommendListOI.getList(parameters[0]);
            if (recommendList == null) {
                recommendList = Collections.emptyList();
            }
            List<?> truthList = truthListOI.getList(parameters[1]);
            if (truthList == null) {
                return;
            }

            int recommendSize = recommendList.size();
            if (parameters.length == 3) {
                recommendSize = recommendSizeOI.get(parameters[2]);
            }
            if (recommendSize < 0 || recommendSize > recommendList.size()) {
                throw new UDFArgumentException(
                    "The third argument `int recommendSize` must be in [0, " + recommendList.size()
                            + "]");
            }

            myAggr.iterate(recommendList, truthList, recommendSize);
        }

        @Override
        public Object terminatePartial(AggregationBuffer agg) throws HiveException {
            RankingAUCAggregationBuffer myAggr = (RankingAUCAggregationBuffer) agg;

            Object[] partialResult = new Object[2];
            partialResult[0] = new DoubleWritable(myAggr.sum);
            partialResult[1] = new LongWritable(myAggr.count);
            return partialResult;
        }

        @Override
        public void merge(AggregationBuffer agg, Object partial) throws HiveException {
            if (partial == null) {
                return;
            }

            Object sumObj = internalMergeOI.getStructFieldData(partial, sumField);
            Object countObj = internalMergeOI.getStructFieldData(partial, countField);
            double sum = PrimitiveObjectInspectorFactory.writableDoubleObjectInspector.get(sumObj);
            long count = PrimitiveObjectInspectorFactory.writableLongObjectInspector.get(countObj);

            RankingAUCAggregationBuffer myAggr = (RankingAUCAggregationBuffer) agg;
            myAggr.merge(sum, count);
        }

        @Override
        public DoubleWritable terminate(AggregationBuffer agg) throws HiveException {
            RankingAUCAggregationBuffer myAggr = (RankingAUCAggregationBuffer) agg;
            double result = myAggr.get();
            return new DoubleWritable(result);
        }

    }

    public static class RankingAUCAggregationBuffer extends AbstractAggregationBuffer {

        double sum;
        long count;

        public RankingAUCAggregationBuffer() {
            super();
        }

        void reset() {
            this.sum = 0.d;
            this.count = 0;
        }

        void merge(double o_sum, long o_count) {
            sum += o_sum;
            count += o_count;
        }

        double get() {
            if (count == 0) {
                return 0.d;
            }
            return sum / count;
        }

        void iterate(@Nonnull List<?> recommendList, @Nonnull List<?> truthList,
                @Nonnull int recommendSize) {
            sum += BinaryResponsesMeasures.AUC(recommendList, truthList, recommendSize);
            count++;
        }
    }

}
