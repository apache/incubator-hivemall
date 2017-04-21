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

import static org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory.javaDoubleObjectInspector;
import static org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory.javaLongObjectInspector;
import static org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
import static org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory.writableLongObjectInspector;
import hivemall.utils.hadoop.HiveUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StandardMapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
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
        private StructField indexScoreField;
        private StructField areaField;
        private StructField fpField;
        private StructField tpField;
        private StructField fpPrevField;
        private StructField tpPrevField;
        private StructField areaPartialMapField;
        private StructField fpPartialMapField;
        private StructField tpPartialMapField;
        private StructField fpPrevPartialMapField;
        private StructField tpPrevPartialMapField;

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
                this.indexScoreField = soi.getStructFieldRef("indexScore");
                this.areaField = soi.getStructFieldRef("area");
                this.fpField = soi.getStructFieldRef("fp");
                this.tpField = soi.getStructFieldRef("tp");
                this.fpPrevField = soi.getStructFieldRef("fpPrev");
                this.tpPrevField = soi.getStructFieldRef("tpPrev");
                this.areaPartialMapField = soi.getStructFieldRef("areaPartialMap");
                this.fpPartialMapField = soi.getStructFieldRef("fpPartialMap");
                this.tpPartialMapField = soi.getStructFieldRef("tpPartialMap");
                this.fpPrevPartialMapField = soi.getStructFieldRef("fpPrevPartialMap");
                this.tpPrevPartialMapField = soi.getStructFieldRef("tpPrevPartialMap");
            }

            // initialize output
            final ObjectInspector outputOI;
            if (mode == Mode.PARTIAL1 || mode == Mode.PARTIAL2) {// terminatePartial
                outputOI = internalMergeOI();
            } else {// terminate
                outputOI = writableDoubleObjectInspector;
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("indexScore");
            fieldOIs.add(writableDoubleObjectInspector);
            fieldNames.add("area");
            fieldOIs.add(writableDoubleObjectInspector);
            fieldNames.add("fp");
            fieldOIs.add(writableLongObjectInspector);
            fieldNames.add("tp");
            fieldOIs.add(writableLongObjectInspector);
            fieldNames.add("fpPrev");
            fieldOIs.add(writableLongObjectInspector);
            fieldNames.add("tpPrev");
            fieldOIs.add(writableLongObjectInspector);

            MapObjectInspector areaPartialMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                javaDoubleObjectInspector, javaDoubleObjectInspector);
            fieldNames.add("areaPartialMap");
            fieldOIs.add(areaPartialMapOI);

            MapObjectInspector fpPartialMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                javaDoubleObjectInspector, javaLongObjectInspector);
            fieldNames.add("fpPartialMap");
            fieldOIs.add(fpPartialMapOI);

            MapObjectInspector tpPartialMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                javaDoubleObjectInspector, javaLongObjectInspector);
            fieldNames.add("tpPartialMap");
            fieldOIs.add(tpPartialMapOI);

            MapObjectInspector fpPrevPartialMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                javaDoubleObjectInspector, javaLongObjectInspector);
            fieldNames.add("fpPrevPartialMap");
            fieldOIs.add(fpPrevPartialMapOI);

            MapObjectInspector tpPrevPartialMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                javaDoubleObjectInspector, javaLongObjectInspector);
            fieldNames.add("tpPrevPartialMap");
            fieldOIs.add(tpPrevPartialMapOI);

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
            partialResult[0] = new DoubleWritable(myAggr.indexScore);
            partialResult[1] = new DoubleWritable(myAggr.area);
            partialResult[2] = new LongWritable(myAggr.fp);
            partialResult[3] = new LongWritable(myAggr.tp);
            partialResult[4] = new LongWritable(myAggr.fpPrev);
            partialResult[5] = new LongWritable(myAggr.tpPrev);
            partialResult[6] = myAggr.areaPartialMap;
            partialResult[7] = myAggr.fpPartialMap;
            partialResult[8] = myAggr.tpPartialMap;
            partialResult[9] = myAggr.fpPrevPartialMap;
            partialResult[10] = myAggr.tpPrevPartialMap;

            return partialResult;
        }

        @SuppressWarnings("unchecked")
        @Override
        public void merge(AggregationBuffer agg, Object partial) throws HiveException {
            if (partial == null) {
                return;
            }

            Object indexScoreObj = internalMergeOI.getStructFieldData(partial, indexScoreField);
            Object areaObj = internalMergeOI.getStructFieldData(partial, areaField);
            Object fpObj = internalMergeOI.getStructFieldData(partial, fpField);
            Object tpObj = internalMergeOI.getStructFieldData(partial, tpField);
            Object fpPrevObj = internalMergeOI.getStructFieldData(partial, fpPrevField);
            Object tpPrevObj = internalMergeOI.getStructFieldData(partial, tpPrevField);
            Object areaPartialMapObj = internalMergeOI.getStructFieldData(partial,
                areaPartialMapField);
            Object fpPartialMapObj = internalMergeOI.getStructFieldData(partial, fpPartialMapField);
            Object tpPartialMapObj = internalMergeOI.getStructFieldData(partial, tpPartialMapField);
            Object fpPrevPartialMapObj = internalMergeOI.getStructFieldData(partial,
                fpPrevPartialMapField);
            Object tpPrevPartialMapObj = internalMergeOI.getStructFieldData(partial,
                tpPrevPartialMapField);

            double indexScore = writableDoubleObjectInspector.get(indexScoreObj);
            double area = writableDoubleObjectInspector.get(areaObj);
            long fp = writableLongObjectInspector.get(fpObj);
            long tp = writableLongObjectInspector.get(tpObj);
            long fpPrev = writableLongObjectInspector.get(fpPrevObj);
            long tpPrev = writableLongObjectInspector.get(tpPrevObj);

            StandardMapObjectInspector ddMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                javaDoubleObjectInspector, javaDoubleObjectInspector);
            StandardMapObjectInspector dlMapOI = ObjectInspectorFactory.getStandardMapObjectInspector(
                javaDoubleObjectInspector, javaLongObjectInspector);

            Map<Double, Double> areaPartialMap = (Map<Double, Double>) ddMapOI.getMap(HiveUtils.castLazyBinaryObject(areaPartialMapObj));
            Map<Double, Long> fpPartialMap = (Map<Double, Long>) dlMapOI.getMap(HiveUtils.castLazyBinaryObject(fpPartialMapObj));
            Map<Double, Long> tpPartialMap = (Map<Double, Long>) dlMapOI.getMap(HiveUtils.castLazyBinaryObject(tpPartialMapObj));
            Map<Double, Long> fpPrevPartialMap = (Map<Double, Long>) dlMapOI.getMap(HiveUtils.castLazyBinaryObject(fpPrevPartialMapObj));
            Map<Double, Long> tpPrevPartialMap = (Map<Double, Long>) dlMapOI.getMap(HiveUtils.castLazyBinaryObject(tpPrevPartialMapObj));

            ClassificationAUCAggregationBuffer myAggr = (ClassificationAUCAggregationBuffer) agg;
            myAggr.merge(indexScore, area, fp, tp, fpPrev, tpPrev, areaPartialMap, fpPartialMap,
                tpPartialMap, fpPrevPartialMap, tpPrevPartialMap);
        }

        @Override
        public DoubleWritable terminate(AggregationBuffer agg) throws HiveException {
            ClassificationAUCAggregationBuffer myAggr = (ClassificationAUCAggregationBuffer) agg;
            double result = myAggr.get();
            return new DoubleWritable(result);
        }

    }

    public static class ClassificationAUCAggregationBuffer extends AbstractAggregationBuffer {

        double area, scorePrev, indexScore;
        long fp, tp, fpPrev, tpPrev;
        Map<Double, Double> areaPartialMap;
        Map<Double, Long> fpPartialMap, tpPartialMap, fpPrevPartialMap, tpPrevPartialMap;

        public ClassificationAUCAggregationBuffer() {
            super();
        }

        void reset() {
            this.area = 0.d;
            this.scorePrev = Double.POSITIVE_INFINITY;
            this.indexScore = 0.d;
            this.fp = 0;
            this.tp = 0;
            this.fpPrev = 0;
            this.tpPrev = 0;
            this.areaPartialMap = new HashMap<Double, Double>();
            this.fpPartialMap = new HashMap<Double, Long>();
            this.tpPartialMap = new HashMap<Double, Long>();
            this.fpPrevPartialMap = new HashMap<Double, Long>();
            this.tpPrevPartialMap = new HashMap<Double, Long>();
        }

        void merge(double o_indexScore, double o_area, long o_fp, long o_tp, long o_fpPrev,
                long o_tpPrev, Map<Double, Double> o_areaPartialMap,
                Map<Double, Long> o_fpPartialMap, Map<Double, Long> o_tpPartialMap,
                Map<Double, Long> o_fpPrevPartialMap, Map<Double, Long> o_tpPrevPartialMap) {

            // merge past partial results
            areaPartialMap.putAll(o_areaPartialMap);
            fpPartialMap.putAll(o_fpPartialMap);
            tpPartialMap.putAll(o_tpPartialMap);
            fpPrevPartialMap.putAll(o_fpPrevPartialMap);
            tpPrevPartialMap.putAll(o_tpPrevPartialMap);

            // finalize source AUC computation
            o_area += trapezoidArea(o_fp, o_fpPrev, o_tp, o_tpPrev);

            // store source results
            areaPartialMap.put(o_indexScore, o_area);
            fpPartialMap.put(o_indexScore, o_fp);
            tpPartialMap.put(o_indexScore, o_tp);
            fpPrevPartialMap.put(o_indexScore, o_fpPrev);
            tpPrevPartialMap.put(o_indexScore, o_tpPrev);
        }

        double get() throws HiveException {
            // store self results
            areaPartialMap.put(indexScore, area);
            fpPartialMap.put(indexScore, fp);
            tpPartialMap.put(indexScore, tp);
            fpPrevPartialMap.put(indexScore, fpPrev);
            tpPrevPartialMap.put(indexScore, tpPrev);

            SortedMap<Double, Double> areaPartialSortedMap = new TreeMap<Double, Double>(
                Collections.reverseOrder());
            areaPartialSortedMap.putAll(areaPartialMap);

            // initialize with leftmost partial result
            double firstKey = areaPartialSortedMap.firstKey();
            double res = areaPartialSortedMap.get(firstKey);
            long fpAccum = fpPartialMap.get(firstKey);
            long tpAccum = tpPartialMap.get(firstKey);
            long fpPrevAccum = fpPrevPartialMap.get(firstKey);
            long tpPrevAccum = tpPrevPartialMap.get(firstKey);

            // Merge from left (larger score) to right (smaller score)
            for (double k : areaPartialSortedMap.keySet()) {
                if (k == firstKey) { // variables are already initialized with the leftmost partial result
                    continue;
                }

                // sum up partial area
                res += areaPartialSortedMap.get(k);

                // adjust combined area by adding missing rectangle
                res += trapezoidArea(0, fpPartialMap.get(k), tpAccum, tpAccum);

                // sum up (prev) TP/FP count
                fpPrevAccum = fpAccum + fpPrevPartialMap.get(k);
                tpPrevAccum = tpAccum + tpPrevPartialMap.get(k);
                fpAccum = fpAccum + fpPartialMap.get(k);
                tpAccum = tpAccum + tpPartialMap.get(k);
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
                    // store maximum score as an index
                    indexScore = score;
                }
                area += trapezoidArea(fp, fpPrev, tp, tpPrev); // under (fp, tp)-(fpPrev, tpPrev)
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
                outputOI = writableDoubleObjectInspector;
            }
            return outputOI;
        }

        private static StructObjectInspector internalMergeOI() {
            ArrayList<String> fieldNames = new ArrayList<String>();
            ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();

            fieldNames.add("sum");
            fieldOIs.add(writableDoubleObjectInspector);
            fieldNames.add("count");
            fieldOIs.add(writableLongObjectInspector);

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
            double sum = writableDoubleObjectInspector.get(sumObj);
            long count = writableLongObjectInspector.get(countObj);

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
