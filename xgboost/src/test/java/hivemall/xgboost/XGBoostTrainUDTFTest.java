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
package hivemall.xgboost;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;
import hivemall.TestUtils;
import hivemall.utils.collections.lists.FloatArrayList;
import hivemall.utils.io.IOUtils;
import hivemall.utils.lang.PrivilegedAccessor;
import hivemall.utils.lang.mutable.MutableObject;
import hivemall.xgboost.utils.DMatrixBuilder;
import hivemall.xgboost.utils.SparseDMatrixBuilder;
import hivemall.xgboost.utils.XGBoostUtils;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;
import org.junit.experimental.theories.DataPoints;
import org.junit.experimental.theories.FromDataPoints;
import org.junit.experimental.theories.Theories;
import org.junit.experimental.theories.Theory;
import org.junit.runner.RunWith;

@RunWith(Theories.class)
public class XGBoostTrainUDTFTest {

    @Test
    public void testSerialization() throws HiveException {
        TestUtils.testGenericUDTFSerialization(XGBoostTrainUDTF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector},
            new Object[][] {{Arrays.asList("1:-2", "2:-1"), 0.d}});
    }

    @DataPoints("trial")
    public static final List<TestParameter> trial = TestParameter.merge(
        new TestParameter.Builder().trainDataset(
            "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train")
                                   .testDataset(
                                       "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.test")
                                   .metric(new ClassificationError(0.1f))
                                   .hyperParams(new String[] {
                                           "-objective binary:logistic -iters 10",
                                           "-objective binary:logistic -iters 10 -num_early_stopping_rounds 3"}));


    static class TestParameter {
        @Nonnull
        private final String trainDatasetUrl, testDatasetUrl;
        @Nonnull
        private final EvalMetric evalMetric;
        @Nonnull
        private final String hyperParams;

        public TestParameter(String trainDatasetUrl, String testDatasetUrl, EvalMetric evalMetric,
                String hyperParams) {
            this.trainDatasetUrl = Objects.requireNonNull(trainDatasetUrl);
            this.testDatasetUrl = Objects.requireNonNull(testDatasetUrl);
            this.evalMetric = Objects.requireNonNull(evalMetric);
            this.hyperParams = Objects.requireNonNull(hyperParams);
        }

        static class Builder {
            private String trainDatasetUrl, testDatasetUrl;
            private EvalMetric evalMetric;
            private String[] hyperParams;

            Builder trainDataset(String trainUrl) {
                this.trainDatasetUrl = trainUrl;
                return this;
            }

            Builder testDataset(String testUrl) {
                this.testDatasetUrl = testUrl;
                return this;
            }

            Builder metric(EvalMetric metric) {
                this.evalMetric = metric;
                return this;
            }

            Builder hyperParams(String[] hyperParams) {
                this.hyperParams = hyperParams;
                return this;
            }

            List<TestParameter> build() {
                List<TestParameter> result = new ArrayList<>();
                for (String p : hyperParams) {
                    result.add(new TestParameter(trainDatasetUrl, testDatasetUrl, evalMetric, p));
                }
                return result;
            }
        }

        static List<TestParameter> merge(Builder... builders) {
            List<TestParameter> result = new ArrayList<>();
            for (Builder builder : builders) {
                result.addAll(builder.build());
            }
            return result;
        }
    }

    public interface EvalMetric {

        void next(double[] predicted, float expected);

        float result();

        void assertExpected();
    }

    static class ClassificationError implements EvalMetric {

        final float acceptedErrorRate;
        int total = 0, errors = 0;

        public ClassificationError(float acceptedErrorRate) {
            this.acceptedErrorRate = acceptedErrorRate;
        }

        @Override
        public void next(double[] predicted, float expected) {
            Assert.assertEquals(1, predicted.length);
            int expectedLabel = (expected > 0) ? 1 : 0;
            double prob = predicted[0];
            Assert.assertTrue(prob >= 0);
            Assert.assertTrue(prob <= 1.0);
            int actuallabel = (prob > 0.5f) ? 1 : 0;
            if (expectedLabel != actuallabel) {
                errors++;
            }
            total++;
        }

        @Override
        public float result() {
            return errors / (float) total;
        }

        @Override
        public void assertExpected() {
            float errorRate = result();
            Assert.assertTrue(String.format(
                "classification error rate expected to be less than or equals to %f but %f",
                acceptedErrorRate, errorRate), errorRate <= acceptedErrorRate);
        }

    }


    @Theory
    public void testHyperParams(@FromDataPoints("trial") final TestParameter trial)
            throws HiveException, IOException, XGBoostError {
        final String trainDataUrl = trial.trainDatasetUrl;
        final String testDataUrl = trial.testDatasetUrl;
        final DMatrix testMatrix = loadDMatrix(testDataUrl);
        final float[] testLabels = testMatrix.getLabel();
        final EvalMetric metric = trial.evalMetric;

        final MutableObject<float[][]> expectedPredictData = new MutableObject<>();
        final XGBoostTrainUDTF udtf = new XGBoostTrainUDTF() {
            @Override
            protected void onFinishTraining(Booster booster) {
                final float[][] result;
                try {
                    result = booster.predict(testMatrix);
                } catch (XGBoostError e) {
                    throw new RuntimeException(e);
                }
                expectedPredictData.set(result);
            }
        };
        udtf.initialize(
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        trial.hyperParams)});

        for (Object[] row : loadDataset(trainDataUrl)) {
            udtf.process(row);
        }

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object input) throws HiveException {
                final float[][] expecteds = expectedPredictData.get();

                Object[] forwardedObj = (Object[]) input;
                String modelId = (String) forwardedObj[0];
                Assert.assertNotNull(modelId);
                Text modelStr = (Text) forwardedObj[1];
                Booster booster = XGBoostUtils.deserializeBooster(modelStr);
                try {
                    float[][] actuals = booster.predict(testMatrix);
                    Assert.assertEquals(expecteds.length, actuals.length);
                    for (int i = 0; i > expecteds.length; i++) {
                        Assert.assertArrayEquals(expecteds[i], actuals[i], 1e-5f);
                    }
                } catch (XGBoostError e) {
                    throw new HiveException(e);
                } finally {
                    XGBoostUtils.close(booster);
                }

                Predictor predictor = XGBoostUtils.loadPredictor(modelStr);
                final String gbmName, objName;
                try {
                    gbmName = (String) PrivilegedAccessor.getValue(predictor, "name_gbm");
                    objName = (String) PrivilegedAccessor.getValue(predictor, "name_obj");
                } catch (Exception e) {
                    throw new HiveException(e);
                }
                Assert.assertEquals(udtf.params.get("booster"), gbmName);
                Assert.assertEquals(udtf.params.get("objective"), objName);

                final List<FVec> fvList;
                try {
                    fvList = loadDatasetFVec(testDataUrl);
                } catch (IOException e) {
                    throw new HiveException(e);
                }
                Assert.assertEquals(expecteds.length, fvList.size());
                for (int i = 0; i < expecteds.length; i++) {
                    float[] expected = expecteds[i];
                    FVec fv = fvList.get(i);
                    double[] actual = predictor.predict(fv);
                    Assert.assertEquals(expected.length, actual.length);
                    for (int j = 0; j < expected.length; j++) {
                        Assert.assertEquals(expected[j], actual[j], 1e-5d);
                    }
                    metric.next(actual, testLabels[i]);
                }
            }
        });
        udtf.close();
        testMatrix.dispose();

        metric.assertExpected();
    }

    @Nonnull
    private static List<Object[]> loadDataset(@Nonnull String urlString) throws IOException {
        final List<Object[]> dataset = new ArrayList<>();

        URL url = new URL(urlString);
        BufferedReader reader = IOUtils.bufferedReader(url.openStream());

        String line = reader.readLine();
        while (line != null) {
            String[] splitted = line.split(" ");

            Object[] row = new Object[2];
            row[0] = Arrays.asList(Arrays.copyOfRange(splitted, 1, splitted.length));
            row[1] = Double.parseDouble(splitted[0]);
            dataset.add(row);

            line = reader.readLine();
        }

        return dataset;
    }

    @Nonnull
    private static DMatrix loadDMatrix(@Nonnull String urlString) throws IOException, XGBoostError {
        final DMatrixBuilder builder = new SparseDMatrixBuilder(1024, false);
        final FloatArrayList labels = new FloatArrayList(1024);

        URL url = new URL(urlString);
        BufferedReader reader = IOUtils.bufferedReader(url.openStream());

        String line = reader.readLine();
        while (line != null) {
            String[] splitted = line.split(" ");

            float label = Float.parseFloat(splitted[0]);
            labels.add(label);
            builder.nextRow(splitted, 1, splitted.length);

            line = reader.readLine();
        }

        return builder.buildMatrix(labels.toArray());
    }

    @Nonnull
    private static List<FVec> loadDatasetFVec(@Nonnull String urlString) throws IOException {
        final List<FVec> dataset = new ArrayList<>();

        URL url = new URL(urlString);
        BufferedReader reader = IOUtils.bufferedReader(url.openStream());

        String line = reader.readLine();
        while (line != null) {
            String[] splitted = line.split(" ");

            FVec fv = parseFVec(splitted, 1, splitted.length);
            dataset.add(fv);

            line = reader.readLine();
        }

        return dataset;
    }

    @Nonnull
    private static FVec parseFVec(@Nonnull final String[] row, final int start, final int end) {
        final Map<Integer, Float> map = new HashMap<>((int) (row.length * 1.5));
        for (int i = start; i < end; i++) {
            String f = row[i];
            if (f == null) {
                continue;
            }
            String str = f.toString();
            final int pos = str.indexOf(':');
            if (pos < 1) {
                throw new IllegalArgumentException("Invalid feature format: " + str);
            }
            final int index;
            final float value;
            try {
                index = Integer.parseInt(str.substring(0, pos));
                value = Float.parseFloat(str.substring(pos + 1));
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Failed to parse a feature value: " + str);
            }
            map.put(index, value);
        }

        return FVec.Transformer.fromMap(map);
    }
}
