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
package hivemall;

import biz.k11i.xgboost.util.FVec;
import hivemall.utils.collections.lists.FloatArrayList;
import hivemall.utils.io.IOUtils;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.math.MathUtils;
import hivemall.xgboost.utils.DMatrixBuilder;
import hivemall.xgboost.utils.DenseDMatrixBuilder;
import hivemall.xgboost.utils.SparseDMatrixBuilder;
import hivemall.xgboost.utils.XGBoostUtils;
import ml.dmlc.xgboost4j.java.DMatrix;

import java.io.BufferedReader;
import java.lang.reflect.Array;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Random;

import javax.annotation.Nonnull;

import org.junit.Assert;

public abstract class TestBase {

    public static class TestParameter {
        @Nonnull
        public final Dataset trainDataset, testDataset;
        @Nonnull
        public final EvalMetric evalMetric;
        @Nonnull
        public final String hyperParams;

        public TestParameter(Dataset trainDataset, Dataset testDataset, EvalMetric evalMetric,
                String hyperParams) {
            this.trainDataset = Objects.requireNonNull(trainDataset);
            this.testDataset = Objects.requireNonNull(testDataset);
            this.evalMetric = Objects.requireNonNull(evalMetric);
            this.hyperParams = Objects.requireNonNull(hyperParams);
        }

        public static class Builder {
            private Dataset trainDataset, testDatase;
            private EvalMetric evalMetric;
            private String[] hyperParams;

            public Builder trainDataset(Dataset trainDataset) {
                this.trainDataset = trainDataset;
                return this;
            }

            public Builder testDatase(Dataset testDatase) {
                this.testDatase = testDatase;
                return this;
            }

            public Builder metric(EvalMetric metric) {
                this.evalMetric = metric;
                return this;
            }

            public Builder hyperParams(String[] hyperParams) {
                this.hyperParams = hyperParams;
                return this;
            }

            public List<TestParameter> build() {
                List<TestParameter> result = new ArrayList<>();
                for (String p : hyperParams) {
                    result.add(new TestParameter(trainDataset, testDatase, evalMetric, p));
                }
                return result;
            }
        }

        @Nonnull
        public static List<TestParameter> merge(@Nonnull Builder... builders) {
            List<TestParameter> result = new ArrayList<>();
            for (Builder builder : builders) {
                result.addAll(builder.build());
            }
            return result;
        }

        @Override
        public String toString() {
            return "TestParameter [trainDataset=" + trainDataset + ", testDataset=" + testDataset
                    + ", evalMetric=" + evalMetric + ", hyperParams=" + hyperParams + "]";
        }
    }

    public interface EvalMetric {

        void next(double[] predicted, float expected);

        float result();

        void assertExpected();
    }

    public static class ClassificationError implements EvalMetric {

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

    public static class MultiClassClassificationError implements EvalMetric {
        final float acceptedErrorRate;
        int total = 0, errors = 0;

        public MultiClassClassificationError(float acceptedErrorRate) {
            this.acceptedErrorRate = acceptedErrorRate;
        }

        @Override
        public void next(double[] predicted, float expected) {
            final int actualLabel;
            if (predicted.length == 1) {//-objective multi:softmax
                actualLabel = (int) predicted[0];
            } else {// -objective multi:softprob
                actualLabel = ArrayUtils.argmax(predicted);
            }

            int expectedLabel = (int) expected;
            if (expectedLabel != actualLabel) {
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

    public static class MAE implements EvalMetric {
        final float ensureLessThan;

        private double diffSum = 0.d;
        private int count = 0;

        public MAE(float ensureLessThan) {
            this.ensureLessThan = ensureLessThan;
        }

        @Override
        public void next(double[] predicted, float expected) {
            Assert.assertEquals(1, predicted.length);

            this.diffSum += Math.abs(predicted[0] - expected);
            this.count += 1;
        }

        @Override
        public float result() {
            return (float) (diffSum / count);
        }

        @Override
        public void assertExpected() {
            float mae = result();
            Assert.assertTrue(
                String.format("MAE expected to be less than %f but %f", ensureLessThan, mae),
                mae < ensureLessThan);
        }


    }

    public static abstract class Dataset {

        @Nonnull
        private final String datasetUrl;
        @Nonnull
        private final String separator;

        public Dataset(@Nonnull String datasetUrl, @Nonnull String separator) {
            this.datasetUrl = datasetUrl;
            this.separator = separator;
        }

        @Nonnull
        public String getDatasetUrl() {
            return datasetUrl;
        }

        public abstract boolean isSparseDataset();

        public void parse(@Nonnull RowProcessor proc) throws Exception {
            URL url = new URL(datasetUrl);
            try (BufferedReader reader = IOUtils.bufferedReader(url.openStream())) {
                String line = reader.readLine();
                while (line != null) {
                    String[] splitted = line.split(separator);
                    proc.handleRow(splitted);
                    line = reader.readLine();
                }
            }
        }

        public abstract List<Object[]> loadDatasetAsListOfObjects() throws Exception;

        public abstract DMatrix loadDatasetAsDMatrix() throws Exception;

        public abstract List<FVec> loadDatasetAsListOfFVec() throws Exception;

    }

    public interface RowProcessor {
        void handleRow(String[] splitted) throws Exception;
    }

    public static class LibsvmDataset extends Dataset {

        public LibsvmDataset(@Nonnull String datasetUrl) {
            super(datasetUrl, " ");
        }

        @Override
        public boolean isSparseDataset() {
            return true;
        }

        @Override
        public List<Object[]> loadDatasetAsListOfObjects() throws Exception {
            final List<Object[]> dataset = new ArrayList<>();

            RowProcessor proc = new RowProcessor() {
                @Override
                public void handleRow(String[] splitted) throws Exception {
                    Object[] row = new Object[2];
                    row[0] = Arrays.asList(Arrays.copyOfRange(splitted, 1, splitted.length));
                    row[1] = Double.parseDouble(splitted[0]);
                    dataset.add(row);
                }

            };
            parse(proc);

            return dataset;
        }

        @Override
        public DMatrix loadDatasetAsDMatrix() throws Exception {
            final DMatrixBuilder builder = new SparseDMatrixBuilder(1024, false);
            final FloatArrayList labels = new FloatArrayList(1024);

            RowProcessor proc = new RowProcessor() {
                @Override
                public void handleRow(String[] splitted) throws Exception {
                    float label = Float.parseFloat(splitted[0]);
                    labels.add(label);
                    builder.nextRow(splitted, 1, splitted.length);
                }
            };
            parse(proc);

            return builder.buildMatrix(labels.toArray());
        }

        @Override
        public List<FVec> loadDatasetAsListOfFVec() throws Exception {
            final List<FVec> dataset = new ArrayList<>();

            RowProcessor proc = new RowProcessor() {
                @Override
                public void handleRow(String[] splitted) throws Exception {
                    FVec fv = XGBoostUtils.parseRowAsFVec(splitted, 1, splitted.length);
                    dataset.add(fv);
                }

            };
            parse(proc);

            return dataset;
        }
    }

    public static class DermatologyDataset extends Dataset {
        // wc -l dermatology.data > 366
        private static final int NUM_ROWS = 366;

        @Nonnull
        private final int[] sliceIndex;

        public DermatologyDataset(boolean train, double trainFrac) {
            super("https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data", ",");
            int[] index = ArrayUtils.shuffle(MathUtils.permutation(NUM_ROWS), new Random(43L));
            int numTrain = (int) (NUM_ROWS * trainFrac);
            if (train) {
                this.sliceIndex = Arrays.copyOfRange(index, 0, numTrain);
            } else {
                this.sliceIndex = Arrays.copyOfRange(index, numTrain, index.length);
            }
        }

        @Override
        public boolean isSparseDataset() {
            return false;
        }

        @Override
        public List<Object[]> loadDatasetAsListOfObjects() throws Exception {
            final List<Object[]> dataset = new ArrayList<>();

            RowProcessor proc = new RowProcessor() {
                @Override
                public void handleRow(String[] splitted) throws Exception {
                    final Float[] features = new Float[34];
                    for (int i = 0; i <= 32; i++) {
                        features[i] = Float.parseFloat(splitted[i]);
                    }
                    features[33] = splitted[33].equals("?") ? 0.f : Float.parseFloat(splitted[33]);
                    int label = Integer.parseInt(splitted[34]) - 1;
                    Object[] row = new Object[2];
                    row[0] = Arrays.asList(features);
                    row[1] = Double.valueOf(label);
                    dataset.add(row);
                }

            };
            parse(proc);

            return slice(dataset, sliceIndex, Object[].class);
        }

        @Override
        public DMatrix loadDatasetAsDMatrix() throws Exception {
            final DMatrixBuilder builder = new DenseDMatrixBuilder(1024);
            final FloatArrayList labels = new FloatArrayList(1024);

            RowProcessor proc = new RowProcessor() {
                @Override
                public void handleRow(String[] splitted) throws Exception {
                    final float[] features = new float[34];
                    for (int i = 0; i <= 32; i++) {
                        features[i] = Float.parseFloat(splitted[i]);
                    }
                    features[33] = splitted[33].equals("?") ? 0.f : Float.parseFloat(splitted[33]);
                    int label = Integer.parseInt(splitted[34]) - 1;

                    labels.add(label);
                    builder.nextRow(features);
                }
            };
            parse(proc);

            return builder.buildMatrix(labels.toArray()).slice(sliceIndex);
        }

        @Override
        public List<FVec> loadDatasetAsListOfFVec() throws Exception {
            final List<FVec> dataset = new ArrayList<>();

            RowProcessor proc = new RowProcessor() {
                @Override
                public void handleRow(String[] splitted) throws Exception {
                    final float[] features = new float[34];
                    for (int i = 0; i <= 32; i++) {
                        features[i] = Float.parseFloat(splitted[i]);
                    }
                    features[33] = splitted[33].equals("?") ? 0.f : Float.parseFloat(splitted[33]);

                    FVec fv = FVec.Transformer.fromArray(features, false);
                    dataset.add(fv);
                }

            };
            parse(proc);

            return slice(dataset, sliceIndex, FVec.class);
        }

    }

    @SuppressWarnings("unchecked")
    @Nonnull
    private static <T> List<T> slice(@Nonnull List<T> original, @Nonnull int[] sliceIndex,
            @Nonnull Class<?> elemType) {
        final T[] sliced = (T[]) Array.newInstance(elemType, sliceIndex.length);
        for (int i = 0; i < sliceIndex.length; i++) {
            sliced[i] = original.get(sliceIndex[i]);
        }
        return Arrays.asList(sliced);
    }

}
