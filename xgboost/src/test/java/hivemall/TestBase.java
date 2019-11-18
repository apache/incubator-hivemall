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
import hivemall.xgboost.utils.DMatrixBuilder;
import hivemall.xgboost.utils.SparseDMatrixBuilder;
import hivemall.xgboost.utils.XGBoostUtils;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import javax.annotation.Nonnull;

import org.junit.Assert;

public abstract class TestBase {


    public static class TestParameter {
        @Nonnull
        public final Dataset dataset;
        @Nonnull
        public final EvalMetric evalMetric;
        @Nonnull
        public final String hyperParams;

        public TestParameter(Dataset dataset, EvalMetric evalMetric, String hyperParams) {
            this.dataset = Objects.requireNonNull(dataset);
            this.evalMetric = Objects.requireNonNull(evalMetric);
            this.hyperParams = Objects.requireNonNull(hyperParams);
        }

        public static class Builder {
            private Dataset dataset;
            private EvalMetric evalMetric;
            private String[] hyperParams;

            public Builder dataset(Dataset dataset) {
                this.dataset = dataset;
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
                    result.add(new TestParameter(dataset, evalMetric, p));
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


    public static abstract class Dataset {

        @Nonnull
        public final String trainDataUrl, testDataUrl;

        public Dataset(@Nonnull String trainDataUrl, @Nonnull String testDataUrl) {
            this.trainDataUrl = Objects.requireNonNull(trainDataUrl);
            this.testDataUrl = Objects.requireNonNull(testDataUrl);;
        }

        @Nonnull
        public String getTrainDataUrl() {
            return trainDataUrl;
        }

        @Nonnull
        public String getTestDataUrl() {
            return testDataUrl;
        }

        public abstract List<Object[]> loadDatasetAsListOfObjects(boolean trainDataset)
                throws IOException;

        public abstract DMatrix loadDatasetAsDMatrix(@Nonnull boolean trainDataset)
                throws IOException, XGBoostError;

        public abstract List<FVec> loadDatasetAsListOfFVec(@Nonnull boolean trainDataset)
                throws IOException;

    }

    public static class LibsvmDataset extends Dataset {

        private final String separator;

        public LibsvmDataset(@Nonnull String trainDataUrl, @Nonnull String testDataUrl) {
            this(trainDataUrl, testDataUrl, " ");
        }

        public LibsvmDataset(@Nonnull String trainDataUrl, @Nonnull String testDataUrl,
                @Nonnull String separator) {
            super(trainDataUrl, testDataUrl);
            this.separator = separator;
        }

        @Override
        public List<Object[]> loadDatasetAsListOfObjects(boolean trainDataset) throws IOException {
            final List<Object[]> dataset = new ArrayList<>();

            URL url = new URL(trainDataset ? trainDataUrl : testDataUrl);
            BufferedReader reader = IOUtils.bufferedReader(url.openStream());

            String line = reader.readLine();
            while (line != null) {
                String[] splitted = line.split(separator);

                Object[] row = new Object[2];
                row[0] = Arrays.asList(Arrays.copyOfRange(splitted, 1, splitted.length));
                row[1] = Double.parseDouble(splitted[0]);
                dataset.add(row);

                line = reader.readLine();
            }

            return dataset;
        }

        @Override
        public DMatrix loadDatasetAsDMatrix(boolean trainDataset) throws IOException, XGBoostError {
            final DMatrixBuilder builder = new SparseDMatrixBuilder(1024, false);
            final FloatArrayList labels = new FloatArrayList(1024);

            URL url = new URL(trainDataset ? trainDataUrl : testDataUrl);
            BufferedReader reader = IOUtils.bufferedReader(url.openStream());

            String line = reader.readLine();
            while (line != null) {
                String[] splitted = line.split(separator);

                float label = Float.parseFloat(splitted[0]);
                labels.add(label);
                builder.nextRow(splitted, 1, splitted.length);

                line = reader.readLine();
            }

            return builder.buildMatrix(labels.toArray());
        }

        @Override
        public List<FVec> loadDatasetAsListOfFVec(boolean trainDataset) throws IOException {
            final List<FVec> dataset = new ArrayList<>();

            URL url = new URL(trainDataset ? trainDataUrl : testDataUrl);
            BufferedReader reader = IOUtils.bufferedReader(url.openStream());

            String line = reader.readLine();
            while (line != null) {
                String[] splitted = line.split(separator);

                FVec fv = XGBoostUtils.parseRowAsFVec(splitted, 1, splitted.length);
                dataset.add(fv);

                line = reader.readLine();
            }

            return dataset;
        }

    }

}
