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
import hivemall.TestBase;
import hivemall.TestUtils;
import hivemall.utils.lang.PrivilegedAccessor;
import hivemall.utils.lang.mutable.MutableObject;
import hivemall.xgboost.utils.XGBoostUtils;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

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
public class XGBoostTrainUDTFTest extends TestBase {

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
        new TestParameter.Builder().dataset(
            // 1. mashroom dataset
            new LibsvmDataset(
                "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train",
                "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.test"))
                                   .metric(new ClassificationError(0.1f))
                                   .hyperParams(new String[] {
                                           "-objective binary:logistic -iters 10",
                                           "-objective binary:logistic -iters 10 -num_early_stopping_rounds 3"}));


    @Theory
    public void testHyperParams(@FromDataPoints("trial") final TestParameter trial)
            throws HiveException, IOException, XGBoostError {
        final Dataset dataset = trial.dataset;
        final DMatrix testMatrix = dataset.loadDatasetAsDMatrix(false);
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

        for (Object[] row : dataset.loadDatasetAsListOfObjects(true)) {
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
                    fvList = dataset.loadDatasetAsListOfFVec(false);
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

}
