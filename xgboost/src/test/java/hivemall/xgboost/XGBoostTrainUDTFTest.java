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
import hivemall.utils.math.MathUtils;
import hivemall.xgboost.utils.XGBoostUtils;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
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
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-objective reg:linear")},
            new Object[][] {{Arrays.asList("1:-2", "2:-1"), 0.d}});
    }

    @DataPoints("trial")
    public static final List<TestParameter> trial = TestParameter.merge(
        // 1. binary classification
        // mashroom dataset
        new TestParameter.Builder().trainDataset(new LibsvmDataset(
            "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train"))
                                   .testDatase(new LibsvmDataset(
                                       "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.test"))
                                   .metric(new ClassificationError(0.1f))
                                   .hyperParams(new String[] {
                                           "-objective binary:logistic -iters 10",
                                           "-objective binary:logistic -iters 10 -num_early_stopping_rounds 3"}),
        // 2. multiclass classification
        // https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data
        new TestParameter.Builder().trainDataset(new DermatologyDataset(true, 0.7f))
                                   .testDatase(new DermatologyDataset(false, 0.7f))
                                   .metric(new MultiClassClassificationError(0.1f))
                                   .hyperParams(new String[] {
                                           "-objective multi:softmax -num_class 6 -max_depth 6 -eta 0.1 -num_round 5"}),
        new TestParameter.Builder().trainDataset(new DermatologyDataset(true, 0.7f))
                                   .testDatase(new DermatologyDataset(false, 0.7f))
                                   .metric(new MultiClassClassificationError(0.1f))
                                   .hyperParams(new String[] {
                                           "-objective multi:softprob -num_class 6 -max_depth 6 -eta 0.1 -num_round 5"}),
        // 3. regression
        // https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
        // https://github.com/dmlc/xgboost/blob/master/demo/regression/machine.conf
        new TestParameter.Builder().trainDataset(new LibsvmDataset(
            "https://raw.githubusercontent.com/myui/ml_dataset/master/regr/computer_hardware/machine.txt.train"))
                                   .testDatase(new LibsvmDataset(
                                       "https://raw.githubusercontent.com/myui/ml_dataset/master/regr/computer_hardware/machine.txt.test"))
                                   .metric(new MAE(40f))
                                   .hyperParams(new String[] {
                                           "-booster gbtree -objective reg:linear -eta 1.0 -gamma 1.0 -min_child_weight 1.0 -max_depth 3 -num_round 5"}));


    @Theory
    public void testHyperParams(@FromDataPoints("trial") final TestParameter trial)
            throws Exception {
        final Dataset trainDataset = trial.trainDataset;
        final Dataset testDataset = trial.testDataset;
        final DMatrix testMatrix = testDataset.loadDatasetAsDMatrix();
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
        if (trainDataset.isSparseDataset()) {
            udtf.initialize(new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        trial.hyperParams)});

        } else {
            udtf.initialize(new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaFloatObjectInspector),
                    PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        trial.hyperParams)});
        }

        for (Object[] row : trainDataset.loadDatasetAsListOfObjects()) {
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
                    fvList = testDataset.loadDatasetAsListOfFVec();
                } catch (Exception e) {
                    throw new HiveException(e);
                }
                Assert.assertEquals(expecteds.length, fvList.size());
                int mismatches = 0;
                for (int i = 0; i < expecteds.length; i++) {
                    float[] expected = expecteds[i];
                    FVec fv = fvList.get(i);
                    double[] actual = predictor.predict(fv);
                    Assert.assertEquals(expected.length, actual.length);
                    if (!objName.startsWith("reg:")) {
                        for (int j = 0; j < expected.length; j++) {
                            if (!MathUtils.equals(expected[j], actual[j], 1e-5d)) {
                                mismatches++;
                                break;
                            }
                        }
                    }
                    metric.next(actual, testLabels[i]);
                }
                Assert.assertTrue(
                    "Too many mismatches in prediction result between xgboost4j and xgboost-predictor: "
                            + mismatches,
                    mismatches <= 2);
            }
        });
        udtf.close();
        testMatrix.dispose();

        metric.assertExpected();
    }

    @Test(expected = UDFArgumentException.class)
    public void testNoObjective() throws HiveException {
        XGBoostTrainUDTF udtf = new XGBoostTrainUDTF();
        udtf.initialize(
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    PrimitiveObjectInspectorFactory.javaFloatObjectInspector,
                    ObjectInspectorUtils.getConstantObjectInspector(
                        PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                        "-num_class 4")});
    }


    //---------------------------------------------------
    // multiclass target value tests

    @Test
    public void testCheckTargetValueSucess() throws HiveException {
        XGBoostTrainUDTF udtf = new XGBoostTrainUDTF();
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaFloatObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-objective multi:softmax -num_class 4")});

        udtf.processTargetValue(1.0f);
        udtf.processTargetValue(3f);
    }

    @Test(expected = UDFArgumentException.class)
    public void testCheckInvalidTargetValue1() throws HiveException {
        XGBoostTrainUDTF udtf = new XGBoostTrainUDTF();
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaFloatObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-objective multi:softmax")});

        udtf.processTargetValue(1.1f);
        Assert.fail("-num_class option is missing");
    }

    @Test(expected = UDFArgumentException.class)
    public void testCheckInvalidTargetValue2() throws HiveException {
        XGBoostTrainUDTF udtf = new XGBoostTrainUDTF();
        udtf.processOptions(new ObjectInspector[] {null, null,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-objective multi:softmax -num_class 3")});

        udtf.processTargetValue(-2f);
        Assert.fail();
    }

    @Test(expected = UDFArgumentException.class)
    public void testCheckInvalidTargetValue3() throws HiveException {
        XGBoostTrainUDTF udtf = new XGBoostTrainUDTF();
        udtf.processOptions(new ObjectInspector[] {null, null,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-objective multi:softmax -num_class 3")});

        udtf.processTargetValue(3f);
        Assert.fail();
    }

}
