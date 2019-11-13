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
import hivemall.TestUtils;
import hivemall.utils.io.IOUtils;
import hivemall.utils.lang.PrivilegedAccessor;
import hivemall.xgboost.utils.XGBoostUtils;
import ml.dmlc.xgboost4j.java.Booster;

import java.io.BufferedReader;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    @Test
    public void testBinaryLogistic() throws HiveException, IOException {
        XGBoostTrainUDTF udtf = new XGBoostTrainUDTF();
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-objective binary:logistic")});

        String url =
                "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train";
        List<Object[]> dataset = loadDataset(url);
        for (Object[] row : dataset) {
            udtf.process(row);
        }

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object input) throws HiveException {
                Object[] forwardedObj = (Object[]) input;
                String modelId = (String) forwardedObj[0];
                Text modelStr = (Text) forwardedObj[1];
                Booster booster = XGBoostUtils.deserializeBooster(modelStr);
                XGBoostUtils.close(booster);
                Predictor predictor = XGBoostUtils.loadPredictor(modelStr);
                final String gbmName, objName;
                try {
                    gbmName = (String) PrivilegedAccessor.getValue(predictor, "name_gbm");
                    objName = (String) PrivilegedAccessor.getValue(predictor, "name_obj");
                } catch (Exception e) {
                    throw new HiveException(e);
                }
                Assert.assertEquals("gbtree", gbmName);
                Assert.assertEquals("binary:logistic", objName);
                System.out.println(modelId);

            }
        });
        udtf.close();
    }

    @Test
    public void testBinaryLogisticEarlyStopping() throws HiveException, IOException {
        XGBoostTrainUDTF udtf = new XGBoostTrainUDTF();
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector,
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                    "-objective binary:logistic -iters 10 -num_early_stopping_rounds 3")});

        String url =
                "https://raw.githubusercontent.com/dmlc/xgboost/master/demo/data/agaricus.txt.train";
        for (Object[] row : loadDataset(url)) {
            udtf.process(row);
        }

        udtf.setCollector(new Collector() {
            @Override
            public void collect(Object input) throws HiveException {
                Object[] forwardedObj = (Object[]) input;
                System.out.println(forwardedObj[0]);
            }
        });
        udtf.close();
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

}
