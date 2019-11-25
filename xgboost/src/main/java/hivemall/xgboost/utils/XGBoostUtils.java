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
package hivemall.xgboost.utils;

import biz.k11i.xgboost.Predictor;
import biz.k11i.xgboost.util.FVec;
import hivemall.utils.io.FastByteArrayInputStream;
import hivemall.utils.io.IOUtils;
import hivemall.xgboost.XGBoostBatchPredictUDTF.LabeledPointWithRowId;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.io.Text;

public final class XGBoostUtils {

    private XGBoostUtils() {}

    @Nonnull
    public static String getVersion() throws HiveException {
        Properties props = new Properties();
        try (InputStream versionResourceFile =
                Thread.currentThread().getContextClassLoader().getResourceAsStream(
                    "xgboost4j-version.properties")) {
            props.load(versionResourceFile);
        } catch (IOException e) {
            throw new HiveException("Failed to load xgboost4j-version.properties", e);
        }
        return props.getProperty("version", "<unknown>");
    }

    @Nonnull
    public static DMatrix createDMatrix(@Nonnull final List<LabeledPointWithRowId> data)
            throws XGBoostError {
        final List<LabeledPoint> points = new ArrayList<>(data.size());
        for (LabeledPointWithRowId d : data) {
            points.add(d);
        }
        return new DMatrix(points.iterator(), "");
    }

    @Nonnull
    public static Booster createBooster(@Nonnull DMatrix matrix,
            @Nonnull Map<String, Object> params) throws NoSuchMethodException, XGBoostError,
            IllegalAccessException, InvocationTargetException, InstantiationException {
        Class<?>[] args = {Map.class, DMatrix[].class};
        Constructor<Booster> ctor = Booster.class.getDeclaredConstructor(args);
        ctor.setAccessible(true);
        return ctor.newInstance(new Object[] {params, new DMatrix[] {matrix}});
    }

    public static void close(@Nullable final DMatrix matrix) {
        if (matrix == null) {
            return;
        }
        try {
            matrix.dispose();
        } catch (Throwable e) {
            ;
        }
    }

    public static void close(@Nullable final Booster booster) {
        if (booster == null) {
            return;
        }
        try {
            booster.dispose();
        } catch (Throwable e) {
            ;
        }
    }

    @Nonnull
    public static Text serializeBooster(@Nonnull final Booster booster) throws HiveException {
        try {
            byte[] b = IOUtils.toCompressedText(booster.toByteArray());
            return new Text(b);
        } catch (Throwable e) {
            throw new HiveException("Failed to serialize a booster", e);
        }
    }

    @Nonnull
    public static Booster deserializeBooster(@Nonnull final Text model) throws HiveException {
        try {
            byte[] b = IOUtils.fromCompressedText(model.getBytes(), model.getLength());
            return XGBoost.loadModel(new FastByteArrayInputStream(b));
        } catch (Throwable e) {
            throw new HiveException("Failed to deserialize a booster", e);
        }
    }

    @Nonnull
    public static Predictor loadPredictor(@Nonnull final Text model) throws HiveException {
        try {
            byte[] b = IOUtils.fromCompressedText(model.getBytes(), model.getLength());
            return new Predictor(new FastByteArrayInputStream(b));
        } catch (Throwable e) {
            throw new HiveException("Failed to create a predictor", e);
        }
    }

    @Nonnull
    public static FVec parseRowAsFVec(@Nonnull final String[] row, final int start, final int end) {
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
