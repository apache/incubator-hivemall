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

import hivemall.utils.io.FastByteArrayInputStream;
import ml.dmlc.xgboost4j.LabeledPoint;
import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.Map;
import java.util.Properties;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.metadata.HiveException;

public final class XGBoostUtils {

    private XGBoostUtils() {}

    /** Transform List<String> inputs into a XGBoost input format */
    @Nullable
    public static LabeledPoint parseFeatures(@Nonnull final String rowId,
            @Nonnull final String[] features, final float target) {
        final int size = features.length;
        if (size == 0) {
            return null;
        }

        final int[] indices = new int[size];
        final float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            if (features[i] == null) {
                continue;
            }
            final String str = features[i];
            final int pos = str.indexOf(':');
            if (pos >= 1) {
                indices[i] = Integer.parseInt(str.substring(0, pos));
                values[i] = Float.parseFloat(str.substring(pos + 1));
            }
        }

        return new LabeledPoint(target, indices, values);
    }

    @Nonnull
    public static String getVersion() throws HiveException {
        Properties props = new Properties();
        try (InputStream versionResourceFile =
                Thread.currentThread()
                      .getContextClassLoader()
                      .getResourceAsStream("xgboost4j-version.properties")) {
            props.load(versionResourceFile);
        } catch (IOException e) {
            throw new HiveException("Failed to load xgboost4j-version.properties", e);
        }
        return props.getProperty("version", "<unknown>");
    }

    @Nonnull
    public static Booster loadBooster(@Nonnull final byte[] input) throws HiveException {
        try {
            return XGBoost.loadModel(new FastByteArrayInputStream(input));
        } catch (Exception e) {
            throw new HiveException(e);
        }
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

}
