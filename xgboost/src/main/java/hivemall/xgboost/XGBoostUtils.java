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

import ml.dmlc.xgboost4j.LabeledPoint;

import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.metadata.HiveException;

public final class XGBoostUtils {

    private XGBoostUtils() {}

    /** Transform List<String> inputs into a XGBoost input format */
    @Nullable
    public static LabeledPoint parseFeatures(final double target,
            @Nonnull final String[] features) {
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


        return new LabeledPoint((float) target, indices, values);
    }

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

}
