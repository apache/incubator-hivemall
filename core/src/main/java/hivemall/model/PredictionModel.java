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
package hivemall.model;

import hivemall.mix.MixedModel;
import hivemall.utils.collections.IMapIterator;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public interface PredictionModel extends MixedModel {

    ModelUpdateHandler getUpdateHandler();

    void configureMix(ModelUpdateHandler handler, boolean cancelMixRequest);

    long getNumMixed();

    boolean hasCovariance();

    void configureParams(boolean sum_of_squared_gradients, boolean sum_of_squared_delta_x,
            boolean sum_of_gradients);

    void configureClock();

    boolean hasClock();

    void resetDeltaUpdates(int feature);

    int size();

    boolean contains(@Nonnull Object feature);

    void delete(@Nonnull Object feature);

    @Nullable
    <T extends IWeightValue> T get(@Nonnull Object feature);

    <T extends IWeightValue> void set(@Nonnull Object feature, @Nonnull T value);

    float getWeight(@Nonnull Object feature);

    void setWeight(@Nonnull Object feature, float value);

    float getCovariance(@Nonnull Object feature);

    <K, V extends IWeightValue> IMapIterator<K, V> entries();

}
