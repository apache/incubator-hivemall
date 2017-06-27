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
package hivemall.topicmodel;

import hivemall.annotations.VisibleForTesting;
import hivemall.model.FeatureValue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public abstract class AbstractProbabilisticTopicModel {

    // number of topics
    protected final int _K;

    // total number of documents
    @Nonnegative
    protected long _D;

    // for mini-batch
    @Nonnull
    protected final List<Map<String, Float>> _miniBatchDocs;
    protected int _miniBatchSize;

    public AbstractProbabilisticTopicModel(@Nonnegative int K) {
        this._K = K;
        this._D = 0L;
        this._miniBatchDocs = new ArrayList<Map<String, Float>>();
    }

    protected static void initMiniBatch(@Nonnull final String[][] miniBatch,
            @Nonnull final List<Map<String, Float>> docs) {
        docs.clear();

        final FeatureValue probe = new FeatureValue();

        // parse document
        for (final String[] e : miniBatch) {
            if (e == null || e.length == 0) {
                continue;
            }

            final Map<String, Float> doc = new HashMap<String, Float>();

            // parse features
            for (String fv : e) {
                if (fv == null) {
                    continue;
                }
                FeatureValue.parseFeatureAsString(fv, probe);
                String label = probe.getFeatureAsString();
                float value = probe.getValueAsFloat();
                doc.put(label, Float.valueOf(value));
            }

            docs.add(doc);
        }
    }

    protected void accumulateDocCount() {
        this._D += 1;
    }

    @Nonnegative
    protected long getDocCount() {
        return _D;
    }

    protected abstract void train(@Nonnull final String[][] miniBatch);

    protected abstract float computePerplexity();

    @Nonnull
    protected abstract SortedMap<Float, List<String>> getTopicWords(@Nonnegative final int k);

    @Nonnull
    protected abstract float[] getTopicDistribution(@Nonnull final String[] doc);

    @VisibleForTesting
    abstract float getWordScore(@Nonnull final String word, @Nonnegative final int topic);

    protected abstract void setWordScore(@Nonnull final String word, @Nonnegative final int topic,
            final float score);
}
