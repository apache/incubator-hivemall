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

import static hivemall.utils.lang.ArrayUtils.newRandomFloatArray;
import static hivemall.utils.math.MathUtils.l1normalize;
import hivemall.annotations.VisibleForTesting;
import hivemall.utils.math.MathUtils;
import hivemall.utils.random.PRNG;
import hivemall.utils.random.RandomNumberGeneratorFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class IncrementalPLSAModel extends AbstractProbabilisticTopicModel {

    // ---------------------------------
    // HyperParameters

    // control how much P(w|z) update is affected by the last value
    private final float _alpha;

    // check convergence of P(w|z) for a document
    private final double _delta;

    // ---------------------------------

    // random number generator
    @Nonnull
    private final PRNG _rnd;

    // optimized in the E step
    private List<Map<String, float[]>> _p_dwz; // P(z|d,w) probability of topics for each document-word (i.e., instance-feature) pair

    // optimized in the M step
    private List<float[]> _p_dz; // P(z|d) probability of topics for documents

    @Nonnull
    private final Map<String, float[]> _p_zw; // P(w|z) probability of words for each topic

    public IncrementalPLSAModel(int K, float alpha, double delta) {
        super(K);

        this._alpha = alpha;
        this._delta = delta;

        this._rnd = RandomNumberGeneratorFactory.createPRNG(1001);

        this._p_zw = new HashMap<String, float[]>();
    }

    protected void train(@Nonnull final String[][] miniBatch) {
        initMiniBatch(miniBatch, _miniBatchDocs);

        this._miniBatchSize = _miniBatchDocs.size();

        initParams();

        final List<float[]> pPrev_dz = new ArrayList<float[]>();

        for (int d = 0; d < _miniBatchSize; d++) {
            do {
                pPrev_dz.clear();
                for (float[] p_dz_d : _p_dz) { // deep copy
                    pPrev_dz.add(p_dz_d.clone());
                }

                // Expectation
                eStep(d);

                // Maximization
                mStep(d);
            } while (!isPdzConverged(d, pPrev_dz, _p_dz)); // until get stable value of P(z|d)
        }
    }

    private void initParams() {
        final List<float[]> p_dz = new ArrayList<float[]>();
        final List<Map<String, float[]>> p_dwz = new ArrayList<Map<String, float[]>>();

        for (int d = 0; d < _miniBatchSize; d++) {
            // init P(z|d)
            float[] p_dz_d = l1normalize(newRandomFloatArray(_K, _rnd));
            p_dz.add(p_dz_d);

            final Map<String, float[]> p_dwz_d = new HashMap<String, float[]>();
            p_dwz.add(p_dwz_d);

            for (final String w : _miniBatchDocs.get(d).keySet()) {
                // init P(z|d,w)
                float[] p_dwz_dw = l1normalize(newRandomFloatArray(_K, _rnd));
                p_dwz_d.put(w, p_dwz_dw);

                // insert new labels to P(w|z)
                if (!_p_zw.containsKey(w)) {
                    _p_zw.put(w, newRandomFloatArray(_K, _rnd));
                }
            }
        }

        // ensure \sum_w P(w|z) = 1
        final double[] sums = new double[_K];
        for (float[] p_zw_w : _p_zw.values()) {
            MathUtils.add(p_zw_w, sums, _K);
        }
        for (float[] p_zw_w : _p_zw.values()) {
            for (int z = 0; z < _K; z++) {
                p_zw_w[z] /= sums[z];
            }
        }

        this._p_dz = p_dz;
        this._p_dwz = p_dwz;
    }

    private void eStep(@Nonnegative final int d) {
        final Map<String, float[]> p_dwz_d = _p_dwz.get(d);
        final float[] p_dz_d = _p_dz.get(d);

        // update P(z|d,w) = P(z|d) * P(w|z)
        for (final String w : _miniBatchDocs.get(d).keySet()) {
            final float[] p_dwz_dw = p_dwz_d.get(w);
            final float[] p_zw_w = _p_zw.get(w);
            for (int z = 0; z < _K; z++) {
                p_dwz_dw[z] = p_dz_d[z] * p_zw_w[z];
            }
            l1normalize(p_dwz_dw);
        }
    }

    private void mStep(@Nonnegative final int d) {
        final Map<String, Float> doc = _miniBatchDocs.get(d);
        final Map<String, float[]> p_dwz_d = _p_dwz.get(d);

        // update P(z|d) = n(d,w) * P(z|d,w)
        final float[] p_dz_d = _p_dz.get(d);
        Arrays.fill(p_dz_d, 0.f); // zero-fill w/ keeping pointer to _p_dz.get(d)
        for (Map.Entry<String, Float> e : doc.entrySet()) {
            final float[] p_dwz_dw = p_dwz_d.get(e.getKey());
            final float n = e.getValue().floatValue();
            for (int z = 0; z < _K; z++) {
                p_dz_d[z] += n * p_dwz_dw[z];
            }
        }
        l1normalize(p_dz_d);

        // update P(w|z) = n(d,w) * P(z|d,w) + alpha * P(w|z)^(n-1)
        final double[] sums = new double[_K];
        for (Map.Entry<String, float[]> e : _p_zw.entrySet()) {
            String w = e.getKey();
            final float[] p_zw_w = e.getValue();

            Float w_value = doc.get(w);
            if (w_value != null) { // all words in the document
                final float n = w_value.floatValue();
                final float[] p_dwz_dw = p_dwz_d.get(w);

                for (int z = 0; z < _K; z++) {
                    p_zw_w[z] = n * p_dwz_dw[z] + _alpha * p_zw_w[z];
                }
            } else { // others
                for (int z = 0; z < _K; z++) {
                    p_zw_w[z] = _alpha * p_zw_w[z];
                }
            }

            MathUtils.add(p_zw_w, sums, _K);
        }
        // normalize to ensure \sum_w P(w|z) = 1
        for (float[] p_zw_w : _p_zw.values()) {
            for (int z = 0; z < _K; z++) {
                p_zw_w[z] = (float) (p_zw_w[z] / sums[z]);
            }
        }
    }

    private boolean isPdzConverged(@Nonnegative final int d, @Nonnull final List<float[]> pPrev_dz,
            @Nonnull final List<float[]> p_dz) {
        final float[] pPrev_dz_d = pPrev_dz.get(d);
        final float[] p_dz_d = p_dz.get(d);

        double diff = 0.d;
        for (int z = 0; z < _K; z++) {
            diff += Math.abs(pPrev_dz_d[z] - p_dz_d[z]);
        }
        return (diff / _K) < _delta;
    }

    protected float computePerplexity() {
        double numer = 0.d;
        double denom = 0.d;

        for (int d = 0; d < _miniBatchSize; d++) {
            final float[] p_dz_d = _p_dz.get(d);
            for (Map.Entry<String, Float> e : _miniBatchDocs.get(d).entrySet()) {
                String w = e.getKey();
                float w_value = e.getValue().floatValue();

                final float[] p_zw_w = _p_zw.get(w);
                double p_dw = 0.d;
                for (int z = 0; z < _K; z++) {
                    p_dw += (double) p_zw_w[z] * p_dz_d[z];
                }

                if (p_dw == 0.d) {
                    throw new IllegalStateException("Perplexity would be Infinity. "
                            + "Try different mini-batch size `-s`, larger `-delta` and/or larger `-alpha`.");
                }
                numer += w_value * Math.log(p_dw);
                denom += w_value;
            }
        }

        return (float) Math.exp(-1.d * (numer / denom));
    }

    @Nonnull
    protected SortedMap<Float, List<String>> getTopicWords(@Nonnegative final int z) {
        final SortedMap<Float, List<String>> res =
                new TreeMap<Float, List<String>>(Collections.reverseOrder());

        for (Map.Entry<String, float[]> e : _p_zw.entrySet()) {
            final String w = e.getKey();
            final float prob = e.getValue()[z];

            List<String> words = res.get(prob);
            if (words == null) {
                words = new ArrayList<String>();
                res.put(prob, words);
            }
            words.add(w);
        }

        return res;
    }

    @Nonnull
    protected float[] getTopicDistribution(@Nonnull final String[] doc) {
        train(new String[][] {doc});
        return _p_dz.get(0);
    }

    @VisibleForTesting
    float getWordScore(@Nonnull final String w, @Nonnegative final int z) {
        return _p_zw.get(w)[z];
    }

    protected void setWordScore(@Nonnull final String w, @Nonnegative final int z,
            final float prob) {
        float[] prob_label = _p_zw.get(w);
        if (prob_label == null) {
            prob_label = newRandomFloatArray(_K, _rnd);
            _p_zw.put(w, prob_label);
        }
        prob_label[z] = prob;

        // ensure \sum_w P(w|z) = 1
        final double[] sums = new double[_K];
        for (float[] p_zw_w : _p_zw.values()) {
            MathUtils.add(p_zw_w, sums, _K);
        }
        for (float[] p_zw_w : _p_zw.values()) {
            for (int zi = 0; zi < _K; zi++) {
                p_zw_w[zi] /= sums[zi];
            }
        }
    }
}
