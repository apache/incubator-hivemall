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
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.math.MathUtils;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

public final class OnlineLDAModel {

    // number of topics
    private int _K;

    // prior on weight vectors "theta ~ Dir(alpha_)"
    private float _alpha = 1 / 2.f;

    // prior on topics "beta"
    private float _eta = 1 / 20.f;

    // total number of documents
    // in the truly online setting, this can be an estimate of the maximum number of documents that could ever seen
    private long _D = -1L;

    // defined by (tau0 + updateCount)^(-kappa_)
    // controls how much old lambda is forgotten
    private double _rhot;

    // positive value which downweights early iterations
    @Nonnegative
    private double _tau0 = 1020;

    // exponential decay rate (i.e., learning rate) which must be in (0.5, 1] to guarantee convergence
    private double _kappa = 0.7d;

    // how many times EM steps are launched; later EM steps do not drastically forget old lambda
    private long _updateCount = 0L;

    // random number generator
    @Nonnull
    private final GammaDistribution _gd;
    private static double SHAPE = 100.d;
    private static double SCALE = 1.d / SHAPE;

    // parameters
    @Nonnull
    private List<Map<String, float[]>> _phi;
    private float[][] _gamma;
    @Nonnull
    private final Map<String, float[]> _lambda;

    // check convergence in the expectation (E) step
    private double _delta = 1e-5;

    @Nonnull
    private final List<Map<String, Float>> _miniBatchMap;
    private int _miniBatchSize;

    // for computing perplexity
    private float _docRatio = 1.f;
    private long _wordCount = 0L;

    public OnlineLDAModel(int K, float alpha, double delta) { // for E step only instantiation
        this(K, alpha, 1 / 20.f, -1L, 1020, 0.7, delta);
    }

    public OnlineLDAModel(int K, float alpha, float eta, long D, double tau0, double kappa,
            double delta) {
        if (tau0 < 0.d) {
            throw new IllegalArgumentException("tau0 MUST be positive: " + tau0);
        }
        if (kappa <= 0.5 || 1.d < kappa) {
            throw new IllegalArgumentException("kappa MUST be in (0.5, 1.0]: " + kappa);
        }

        this._K = K;
        this._alpha = alpha;
        this._eta = eta;
        this._D = D;
        this._tau0 = tau0;
        this._kappa = kappa;
        this._delta = delta;

        // initialize a random number generator
        this._gd = new GammaDistribution(SHAPE, SCALE);
        _gd.reseedRandomGenerator(1001);

        // initialize the parameters
        this._lambda = new HashMap<String, float[]>(100);

        this._miniBatchMap = new ArrayList<Map<String, Float>>();
    }

    /**
     * In a truly online setting, total number of documents corresponds to the number of documents
     * that have ever seen. In that case, users need to manually set the current max number of documents
     * via this method.
     * Note that, since the same set of documents could be repeatedly passed to `train()`,
     * simply accumulating `_miniBatchSize`s as estimated `_D` is not sufficient.
     */
    public void setNumTotalDocs(@Nonnegative long D) {
        this._D = D;
    }

    public void train(@Nonnull final String[][] miniBatch) {
        if (_D <= 0L) {
            throw new RuntimeException("Total number of documents MUST be set via `setNumTotalDocs()`");
        }

        preprocessMiniBatch(miniBatch);

        initParams(true);

        // Expectation
        eStep();

        this._rhot = Math.pow(_tau0 + _updateCount, -_kappa);

        // Maximization
        mStep();

        _updateCount++;
    }

    private void preprocessMiniBatch(@Nonnull final String[][] miniBatch) {
        this._miniBatchSize = miniBatch.length;

        initMiniBatchMap(miniBatch, _miniBatchMap);

        // accumulate the number of words for each documents
        this._wordCount = 0L;
        for (int d = 0; d < _miniBatchSize; d++) {
            for (float n : _miniBatchMap.get(d).values()) {
                this._wordCount += n;
            }
        }

        this._docRatio = (float)((double) _D / _miniBatchSize);
    }

    private static void initMiniBatchMap(@Nonnull final String[][] miniBatch,
            @Nonnull final List<Map<String, Float>> map) {
        map.clear();

        final FeatureValue probe = new FeatureValue();

        // parse document
        for (final String[] e : miniBatch) {
            if (e == null) {
                continue;
            }

            final Map<String, Float> docMap = new HashMap<String, Float>();

            // parse features
            for (String fv : e) {
                if (fv == null) {
                    continue;
                }
                FeatureValue.parseFeatureAsString(fv, probe);
                String label = probe.getFeatureAsString();
                float value = probe.getValueAsFloat();
                docMap.put(label, value);
            }

            map.add(docMap);
        }
    }

    private void initParams(boolean gammaWithRandom) {
        _phi = new ArrayList<Map<String, float[]>>();
        _gamma = new float[_miniBatchSize][];

        for (int d = 0; d < _miniBatchSize; d++) {
            if (gammaWithRandom) {
                _gamma[d] = ArrayUtils.newRandomFloatArray(_K, _gd);
            } else {
                _gamma[d] = ArrayUtils.newInstance(_K, 1.f);
            }

            final Map<String, float[]> phi_d = new HashMap<String, float[]>();
            _phi.add(phi_d);
            for (String label : _miniBatchMap.get(d).keySet()) {
                phi_d.put(label, new float[_K]);
                if (!_lambda.containsKey(label)) { // lambda for newly observed word
                    _lambda.put(label, ArrayUtils.newRandomFloatArray(_K, _gd));
                }
            }
        }
    }

    private void eStep() {
        float[] gamma_d, gammaPrev_d;

        // for each of mini-batch documents, update gamma until convergence
        for (int d = 0; d < _miniBatchSize; d++) {
            gamma_d = _gamma[d];
            do {
                // (deep) copy the last gamma values
                gammaPrev_d = gamma_d.clone();

                updatePhiPerDoc(d);
                updateGammaPerDoc(d);
            } while (!checkGammaDiff(gammaPrev_d, gamma_d));
        }
    }

    private void updatePhiPerDoc(@Nonnegative final int d) {
        final Map<String, Float> doc = _miniBatchMap.get(d);

        // Dirichlet expectation (2d) for gamma
        final float[] eLogTheta_d = new float[_K];
        final float gammaSum_d = MathUtils.sum(_gamma[d]);
        for (int k = 0; k < _K; k++) {
            eLogTheta_d[k] = (float) (Gamma.digamma(_gamma[d][k]) - Gamma.digamma(gammaSum_d));
        }

        // Dirichlet expectation (2d) for lambda
        final float[] lambdaSum = new float[_K];
        for (float[] lambda : _lambda.values()) {
            MathUtils.add(lambdaSum, lambda, _K);
        }
        final Map<String, float[]> eLogBeta_d = new HashMap<String, float[]>();
        for (int k = 0; k < _K; k++) {
            for (String label : doc.keySet()) {
                final float[] eLogBeta_label;
                if (eLogBeta_d.containsKey(label)) {
                    eLogBeta_label = eLogBeta_d.get(label);
                } else {
                    eLogBeta_label = new float[_K];
                }
                eLogBeta_d.put(label, eLogBeta_label);

                eLogBeta_label[k] = (float) (Gamma.digamma(_lambda.get(label)[k]) - Gamma.digamma(lambdaSum[k]));
            }
        }

        // updating phi w/ normalization
        for (String label : doc.keySet()) {
            final float[] phi_label = _phi.get(d).get(label);
            _phi.get(d).put(label, phi_label);

            float normalizer = 0.f;
            for (int k = 0; k < _K; k++) {
                phi_label[k] = (float) Math.exp(eLogTheta_d[k] + eLogBeta_d.get(label)[k]) + 1E-20f;
                normalizer += phi_label[k];
            }

            // normalize
            for (int k = 0; k < _K; k++) {
                phi_label[k] /= normalizer;
            }
        }
    }

    private void updateGammaPerDoc(@Nonnegative final int d) {
        final Map<String, Float> doc = _miniBatchMap.get(d);
        final Map<String, float[]> phi_d = _phi.get(d);

        for (int k = 0; k < _K; k++) {
            float gamma_dk = _alpha;
            for (Map.Entry<String, Float> e : doc.entrySet()) {
                gamma_dk += phi_d.get(e.getKey())[k] * e.getValue();
            }
            _gamma[d][k] = gamma_dk;
        }
    }

    private boolean checkGammaDiff(@Nonnull final float[] gammaPrev,
            @Nonnull final float[] gammaNext) {
        double diff = 0.d;
        for (int k = 0; k < _K; k++) {
            diff += Math.abs(gammaPrev[k] - gammaNext[k]);
        }
        return (diff / _K) < _delta;
    }

    private void mStep() {
        // calculate lambdaTilde for vocabularies in the current mini-batch
        final Map<String, float[]> lambdaTilde = new HashMap<String, float[]>();
        for (int d = 0; d < _miniBatchSize; d++) {
            final Map<String, float[]> phi_d = _phi.get(d);
            for (String label : _miniBatchMap.get(d).keySet()) {
                float[] lambdaTilde_label;
                if (lambdaTilde.containsKey(label)) {
                    lambdaTilde_label = lambdaTilde.get(label);
                } else {
                    lambdaTilde_label = ArrayUtils.newInstance(_K, _eta);
                }
                lambdaTilde.put(label, lambdaTilde_label);

                final float[] phi_label = phi_d.get(label);
                for (int k = 0; k < _K; k++) {
                    lambdaTilde_label[k] += _docRatio * phi_label[k];
                }
            }
        }

        // update lambda for all vocabularies
        for (Map.Entry<String, float[]> e : _lambda.entrySet()) {
            String label = e.getKey();
            final float[] lambda_label = e.getValue();

            float[] lambdaTilde_label;
            if (lambdaTilde.containsKey(label)) {
                lambdaTilde_label = lambdaTilde.get(label);
            } else {
                lambdaTilde_label = ArrayUtils.newInstance(_K, _eta);
            }
            _lambda.put(label, lambda_label);

            for (int k = 0; k < _K; k++) {
                lambda_label[k] = (float) ((1.d - _rhot) * lambda_label[k] + _rhot
                        * lambdaTilde_label[k]);
            }
        }
    }

    /**
     * Calculate approximate perplexity for the current mini-batch.
     */
    public float computePerplexity() {
        float bound = computeApproxBound();
        float perWordBound = bound / (_docRatio * _wordCount);
        return (float) Math.exp(-1.f * perWordBound);
    }

    /**
     * Estimates the variational bound over all documents using only the documents passed as mini-batch.
     */
    private float computeApproxBound() {
        float score = 0.f;

        // prepare
        final float[] gammaSum = new float[_miniBatchSize];
        for (int d = 0; d < _miniBatchSize; d++) {
            gammaSum[d] = MathUtils.sum(_gamma[d]);
        }
        final float[] lambdaSum = new float[_K];
        for (float[] lambda : _lambda.values()) {
            MathUtils.add(lambdaSum, lambda, _K);
        }

        for (int d = 0; d < _miniBatchSize; d++) {
            // E[log p(doc | theta, beta)]
            for (Map.Entry<String, Float> e : _miniBatchMap.get(d).entrySet()) {
                final float[] lambda = _lambda.get(e.getKey());

                // logsumexp( Elogthetad + Elogbetad )
                final float[] temp = new float[_K];
                float max = Float.MIN_VALUE;
                for (int k = 0; k < _K; k++) {
                    final float eLogTheta_dk = (float) (Gamma.digamma(_gamma[d][k]) - Gamma.digamma(gammaSum[d]));
                    final float eLogBeta_kw = (float) (Gamma.digamma(lambda[k]) - Gamma.digamma(lambdaSum[k]));

                    temp[k] = eLogTheta_dk + eLogBeta_kw;
                    if (temp[k] > max) {
                        max = temp[k];
                    }
                }
                float logsumexp = 0.f;
                for (int k = 0; k < _K; k++) {
                    logsumexp += (float) Math.exp(temp[k] - max);
                }
                logsumexp = max + (float) Math.log(logsumexp);

                // sum( word count * logsumexp(...) )
                score += e.getValue() * logsumexp;
            }

            // E[log p(theta | alpha) - log q(theta | gamma)]
            for (int k = 0; k < _K; k++) {
                // sum( (alpha - gammad) * Elogthetad )
                score += (_alpha - _gamma[d][k])
                        * (float) (Gamma.digamma(_gamma[d][k]) - Gamma.digamma(gammaSum[d]));

                // sum( gammaln(gammad) - gammaln(alpha) )
                score += (float) (Gamma.logGamma(_gamma[d][k]) - Gamma.logGamma(_alpha));
            }
            score += (float) Gamma.logGamma(_K * _alpha); // gammaln(sum(alpha))
            score -= Gamma.logGamma(gammaSum[d]); // gammaln(sum(gammad))
        }

        // assuming likelihood for when corpus in the documents is only a subset of the whole corpus
        // (i.e., online setting); likelihood should be always roughly on the same scale
        score *= _docRatio;

        // E[log p(beta | eta) - log q (beta | lambda)]
        float etaSum = _eta * _lambda.size(); // vocabulary size * eta
        for (int k = 0; k < _K; k++) {
            for (float[] lambda : _lambda.values()) {
                // sum( (eta - lambda) * Elogbeta )
                score += (_eta - lambda[k])
                        * (float) (Gamma.digamma(lambda[k]) - Gamma.digamma(lambdaSum[k]));

                // sum( gammaln(lambda) - gammaln(eta) )
                score += (float) (Gamma.logGamma(lambda[k]) - Gamma.logGamma(_eta));
            }

            // sum( gammaln(etaSum) - gammaln( lambdaSum_k )
            score += (float) (Gamma.logGamma(etaSum) - Gamma.logGamma(lambdaSum[k]));
        }

        return score;
    }

    @VisibleForTesting
    double getLambda(@Nonnull final String label, @Nonnegative final int k) {
        final float[] lambda = _lambda.get(label);
        if (lambda == null) {
            throw new IllegalArgumentException("Word `" + label + "` is not in the corpus.");
        }
        if (k >= lambda.length) {
            throw new IllegalArgumentException("Topic index must be in [0, "
                    + _lambda.get(label).length + "]");
        }
        return lambda[k];
    }

    public void setLambda(@Nonnull final String label, @Nonnegative final int k, final float lambda) {
        float[] lambda_label = _lambda.get(label);
        if (lambda_label == null) {
            lambda_label = ArrayUtils.newRandomFloatArray(_K, _gd);
            _lambda.put(label, lambda_label);
        }
        lambda_label[k] = lambda;
    }

    @Nonnull
    public SortedMap<Float, List<String>> getTopicWords(@Nonnegative final int k) {
        return getTopicWords(k, _lambda.keySet().size());
    }

    @Nonnull
    public SortedMap<Float, List<String>> getTopicWords(@Nonnegative final int k,
            @Nonnegative int topN) {
        float lambdaSum = 0.f;
        final SortedMap<Float, List<String>> sortedLambda = new TreeMap<Float, List<String>>(
            Collections.reverseOrder());

        for (Map.Entry<String, float[]> e : _lambda.entrySet()) {
            final float lambda = e.getValue()[k];
            lambdaSum += lambda;

            List<String> labels = sortedLambda.get(lambda);
            if (labels == null) {
                labels = new ArrayList<String>();
                sortedLambda.put(lambda, labels);
            }
            labels.add(e.getKey());
        }

        final SortedMap<Float, List<String>> ret = new TreeMap<Float, List<String>>(
            Collections.reverseOrder());

        topN = Math.min(topN, _lambda.keySet().size());
        int tt = 0;
        for (Map.Entry<Float, List<String>> e : sortedLambda.entrySet()) {
            ret.put(e.getKey() / lambdaSum, e.getValue());

            if (++tt == topN) {
                break;
            }
        }

        return ret;
    }

    @Nonnull
    public float[] getTopicDistribution(@Nonnull final String[] doc) {
        preprocessMiniBatch(new String[][] {doc});

        initParams(false);

        eStep();

        // normalize topic distribution
        final float[] topicDistr = new float[_K];
        final float[] gamma0 = _gamma[0];
        final float gammaSum = MathUtils.sum(gamma0);
        for (int k = 0; k < _K; k++) {
            topicDistr[k] = gamma0[k] / gammaSum;
        }
        return topicDistr;
    }

}
