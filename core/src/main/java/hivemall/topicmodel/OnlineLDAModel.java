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
    private final int _K;

    // prior on weight vectors "theta ~ Dir(alpha_)"
    private final float _alpha;

    // prior on topics "beta"
    private final float _eta;

    // total number of documents
    // in the truly online setting, this can be an estimate of the maximum number of documents that could ever seen
    private long _D = -1L;

    // defined by (tau0 + updateCount)^(-kappa_)
    // controls how much old lambda is forgotten
    private double _rhot;

    // positive value which downweights early iterations
    @Nonnegative
    private final double _tau0;

    // exponential decay rate (i.e., learning rate) which must be in (0.5, 1] to guarantee convergence
    private final double _kappa;

    // how many times EM steps are launched; later EM steps do not drastically forget old lambda
    private long _updateCount = 0L;

    // random number generator
    @Nonnull
    private final GammaDistribution _gd;
    private static final double SHAPE = 100.d;
    private static final double SCALE = 1.d / SHAPE;

    // parameters
    @Nonnull
    private List<Map<String, float[]>> _phi;
    private float[][] _gamma;
    @Nonnull
    private final Map<String, float[]> _lambda;

    // check convergence in the expectation (E) step
    private final double _delta;

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
        initMiniBatchMap(miniBatch, _miniBatchMap);

        this._miniBatchSize = _miniBatchMap.size();

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
        for (float[] lambda_label : _lambda.values()) {
            MathUtils.add(lambdaSum, lambda_label, _K);
        }
        final Map<String, float[]> eLogBeta_d = new HashMap<String, float[]>();
        for (String label : doc.keySet()) {
            final float[] lambda_label = _lambda.get(label);
            float[] eLogBeta_label = eLogBeta_d.get(label);
            if (eLogBeta_label == null) {
                eLogBeta_label = new float[_K];
                eLogBeta_d.put(label, eLogBeta_label);
            }
            for (int k = 0; k < _K; k++) {
                eLogBeta_label[k] = (float) (Gamma.digamma(lambda_label[k]) - Gamma.digamma(lambdaSum[k]));
            }
        }

        // updating phi w/ normalization
        for (String label : doc.keySet()) {
            final float[] phi_label = _phi.get(d).get(label);
            final float[] eLogBeta_label = eLogBeta_d.get(label);

            float normalizer = 0.f;
            for (int k = 0; k < _K; k++) {
                final float phiVal = (float) Math.exp(eLogTheta_d[k] + eLogBeta_label[k]) + 1E-20f;
                phi_label[k] = phiVal;
                normalizer += phiVal;
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

        final float[] gamma_d = _gamma[d];
        for (int k = 0; k < _K; k++) {
            gamma_d[k] = _alpha;
        }
        for (Map.Entry<String, Float> e : doc.entrySet()) {
            final float[] phi_label = phi_d.get(e.getKey());
            final float val = e.getValue();
            for (int k = 0; k < _K; k++) {
                gamma_d[k] += phi_label[k] * val;
            }
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
                float[] lambdaTilde_label = lambdaTilde.get(label);
                if (lambdaTilde_label == null) {
                    lambdaTilde_label = ArrayUtils.newInstance(_K, _eta);
                    lambdaTilde.put(label, lambdaTilde_label);
                }

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

            float[] lambdaTilde_label = lambdaTilde.get(label);
            if (lambdaTilde_label == null) {
                lambdaTilde_label = ArrayUtils.newInstance(_K, _eta);
            }

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
        final float[] digamma_gammaSum = MathUtils.digamma(gammaSum);

        final float[] lambdaSum = new float[_K];
        for (float[] lambda_label : _lambda.values()) {
            MathUtils.add(lambdaSum, lambda_label, _K);
        }
        final float[] digamma_lambdaSum = MathUtils.digamma(lambdaSum);

        final float logGamma_alpha = (float) Gamma.logGamma(_alpha);
        final float logGamma_alphaSum = (float) Gamma.logGamma(_K * _alpha);

        for (int d = 0; d < _miniBatchSize; d++) {
            final float digamma_gammaSum_d = digamma_gammaSum[d];

            // E[log p(doc | theta, beta)]
            for (Map.Entry<String, Float> e : _miniBatchMap.get(d).entrySet()) {
                final float[] lambda_label = _lambda.get(e.getKey());

                // logsumexp( Elogthetad + Elogbetad )
                final float[] temp = new float[_K];
                float max = Float.MIN_VALUE;
                for (int k = 0; k < _K; k++) {
                    final float eLogTheta_dk = (float) Gamma.digamma(_gamma[d][k]) - digamma_gammaSum_d;
                    final float eLogBeta_kw = (float) Gamma.digamma(lambda_label[k]) - digamma_lambdaSum[k];

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
                final float gamma_dk = _gamma[d][k];

                // sum( (alpha - gammad) * Elogthetad )
                score += (_alpha - gamma_dk)
                        * ((float) Gamma.digamma(gamma_dk) - digamma_gammaSum_d);

                // sum( gammaln(gammad) - gammaln(alpha) )
                score += (float) Gamma.logGamma(gamma_dk) - logGamma_alpha;
            }
            score += logGamma_alphaSum; // gammaln(sum(alpha))
            score -= Gamma.logGamma(gammaSum[d]); // gammaln(sum(gammad))
        }

        // assuming likelihood for when corpus in the documents is only a subset of the whole corpus
        // (i.e., online setting); likelihood should be always roughly on the same scale
        score *= _docRatio;

        final float logGamma_eta = (float) Gamma.logGamma(_eta);
        final float logGamma_etaSum = (float) Gamma.logGamma(_eta * _lambda.size()); // vocabulary size * eta

        // E[log p(beta | eta) - log q (beta | lambda)]
        for (float[] lambda_label : _lambda.values()) {
            for (int k = 0; k < _K; k++) {
                final float lambda_k = lambda_label[k];

                // sum( (eta - lambda) * Elogbeta )
                score += (_eta - lambda_k)
                        * (float) (Gamma.digamma(lambda_k) - digamma_lambdaSum[k]);

                // sum( gammaln(lambda) - gammaln(eta) )
                score += (float) Gamma.logGamma(lambda_k) - logGamma_eta;
            }
        }
        for (int k = 0; k < _K; k++) {
            // sum( gammaln(etaSum) - gammaln( lambdaSum_k )
            score += logGamma_etaSum - (float) Gamma.logGamma(lambdaSum[k]);
        }

        return score;
    }

    @VisibleForTesting
    double getLambda(@Nonnull final String label, @Nonnegative final int k) {
        final float[] lambda_label = _lambda.get(label);
        if (lambda_label == null) {
            throw new IllegalArgumentException("Word `" + label + "` is not in the corpus.");
        }
        if (k >= lambda_label.length) {
            throw new IllegalArgumentException("Topic index must be in [0, "
                    + _lambda.get(label).length + "]");
        }
        return lambda_label[k];
    }

    public void setLambda(@Nonnull final String label, @Nonnegative final int k, final float lambda_k) {
        float[] lambda_label = _lambda.get(label);
        if (lambda_label == null) {
            lambda_label = ArrayUtils.newRandomFloatArray(_K, _gd);
            _lambda.put(label, lambda_label);
        }
        lambda_label[k] = lambda_k;
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
            final float lambda_k = e.getValue()[k];
            lambdaSum += lambda_k;

            List<String> labels = sortedLambda.get(lambda_k);
            if (labels == null) {
                labels = new ArrayList<String>();
                sortedLambda.put(lambda_k, labels);
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
