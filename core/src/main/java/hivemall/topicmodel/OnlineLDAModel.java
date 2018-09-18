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

public final class OnlineLDAModel extends AbstractProbabilisticTopicModel {

    private static final double SHAPE = 100.d;
    private static final double SCALE = 1.d / SHAPE;

    // ---------------------------------
    // HyperParameters

    // prior on weight vectors "theta ~ Dir(alpha_)"
    private final float _alpha;

    // prior on topics "beta"
    private final float _eta;

    // positive value which downweights early iterations
    @Nonnegative
    private final double _tau0;

    // exponential decay rate (i.e., learning rate) which must be in (0.5, 1] to guarantee convergence
    @Nonnegative
    private final double _kappa;

    // check convergence in the expectation (E) step
    private final double _delta;

    // ---------------------------------

    // how many times EM steps are launched; later EM steps do not drastically forget old lambda
    private long _updateCount = 0L;

    // defined by (tau0 + updateCount)^(-kappa_)
    // controls how much old lambda is forgotten
    private double _rhot;

    // if `num_docs` option is not given, this flag will be true
    // in that case, UDTF automatically sets `count` value to the _D parameter in an online LDA model
    private final boolean _isAutoD;

    // parameters
    private List<Map<String, float[]>> _phi;
    private float[][] _gamma;
    @Nonnull
    private final Map<String, float[]> _lambda;

    // random number generator
    @Nonnull
    private final GammaDistribution _gd;

    // for computing perplexity
    private float _docRatio = 1.f;
    private double _valueSum = 0.d;

    public OnlineLDAModel(int K, float alpha, double delta) { // for E step only instantiation
        this(K, alpha, 1 / 20.f, -1L, 1020, 0.7, delta);
    }

    public OnlineLDAModel(int K, float alpha, float eta, long D, double tau0, double kappa,
            double delta) {
        super(K);

        if (tau0 < 0.d) {
            throw new IllegalArgumentException("tau0 MUST be positive: " + tau0);
        }
        if (kappa <= 0.5 || 1.d < kappa) {
            throw new IllegalArgumentException("kappa MUST be in (0.5, 1.0]: " + kappa);
        }

        this._alpha = alpha;
        this._eta = eta;
        this._D = D;
        this._tau0 = tau0;
        this._kappa = kappa;
        this._delta = delta;

        this._isAutoD = (_D <= 0L);

        // initialize a random number generator
        this._gd = new GammaDistribution(SHAPE, SCALE);
        _gd.reseedRandomGenerator(1001);

        // initialize the parameters
        this._lambda = new HashMap<String, float[]>(100);
    }

    @Override
    protected void accumulateDocCount() {
        /*
         * In a truly online setting, total number of documents equals to the number of documents that have ever seen.
         * In that case, users need to manually set the current max number of documents via this method.
         * Note that, since the same set of documents could be repeatedly passed to `train()`,
         * simply accumulating `_miniBatchSize`s as estimated `_D` is not sufficient.
         */
        if (_isAutoD) {
            this._D += 1;
        }
    }

    protected void train(@Nonnull final String[][] miniBatch) {
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
        initMiniBatch(miniBatch, _miniBatchDocs);

        this._miniBatchSize = _miniBatchDocs.size();

        // accumulate the number of words for each documents
        double valueSum = 0.d;
        for (int d = 0; d < _miniBatchSize; d++) {
            for (Float n : _miniBatchDocs.get(d).values()) {
                valueSum += n.floatValue();
            }
        }
        this._valueSum = valueSum;

        this._docRatio = (float) ((double) _D / _miniBatchSize);
    }

    private void initParams(final boolean gammaWithRandom) {
        final List<Map<String, float[]>> phi = new ArrayList<Map<String, float[]>>();
        final float[][] gamma = new float[_miniBatchSize][];

        for (int d = 0; d < _miniBatchSize; d++) {
            if (gammaWithRandom) {
                gamma[d] = ArrayUtils.newRandomFloatArray(_K, _gd);
            } else {
                gamma[d] = ArrayUtils.newFloatArray(_K, 1.f);
            }

            final Map<String, float[]> phi_d = new HashMap<String, float[]>();
            phi.add(phi_d);
            for (final String label : _miniBatchDocs.get(d).keySet()) {
                phi_d.put(label, new float[_K]);
                if (!_lambda.containsKey(label)) { // lambda for newly observed word
                    _lambda.put(label, ArrayUtils.newRandomFloatArray(_K, _gd));
                }
            }
        }

        this._phi = phi;
        this._gamma = gamma;
    }

    private void eStep() {
        // since lambda is invariant in the expectation step,
        // `digamma`s of lambda values for Elogbeta are pre-computed
        final double[] lambdaSum = new double[_K];
        final Map<String, float[]> digamma_lambda = new HashMap<String, float[]>();
        for (Map.Entry<String, float[]> e : _lambda.entrySet()) {
            String label = e.getKey();
            float[] lambda_label = e.getValue();

            // for digamma(lambdaSum)
            MathUtils.add(lambda_label, lambdaSum, _K);

            digamma_lambda.put(label, MathUtils.digamma(lambda_label));
        }

        final double[] digamma_lambdaSum = MathUtils.digamma(lambdaSum);
        // for each of mini-batch documents, update gamma until convergence
        float[] gamma_d, gammaPrev_d;
        Map<String, float[]> eLogBeta_d;
        for (int d = 0; d < _miniBatchSize; d++) {
            gamma_d = _gamma[d];
            eLogBeta_d = computeElogBetaPerDoc(d, digamma_lambda, digamma_lambdaSum);

            do {
                gammaPrev_d = gamma_d.clone(); // deep copy the last gamma values

                updatePhiPerDoc(d, eLogBeta_d);
                updateGammaPerDoc(d);
            } while (!checkGammaDiff(gammaPrev_d, gamma_d));
        }
    }

    @Nonnull
    private Map<String, float[]> computeElogBetaPerDoc(@Nonnegative final int d,
            @Nonnull final Map<String, float[]> digamma_lambda,
            @Nonnull final double[] digamma_lambdaSum) {
        final Map<String, Float> doc = _miniBatchDocs.get(d);

        // Dirichlet expectation (2d) for lambda
        final Map<String, float[]> eLogBeta_d = new HashMap<String, float[]>(doc.size());
        for (final String label : doc.keySet()) {
            float[] eLogBeta_label = eLogBeta_d.get(label);
            if (eLogBeta_label == null) {
                eLogBeta_label = new float[_K];
                eLogBeta_d.put(label, eLogBeta_label);
            }
            final float[] digamma_lambda_label = digamma_lambda.get(label);
            for (int k = 0; k < _K; k++) {
                eLogBeta_label[k] = (float) (digamma_lambda_label[k] - digamma_lambdaSum[k]);
            }
        }

        return eLogBeta_d;
    }

    private void updatePhiPerDoc(@Nonnegative final int d,
            @Nonnull final Map<String, float[]> eLogBeta_d) {
        // Dirichlet expectation (2d) for gamma
        final float[] gamma_d = _gamma[d];
        final double digamma_gammaSum_d = Gamma.digamma(MathUtils.sum(gamma_d));
        final double[] eLogTheta_d = new double[_K];
        for (int k = 0; k < _K; k++) {
            eLogTheta_d[k] = Gamma.digamma(gamma_d[k]) - digamma_gammaSum_d;
        }

        // updating phi w/ normalization
        final Map<String, float[]> phi_d = _phi.get(d);
        final Map<String, Float> doc = _miniBatchDocs.get(d);
        for (String label : doc.keySet()) {
            final float[] phi_label = phi_d.get(label);
            final float[] eLogBeta_label = eLogBeta_d.get(label);

            double normalizer = 0.d;
            for (int k = 0; k < _K; k++) {
                float phiVal = (float) Math.exp(eLogBeta_label[k] + eLogTheta_d[k]) + 1E-20f;
                phi_label[k] = phiVal;
                normalizer += phiVal;
            }

            for (int k = 0; k < _K; k++) {
                phi_label[k] /= normalizer;
            }
        }
    }

    private void updateGammaPerDoc(@Nonnegative final int d) {
        final Map<String, Float> doc = _miniBatchDocs.get(d);
        final Map<String, float[]> phi_d = _phi.get(d);

        final float[] gamma_d = _gamma[d];
        for (int k = 0; k < _K; k++) {
            gamma_d[k] = _alpha;
        }
        for (Map.Entry<String, Float> e : doc.entrySet()) {
            final float[] phi_label = phi_d.get(e.getKey());
            final float val = e.getValue().floatValue();
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
            for (String label : _miniBatchDocs.get(d).keySet()) {
                float[] lambdaTilde_label = lambdaTilde.get(label);
                if (lambdaTilde_label == null) {
                    lambdaTilde_label = ArrayUtils.newFloatArray(_K, _eta);
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
                lambdaTilde_label = ArrayUtils.newFloatArray(_K, _eta);
            }

            for (int k = 0; k < _K; k++) {
                lambda_label[k] =
                        (float) ((1.d - _rhot) * lambda_label[k] + _rhot * lambdaTilde_label[k]);
            }
        }
    }

    /**
     * Calculate approximate perplexity for the current mini-batch.
     */
    protected float computePerplexity() {
        double bound = computeApproxBound();
        double perWordBound = bound / (_docRatio * _valueSum);
        return (float) Math.exp(-1.d * perWordBound);
    }

    /**
     * Estimates the variational bound over all documents using only the documents passed as
     * mini-batch.
     */
    private double computeApproxBound() {
        // prepare
        final double[] gammaSum = new double[_miniBatchSize];
        for (int d = 0; d < _miniBatchSize; d++) {
            gammaSum[d] = MathUtils.sum(_gamma[d]);
        }
        final double[] digamma_gammaSum = MathUtils.digamma(gammaSum);

        final double[] lambdaSum = new double[_K];
        for (float[] lambda_label : _lambda.values()) {
            MathUtils.add(lambda_label, lambdaSum, _K);
        }
        final double[] digamma_lambdaSum = MathUtils.digamma(lambdaSum);

        final double logGamma_alpha = Gamma.logGamma(_alpha);
        final double logGamma_alphaSum = Gamma.logGamma(_K * _alpha);

        double score = 0.d;
        for (int d = 0; d < _miniBatchSize; d++) {
            final double digamma_gammaSum_d = digamma_gammaSum[d];
            final float[] gamma_d = _gamma[d];

            // E[log p(doc | theta, beta)]
            for (Map.Entry<String, Float> e : _miniBatchDocs.get(d).entrySet()) {
                final float[] lambda_label = _lambda.get(e.getKey());

                // logsumexp( Elogthetad + Elogbetad )
                final double[] temp = new double[_K];
                double max = Double.MIN_VALUE;
                for (int k = 0; k < _K; k++) {
                    double eLogTheta_dk = Gamma.digamma(gamma_d[k]) - digamma_gammaSum_d;
                    double eLogBeta_kw = Gamma.digamma(lambda_label[k]) - digamma_lambdaSum[k];
                    final double tempK = eLogTheta_dk + eLogBeta_kw;
                    if (tempK > max) {
                        max = tempK;
                    }
                    temp[k] = tempK;
                }
                double logsumexp = MathUtils.logsumexp(temp, max);

                // sum( word count * logsumexp(...) )
                score += e.getValue().floatValue() * logsumexp;
            }

            // E[log p(theta | alpha) - log q(theta | gamma)]
            for (int k = 0; k < _K; k++) {
                float gamma_dk = gamma_d[k];

                // sum( (alpha - gammad) * Elogthetad )
                score += (_alpha - gamma_dk) * (Gamma.digamma(gamma_dk) - digamma_gammaSum_d);

                // sum( gammaln(gammad) - gammaln(alpha) )
                score += Gamma.logGamma(gamma_dk) - logGamma_alpha;
            }
            score += logGamma_alphaSum; // gammaln(sum(alpha))
            score -= Gamma.logGamma(gammaSum[d]); // gammaln(sum(gammad))
        }

        // assuming likelihood for when corpus in the documents is only a subset of the whole corpus
        // (i.e., online setting); likelihood should be always roughly on the same scale
        score *= _docRatio;

        final double logGamma_eta = Gamma.logGamma(_eta);
        final double logGamma_etaSum = Gamma.logGamma(_eta * _lambda.size()); // vocabulary size * eta

        // E[log p(beta | eta) - log q (beta | lambda)]
        for (final float[] lambda_label : _lambda.values()) {
            for (int k = 0; k < _K; k++) {
                float lambda_label_k = lambda_label[k];

                // sum( (eta - lambda) * Elogbeta )
                score += (_eta - lambda_label_k)
                        * (Gamma.digamma(lambda_label_k) - digamma_lambdaSum[k]);

                // sum( gammaln(lambda) - gammaln(eta) )
                score += Gamma.logGamma(lambda_label_k) - logGamma_eta;
            }
        }
        for (int k = 0; k < _K; k++) {
            // sum( gammaln(etaSum) - gammaln( lambdaSum_k )
            score += logGamma_etaSum - Gamma.logGamma(lambdaSum[k]);
        }

        return score;
    }

    @VisibleForTesting
    float getWordScore(@Nonnull final String label, @Nonnegative final int k) {
        final float[] lambda_label = _lambda.get(label);
        if (lambda_label == null) {
            throw new IllegalArgumentException("Word `" + label + "` is not in the corpus.");
        }
        if (k >= lambda_label.length) {
            throw new IllegalArgumentException(
                "Topic index must be in [0, " + _lambda.get(label).length + "]");
        }
        return lambda_label[k];
    }

    protected void setWordScore(@Nonnull final String label, @Nonnegative final int k,
            final float lambda_k) {
        float[] lambda_label = _lambda.get(label);
        if (lambda_label == null) {
            lambda_label = ArrayUtils.newRandomFloatArray(_K, _gd);
            _lambda.put(label, lambda_label);
        }
        lambda_label[k] = lambda_k;
    }

    @Nonnull
    protected SortedMap<Float, List<String>> getTopicWords(@Nonnegative final int k) {
        return getTopicWords(k, _lambda.keySet().size());
    }

    @Nonnull
    public SortedMap<Float, List<String>> getTopicWords(@Nonnegative final int k,
            @Nonnegative int topN) {
        double lambdaSum = 0.d;
        final SortedMap<Float, List<String>> sortedLambda =
                new TreeMap<Float, List<String>>(Collections.reverseOrder());

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

        final SortedMap<Float, List<String>> ret =
                new TreeMap<Float, List<String>>(Collections.reverseOrder());

        topN = Math.min(topN, _lambda.keySet().size());
        int tt = 0;
        for (Map.Entry<Float, List<String>> e : sortedLambda.entrySet()) {
            float key = (float) (e.getKey().floatValue() / lambdaSum);
            ret.put(Float.valueOf(key), e.getValue());

            if (++tt == topN) {
                break;
            }
        }

        return ret;
    }

    @Nonnull
    protected float[] getTopicDistribution(@Nonnull final String[] doc) {
        preprocessMiniBatch(new String[][] {doc});

        initParams(false);

        eStep();

        // normalize topic distribution
        final float[] topicDistr = new float[_K];
        final float[] gamma0 = _gamma[0];
        final double gammaSum = MathUtils.sum(gamma0);
        for (int k = 0; k < _K; k++) {
            topicDistr[k] = (float) (gamma0[k] / gammaSum);
        }
        return topicDistr;
    }

}
