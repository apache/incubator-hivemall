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

import hivemall.utils.lang.Preconditions;

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
    private int _D = 11102;

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
    private Map<String, float[]> _lambda;

    // check convergence in the expectation (E) step
    private double _delta = 1e-5;

    private List<Map<String, Float>> _miniBatchMap;
    private int _miniBatchSize;

    private int _docCount = 0;
    private int _wordCount = 0;

    public OnlineLDAModel(int K, float alpha, double delta) { // for E step only instantiation
        this(K, alpha, 1 / 20.f, 11102, 1020, 0.7, delta);
    }

    public OnlineLDAModel(int K, float alpha, float eta, int D, double tau0, double kappa,
            double delta) {
        Preconditions.checkArgument(0.d < tau0, "tau0 MUST be positive: " + tau0);
        Preconditions.checkArgument(0.5 < kappa && kappa <= 1.d, "kappa MUST be in (0.5, 1.0]: "
                + kappa);

        _K = K;
        _alpha = alpha;
        _eta = eta;
        _D = D;
        _tau0 = tau0;
        _kappa = kappa;
        _delta = delta;

        // initialize a random number generator
        _gd = new GammaDistribution(SHAPE, SCALE);
        _gd.reseedRandomGenerator(1001);

        // initialize the parameters
        _lambda = new HashMap<String, float[]>(100);
    }

    public void train(@Nonnull String[][] miniBatch) {
        _miniBatchSize = miniBatch.length;

        // get the number of words(Nd) for each documents
        for (int d = 0; d < _miniBatchSize; d++) {
            _wordCount += miniBatch[d].length;
        }
        _docCount += _miniBatchSize;

        makeMiniBatchMap(miniBatch);

        initParams(true);

        _rhot = Math.pow(_tau0 + _updateCount, -_kappa);

        // Expectation
        eStep();

        // Maximization
        mStep();

        _updateCount += 1;
    }

    private void makeMiniBatchMap(String[][] miniBatch) {
        _miniBatchMap = new ArrayList<Map<String, Float>>(); // initialize

        // parse document
        for (int d = 0; d < _miniBatchSize; d++) {
            Map<String, Float> docMap = new HashMap<String, Float>();

            // parse features
            for (int w = 0; w < miniBatch[d].length; w++) {
                String fv = miniBatch[d][w];
                String[] parsedFeature = fv.split(":"); // [`label`, `value`]
                if (parsedFeature.length == 1) { // wrong format
                    continue;
                }
                String label = parsedFeature[0];
                float value = Float.parseFloat(parsedFeature[1]);
                docMap.put(label, value);
            }

            _miniBatchMap.add(docMap);
        }
    }

    private void initParams(boolean gammaWithRandom) {
        _phi = new ArrayList<Map<String, float[]>>();
        _gamma = new float[_miniBatchSize][];

        for (int d = 0; d < _miniBatchSize; d++) {
            if (gammaWithRandom) {
                _gamma[d] = generateRandomFloatArray(_K);
            } else {
                _gamma[d] = new float[_K];
                Arrays.fill(_gamma[d], 1.f);
            }

            // phi does not needed to be initialized
            Map<String, float[]> phi_d = new HashMap<String, float[]>();
            for (String label : _miniBatchMap.get(d).keySet()) {
                phi_d.put(label, new float[_K]);
            }
            _phi.add(phi_d);

            // lambda for newly observed word
            for (String label : _miniBatchMap.get(d).keySet()) {
                if (!_lambda.containsKey(label)) {
                    _lambda.put(label, generateRandomFloatArray(_K));
                }
            }
        }
    }

    @Nonnull
    private float[] generateRandomFloatArray(@Nonnegative int size) {
        float[] ret = new float[size];
        for (int i = 0; i < size; i++) {
            ret[i] = (float) _gd.sample();
        }
        return ret;
    }

    private void eStep() {
        float[] gammaPrev_d;

        // for each of mini-batch documents, update gamma until convergence
        for (int d = 0; d < _miniBatchSize; d++) {
            do {
                // (deep) copy the last gamma values
                gammaPrev_d = _gamma[d].clone();

                updatePhiForSingleDoc(d);
                updateGammaForSingleDoc(d);
            } while (!checkGammaDiff(gammaPrev_d, _gamma[d]));
        }

    }

    private void updatePhiForSingleDoc(int d) {
        // dirichlet_expectation_2d(gamma_)
        float[] eLogTheta_d = new float[_K];
        float gammaSum_d = 0.f;
        for (int k = 0; k < _K; k++) {
            gammaSum_d += _gamma[d][k];
        }
        for (int k = 0; k < _K; k++) {
            eLogTheta_d[k] = (float) (Gamma.digamma(_gamma[d][k]) - Gamma.digamma(gammaSum_d));
        }

        // dirichlet_expectation_2d(lambda_)
        Map<String, float[]> eLogBeta_d = new HashMap<String, float[]>();
        for (int k = 0; k < _K; k++) {
            float lambdaSum_k = 0.f;
            for (String label : _lambda.keySet()) {
                lambdaSum_k += _lambda.get(label)[k];
            }
            for (String label : _miniBatchMap.get(d).keySet()) {
                float[] eLogBeta_label;
                if (eLogBeta_d.containsKey(label)) {
                    eLogBeta_label = eLogBeta_d.get(label);
                } else {
                    eLogBeta_label = new float[_K];
                    Arrays.fill(eLogBeta_label, 0.f);
                }

                eLogBeta_label[k] = (float) (Gamma.digamma(_lambda.get(label)[k]) - Gamma.digamma(lambdaSum_k));
                eLogBeta_d.put(label, eLogBeta_label);
            }
        }

        // updating phi w/ normalization
        for (String label : _miniBatchMap.get(d).keySet()) {
            float normalizer = 0.f;
            for (int k = 0; k < _K; k++) {
                float phi_dwk = (float) Math.exp(eLogTheta_d[k] + eLogBeta_d.get(label)[k]) + 1E-20f;
                _phi.get(d).get(label)[k] = phi_dwk;
                normalizer += phi_dwk;
            }

            // normalize
            for (int k = 0; k < _K; k++) {
                _phi.get(d).get(label)[k] /= normalizer;
            }
        }
    }

    private void updateGammaForSingleDoc(int d) {
        for (int k = 0; k < _K; k++) {
            float gamma_dk = _alpha;
            for (String label : _miniBatchMap.get(d).keySet()) {
                gamma_dk += _phi.get(d).get(label)[k] * _miniBatchMap.get(d).get(label);
            }
            _gamma[d][k] = gamma_dk;
        }
    }

    private boolean checkGammaDiff(float[] gammaPrev, float[] gammaNext) {
        double diff = 0.d;
        for (int k = 0; k < _K; k++) {
            diff += Math.abs(gammaPrev[k] - gammaNext[k]);
        }
        return (diff / _K) < _delta;
    }

    private void mStep() {
        // calculate lambdaNext
        Map<String, float[]> lambdaNext = new HashMap<String, float[]>();

        float docRatio = (float) _D / (float) _miniBatchSize;

        for (int d = 0; d < _miniBatchSize; d++) {
            for (String label : _miniBatchMap.get(d).keySet()) {
                float[] lambdaNext_label;
                if (lambdaNext.containsKey(label)) {
                    lambdaNext_label = lambdaNext.get(label);
                } else {
                    lambdaNext_label = new float[_K];
                    Arrays.fill(lambdaNext_label, _eta);
                }
                for (int k = 0; k < _K; k++) {
                    lambdaNext_label[k] += docRatio * _phi.get(d).get(label)[k];
                }
                lambdaNext.put(label, lambdaNext_label);
            }
        }

        // update lambda_
        for (Map.Entry<String, float[]> e : _lambda.entrySet()) {
            String label = e.getKey();
            float[] lambda_label = e.getValue();

            float[] lambdaNext_label;
            if (lambdaNext.containsKey(label)) {
                lambdaNext_label = lambdaNext.get(label);
            } else {
                lambdaNext_label = new float[_K];
                Arrays.fill(lambdaNext_label, _eta);
            }
            for (int k = 0; k < _K; k++) {
                lambda_label[k] = (float) ((1.d - _rhot) * lambda_label[k] + _rhot
                        * lambdaNext_label[k]);
            }
            _lambda.put(label, lambda_label);
        }
    }

    /**
     * Methods for debugging and checking convergence:
     */

    /**
     * Calculate approximate perplexity for the current mini-batch.
     */
    public float computePerplexity() {
        float bound = computeApproxBoundForMiniBatch();
        float perWordBound = bound / (float) _wordCount;
        return (float) Math.exp(-1.f * perWordBound);
    }

    /**
     * Estimates the variational bound over all documents using only the documents passed as mini-batch.
     */
    private float computeApproxBoundForMiniBatch() {
        float score = 0.f;
        float tmp;

        // prepare
        float[] gammaSum = new float[_miniBatchSize];
        Arrays.fill(gammaSum, 0.f);
        for (int d = 0; d < _miniBatchSize; d++) {
            for (int k = 0; k < _K; k++) {
                gammaSum[d] += _gamma[d][k];
            }
        }
        float[] lambdaSum = new float[_K];
        Arrays.fill(lambdaSum, 0.f);
        for (int k = 0; k < _K; k++) {
            for (String label : _lambda.keySet()) {
                lambdaSum[k] += _lambda.get(label)[k];
            }
        }

        // E[log p(docs | theta, beta)]
        for (int d = 0; d < _miniBatchSize; d++) {

            // for each word in the document
            for (String label : _miniBatchMap.get(d).keySet()) {
                float wordCount = _miniBatchMap.get(d).get(label);

                tmp = 0.f;
                for (int k = 0; k < _K; k++) {
                    float eLogTheta_dk = (float) (Gamma.digamma(_gamma[d][k]) - Gamma.digamma(gammaSum[d]));
                    float eLogBeta_kw = 0.f;
                    if (_lambda.containsKey(d)) {
                        eLogBeta_kw = (float) (Gamma.digamma(_lambda.get(d)[k]
                                - Gamma.digamma(lambdaSum[k])));
                    }

                    tmp += _phi.get(d).get(label)[k]
                            * (eLogTheta_dk + eLogBeta_kw - Math.log(_phi.get(d).get(label)[k]));
                }
                score += wordCount * tmp;
            }

            // E[log p(theta | alpha) - log q(theta | gamma)]
            score -= (Gamma.logGamma(gammaSum[d]));
            tmp = 0.f;
            for (int k = 0; k < _K; k++) {
                tmp += (_alpha - _gamma[d][k])
                        * (Gamma.digamma(_gamma[d][k]) - Gamma.digamma(gammaSum[d]))
                        + (Gamma.logGamma(_gamma[d][k]));
                tmp /= _docCount;
            }
            score += tmp;

            // E[log p(beta | eta) - log q (beta | lambda)]
            tmp = 0.f;
            for (int k = 0; k < _K; k++) {
                float tmpPartial = 0.f;
                for (String label : _lambda.keySet()) {
                    tmpPartial += (_eta - _lambda.get(label)[k])
                            * (float) (Gamma.digamma(_lambda.get(label)[k]) - Gamma.digamma(lambdaSum[k]))
                            * (float) (Gamma.logGamma(_lambda.get(label)[k]));
                }

                tmp += (-1.f * (float) Gamma.logGamma(lambdaSum[k]) - tmpPartial);
            }
            score += (tmp / _miniBatchSize);

            float W = _lambda.size();
            tmp = (float) (Gamma.logGamma(_K * _alpha))
                    - (float) (_K * (Gamma.logGamma(_alpha)))
                    + (((float) (Gamma.logGamma(W * _eta)) - (float) (-1.f * W * (Gamma.logGamma(_eta)))) / _docCount);
            score += tmp;
        }

        return score;
    }

    public double getLambda(String label, int k) {
        Preconditions.checkArgument(_lambda.containsKey(label), "Word `" + label
                + "` is not in the corpus.");
        Preconditions.checkArgument(k < _lambda.get(label).length, "Topic index must be in [0, "
                + _lambda.get(label).length + "]");
        return _lambda.get(label)[k];
    }

    public void setLambda(String label, int k, float lambda) {
        float[] lambda_label;
        if (!_lambda.containsKey(label)) {
            lambda_label = generateRandomFloatArray(_K);
        } else {
            lambda_label = this._lambda.get(label);
        }
        lambda_label[k] = lambda;
        this._lambda.put(label, lambda_label);
    }

    public SortedMap<Float, List<String>> getTopicWords(int k) {
        return getTopicWords(k, _lambda.keySet().size());
    }

    public SortedMap<Float, List<String>> getTopicWords(int k, int topN) {
        float lambdaSum = 0.f;
        SortedMap<Float, List<String>> sortedLambda = new TreeMap<Float, List<String>>(
            Collections.reverseOrder());

        for (String label : _lambda.keySet()) {
            float lambda = _lambda.get(label)[k];
            lambdaSum += lambda;

            List<String> labels = new ArrayList<String>();
            if (sortedLambda.containsKey(lambda)) {
                labels = sortedLambda.get(lambda);
            }
            labels.add(label);

            sortedLambda.put(lambda, labels);
        }

        SortedMap<Float, List<String>> ret = new TreeMap<Float, List<String>>(
            Collections.reverseOrder());

        topN = Math.min(topN, _lambda.keySet().size());
        int tt = 0;
        for (Map.Entry<Float, List<String>> e : sortedLambda.entrySet()) {
            float lambda = e.getKey();
            List<String> labels = e.getValue();
            ret.put(lambda / lambdaSum, labels);

            if (++tt == topN) {
                break;
            }
        }

        return ret;
    }

    @Nonnull
    public float[] getTopicDistribution(@Nonnull String[] doc) {
        _miniBatchSize = 1;
        makeMiniBatchMap(new String[][] {doc});
        initParams(false);
        eStep();

        float[] topicDistr = new float[_K];

        // normalize
        float gammaSum = 0.f;
        for (int k = 0; k < _K; k++) {
            gammaSum += _gamma[0][k];
        }
        for (int k = 0; k < _K; k++) {
            topicDistr[k] = _gamma[0][k] / gammaSum;
        }

        return topicDistr;
    }

}
