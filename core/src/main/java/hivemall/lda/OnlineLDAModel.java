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
package hivemall.lda;

import hivemall.utils.lang.Preconditions;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.HashMap;
import java.util.SortedMap;
import java.util.TreeMap;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public final class OnlineLDAModel {

    // number of topics
    private int K_;

    // prior on weight vectors "theta ~ Dir(alpha_)"
    private float alpha_ = 1 / 2.f;

    // prior on topics "beta"
    private float eta_ = 1 / 20.f;

    // total number of documents
    // in the truly online setting, this can be an estimate of the maximum number of documents that could ever seen
    private int D_ = 11102;

    // defined by (tau0 + updateCount)^(-kappa_)
    // controls how much old lambda is forgotten
    private double rhot_;

    // positive value which downweights early iterations
    @Nonnegative
    private double tau0_ = 1020;

    // exponential decay rate (i.e., learning rate) which must be in (0.5, 1] to guarantee convergence
    private double kappa_ = 0.7;

    // how many times EM steps are launched; later EM steps do not drastically forget old lambda
    private long updateCount_ = 0L;

    // random number generator
    private final GammaDistribution gd_;
    private static double SHAPE = 100d;
    private static double SCALE = 1.d / SHAPE;

    // parameters
    @Nonnull
    private List<Map<String, float[]>> phi_;
    private float[][] gamma_;
    private Map<String, float[]> lambda_;

    // check convergence in the expectation (E) step
    private double delta_ = 1E-5;

    private List<Map<String, Float>> miniBatchMap_;
    private int miniBatchSize_;

    private int docCount_ = 0;
    private int wordCount_ = 0;

    public OnlineLDAModel(int K, float alpha, double delta) { // for E step only instantiation
        this(K, alpha, 1 / 20.f, 11102, 1020, 0.7, delta);
    }

    public OnlineLDAModel(int K, float alpha, float eta, int D, double tau0, double kappa, double delta) {
        Preconditions.checkArgument(0.d < tau0, "tau0 MUST be positive: " + tau0);
        Preconditions.checkArgument(0.5 < kappa && kappa <= 1.d,
            "kappa MUST be in (0.5, 1.0]: " + kappa);

        K_ = K;
        alpha_ = alpha;
        eta_ = eta;
        D_ = D;
        tau0_ = tau0;
        kappa_ = kappa;
        delta_ = delta;

        // initialize a random number generator
        gd_ = new GammaDistribution(SHAPE, SCALE);
        gd_.reseedRandomGenerator(1001);

        // initialize the parameters
        lambda_ = new HashMap<String, float[]>(100);
    }

    public void train(@Nonnull String[][] miniBatch) {
        miniBatchSize_ = miniBatch.length;

        // get the number of words(Nd) for each documents
        for (int d = 0; d < miniBatchSize_; d++) {
            wordCount_ += miniBatch[d].length;
        }
        docCount_ += miniBatchSize_;

        makeMiniBatchMap(miniBatch);

        initParams(true);

        rhot_ = Math.pow(tau0_ + updateCount_, -kappa_);

        // Expectation
        stepE();

        // Maximization
        stepM();

        updateCount_ += 1;
    }

    private void makeMiniBatchMap(String[][] miniBatch) {
        miniBatchMap_ = new ArrayList<Map<String, Float>>(); // initialize

        // parse document
        for (int d = 0; d < miniBatchSize_; d++) {
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

            miniBatchMap_.add(docMap);
        }
    }

    private void initParams(boolean gammaWithRandom) {
        phi_ = new ArrayList<Map<String, float[]>>();
        gamma_ = new float[miniBatchSize_][];

        for (int d = 0; d < miniBatchSize_; d++) {
            if (gammaWithRandom) {
                gamma_[d] = generateRandomFloatArray(K_);
            } else {
                gamma_[d] = new float[K_];
                Arrays.fill(gamma_[d], 1.f);;
            }

            // phi does not needed to be initialized
            Map<String, float[]> phi_d = new HashMap<String, float[]>();
            for (String label : miniBatchMap_.get(d).keySet()) {
                phi_d.put(label, new float[K_]);
            }
            phi_.add(phi_d);

            // lambda for newly observed word
            for (String label : miniBatchMap_.get(d).keySet()) {
                if (!lambda_.containsKey(label)) {
                    lambda_.put(label, generateRandomFloatArray(K_));
                }
            }
        }
    }

    private float[] generateRandomFloatArray(int size) {
        float[] ret = new float[size];
        for (int i = 0; i < size; i++) {
            ret[i] = (float) gd_.sample();
        }
        return ret;
    }

    private void stepE() {
        float[] gammaPrev_d;

        // for each of mini-batch documents, update gamma until convergence
        for (int d = 0; d < miniBatchSize_; d++) {
            do {
                // (deep) copy the last gamma values
                gammaPrev_d = gamma_[d].clone();

                updatePhiForSingleDoc(d);
                updateGammaForSingleDoc(d);
            } while (!checkGammaDiff(gammaPrev_d, gamma_[d]));
        }

    }

    private void updatePhiForSingleDoc(int d) {
        // dirichlet_expectation_2d(gamma_)
        float[] eLogTheta_d = new float[K_];
        float gammaSum_d = 0.f;
        for (int k = 0; k < K_; k++) {
            gammaSum_d += gamma_[d][k];
        }
        for (int k = 0; k < K_; k++) {
            eLogTheta_d[k] = (float) (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum_d));
        }

        // dirichlet_expectation_2d(lambda_)
        Map<String, float[]> eLogBeta_d = new HashMap<String, float[]>();
        for (int k = 0; k < K_; k++) {
            float lambdaSum_k = 0.f;
            for (String label : lambda_.keySet()) {
                lambdaSum_k += lambda_.get(label)[k];
            }
            for (String label : miniBatchMap_.get(d).keySet()) {
                float[] eLogBeta_label;
                if (eLogBeta_d.containsKey(label)) {
                    eLogBeta_label = eLogBeta_d.get(label);
                } else {
                    eLogBeta_label = new float[K_];
                    Arrays.fill(eLogBeta_label, 0.f);
                }

                eLogBeta_label[k] = (float) (Gamma.digamma(lambda_.get(label)[k]) - Gamma.digamma(lambdaSum_k));
                eLogBeta_d.put(label, eLogBeta_label);
            }
        }

        // updating phi w/ normalization
        for (String label : miniBatchMap_.get(d).keySet()) {
            float normalizer = 0.f;
            for (int k = 0; k < K_; k++) {
                float phi_dwk = (float) Math.exp(eLogTheta_d[k]
                        + eLogBeta_d.get(label)[k]) + 1E-20f;
                phi_.get(d).get(label)[k] = phi_dwk;
                normalizer += phi_dwk;
            }

            // normalize
            for (int k = 0; k < K_; k++) {
                phi_.get(d).get(label)[k] /= normalizer;
            }
        }
    }

    private void updateGammaForSingleDoc(int d) {
        for (int k = 0; k < K_; k++) {
            float gamma_dk = alpha_;
            for (String label : miniBatchMap_.get(d).keySet()) {
                gamma_dk += phi_.get(d).get(label)[k] * miniBatchMap_.get(d).get(label);
            }
            gamma_[d][k] = gamma_dk;
        }
    }

    private boolean checkGammaDiff(float[] gammaPrev, float[] gammaNext) {
        double diff = 0.d;
        for (int k = 0; k < K_; k++) {
            diff += Math.abs(gammaPrev[k] - gammaNext[k]);
        }
        return (diff / K_) < delta_;
    }

    private void stepM() {
        // calculate lambdaNext
        Map<String, float[]> lambdaNext = new HashMap<String, float[]>();

        float docRatio = (float) D_ / (float) miniBatchSize_;

        for (int d = 0; d < miniBatchSize_; d++) {
            for (String label : miniBatchMap_.get(d).keySet()) {
                float[] lambdaNext_label;
                if (lambdaNext.containsKey(label)) {
                    lambdaNext_label = lambdaNext.get(label);
                } else {
                    lambdaNext_label = new float[K_];
                    Arrays.fill(lambdaNext_label, eta_);
                }
                for (int k = 0; k < K_; k++) {
                    lambdaNext_label[k] += docRatio * phi_.get(d).get(label)[k];
                }
                lambdaNext.put(label, lambdaNext_label);
            }
        }

        // update lambda_
        for (Map.Entry<String, float[]> e : lambda_.entrySet()) {
            String label = e.getKey();
            float[] lambda_label = e.getValue();

            float[] lambdaNext_label;
            if (lambdaNext.containsKey(label)) {
                lambdaNext_label = lambdaNext.get(label);
            } else {
                lambdaNext_label = new float[K_];
                Arrays.fill(lambdaNext_label, eta_);
            }
            for (int k = 0; k < K_; k++) {
                lambda_label[k] = (float) ((1.d - rhot_) * lambda_label[k] + rhot_ * lambdaNext_label[k]);
            }
            lambda_.put(label, lambda_label);
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
        float perWordBound = bound / (float) wordCount_;
        return (float) Math.exp(-1.f * perWordBound);
    }

    /**
     * Estimates the variational bound over all documents using only the documents passed as mini-batch.
     */
    private float computeApproxBoundForMiniBatch() {
        float score = 0.f;
        float tmp;

        // prepare
        float[] gammaSum = new float[miniBatchSize_];
        Arrays.fill(gammaSum, 0.f);
        for (int d = 0; d < miniBatchSize_; d++) {
            for (int k = 0; k < K_; k++) {
                gammaSum[d] += gamma_[d][k];
            }
        }
        float[] lambdaSum = new float[K_];
        Arrays.fill(lambdaSum, 0.f);
        for (int k = 0; k < K_; k++) {
            for (String label : lambda_.keySet()) {
                lambdaSum[k] += lambda_.get(label)[k];
            }
        }

        // E[log p(docs | theta, beta)]
        for (int d = 0; d < miniBatchSize_; d++) {

            // for each word in the document
            for (String label : miniBatchMap_.get(d).keySet()) {
                float wordCount = miniBatchMap_.get(d).get(label);

                tmp = 0.f;
                for (int k = 0; k < K_; k++) {
                    float eLogTheta_dk = (float) (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]));
                    float eLogBeta_kw = 0.f;
                    if (lambda_.containsKey(d)) {
                        eLogBeta_kw = (float) (Gamma.digamma(lambda_.get(d)[k] - Gamma.digamma(lambdaSum[k])));
                    }

                    tmp += phi_.get(d).get(label)[k]
                            * (eLogTheta_dk + eLogBeta_kw - Math.log(phi_.get(d).get(label)[k]));
                }
                score += wordCount * tmp;
            }

            // E[log p(theta | alpha) - log q(theta | gamma)]
            score -= (Gamma.logGamma(gammaSum[d]));
            tmp = 0.f;
            for (int k = 0; k < K_; k++) {
                tmp += (alpha_ - gamma_[d][k])
                        * (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]))
                        + (Gamma.logGamma(gamma_[d][k]));
                tmp /= docCount_;
            }
            score += tmp;

            // E[log p(beta | eta) - log q (beta | lambda)]
            tmp = 0.f;
            for (int k = 0; k < K_; k++) {
                float tmpPartial = 0.f;
                for (String label : lambda_.keySet()) {
                    tmpPartial += (eta_ - lambda_.get(label)[k])
                            * (float) (Gamma.digamma(lambda_.get(label)[k]) - Gamma.digamma(lambdaSum[k]))
                            * (float) (Gamma.logGamma(lambda_.get(label)[k]));
                }

                tmp += (-1.f * (float) Gamma.logGamma(lambdaSum[k])
                        - tmpPartial);
            }
            score += (tmp / miniBatchSize_);

            float W = lambda_.size();
            tmp = (float) (Gamma.logGamma(K_ * alpha_))
                    - (float) (K_ * (Gamma.logGamma(alpha_)))
                    + (((float) (Gamma.logGamma(W * eta_)) - (float) (-1.f * W * (Gamma.logGamma(eta_)))) / docCount_);
            score += tmp;
        }

        return score;
    }

    public double getLambda(String label, int k) {
        Preconditions.checkArgument(lambda_.containsKey(label),
            "Word `" + label + "` is not in the corpus.");
        Preconditions.checkArgument(k < lambda_.get(label).length,
            "Topic index must be in [0, " + lambda_.get(label).length + "]");
        return lambda_.get(label)[k];
    }

    public void setLambda(String label, int k, float lambda) {
        float[] lambda_label;
        if (!lambda_.containsKey(label)) {
            lambda_label = generateRandomFloatArray(K_);
        } else {
            lambda_label = this.lambda_.get(label);
        }
        lambda_label[k] = lambda;
        this.lambda_.put(label, lambda_label);
    }

    public SortedMap<Float, List<String>> getTopicWords(int k) {
        return getTopicWords(k, lambda_.keySet().size());
    }

    public SortedMap<Float, List<String>> getTopicWords(int k, int topN) {
        float lambdaSum = 0.f;
        SortedMap<Float, List<String>> sortedLambda = new TreeMap<Float, List<String>>(Collections.reverseOrder());

        for (String label : lambda_.keySet()) {
            float lambda = lambda_.get(label)[k];
            lambdaSum += lambda;

            List<String> labels = new ArrayList<String>();
            if (sortedLambda.containsKey(lambda)) {
                labels = sortedLambda.get(lambda);
            }
            labels.add(label);

            sortedLambda.put(lambda, labels);
        }

        SortedMap<Float, List<String>> ret = new TreeMap<Float, List<String>>(Collections.reverseOrder());

        topN = Math.min(topN, lambda_.keySet().size());
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

    public float[] getTopicDistribution(@Nonnull String[] doc) {
        miniBatchSize_ = 1;
        makeMiniBatchMap(new String[][] {doc});
        initParams(false);
        stepE();

        float[] topicDistr = new float[K_];

        // normalize
        float gammaSum = 0.f;
        for (int k = 0; k < K_; k++) {
            gammaSum += gamma_[0][k];
        }
        for (int k = 0; k < K_; k++) {
            topicDistr[k] = gamma_[0][k] / gammaSum;
        }

        return topicDistr;
    }
}
