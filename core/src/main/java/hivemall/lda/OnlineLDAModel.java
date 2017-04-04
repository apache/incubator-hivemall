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

    // defined by (tau0 + countEMStep)^(-kappa_)
    private double rhot;

    // positive value which downweights early iterations
    @Nonnegative
    private double tau0_ = 1020;

    // exponential decay rate (i.e., learning rate) which must be in (0.5, 1] to guarantee convergence
    private double kappa_ = 0.7;

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
    private static double DELTA = 1E-5;

    // for update size of lambda_
    private static int dummySize_ = 100;
    @Nonnull
    private final float[][] dummyLambdas_;

    private List<Map<String, Float>> miniBatchMap_;
    private int miniBatchSize_;

    private int accumDocCount_ = 0;
    private int accumWordCount_ = 0;

    public OnlineLDAModel(int K, float alpha, float eta, int D, double tau0, double kappa) {
        Preconditions.checkArgument(0.d < tau0, "tau0 MUST be positive: " + tau0);
        Preconditions.checkArgument(0.5 < kappa && kappa <= 1.d,
            "kappa MUST be in (0.5, 1.0]: " + kappa);

        K_ = K;
        alpha_ = alpha;
        eta_ = eta;
        D_ = D;
        tau0_ = tau0;
        kappa_ = kappa;

        // initialize a random number generator
        gd_ = new GammaDistribution(SHAPE, SCALE);
        gd_.reseedRandomGenerator(1001);

        // initialize the parameters
        lambda_ = new HashMap<String, float[]>();

        dummyLambdas_ = generateDummyLambda();
    }

    @Nonnull
    private float[][] generateDummyLambda() {
        float[][] ary = new float[dummySize_][];
        double[] tmpDArray;
        for (int b = 0; b < dummySize_; b++) {
            float[] tmpDummyLambda = new float[K_];
            tmpDArray = gd_.sample(K_);
            for (int k = 0; k < K_; k++) {
                tmpDummyLambda[k] = (float) tmpDArray[k];
            }
            ary[b] = tmpDummyLambda;
        }
        return ary;
    }

    public void train(@Nonnull String[][] miniBatch, int time) {
        miniBatchSize_ = miniBatch.length;

        rhot = Math.pow(tau0_ + time, -kappa_);

        // get the number of words(Nd) for each documents
        getMiniBatchParams(miniBatch);
        accumDocCount_ += miniBatchSize_;

        makeMiniBatchMap(miniBatch);

        updateSizeOfParameterForMiniBatch();

        // Expectation
        float[][] lastGamma;
        do {
            // (deep) copy the last gamma values
            lastGamma = new float[gamma_.length][];
            for (int d = 0; d < gamma_.length; d++) {
                lastGamma[d] = gamma_[d].clone();
            }

            stepE();
        } while (!checkGammaDiff(lastGamma, gamma_));

        // Maximization
        stepM();
    }

    private void stepE() {
        // 1) Updating phi_

        // dirichlet_expectation_2d(gamma_)
        float[][] ElogTheta = new float[miniBatchSize_][K_];
        for (int d = 0; d < miniBatchSize_; d++) {
            float gammaSum_d = 0.f;
            for (int k = 0; k < K_; k++) {
                gammaSum_d += gamma_[d][k];
            }
            for (int k = 0; k < K_; k++) {
                ElogTheta[d][k] = (float) (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum_d));
            }
        }

        // dirichlet_expectation_2d(lambda_)
        Map<String, float[]> ElogBeta = new HashMap<String, float[]>();
        for (int k = 0; k < K_; k++) {
            float lambdaSum_k = 0.f;
            for (String label : lambda_.keySet()) {
                lambdaSum_k += lambda_.get(label)[k];
            }
            for (int d = 0; d < miniBatchSize_; d++) {
                for (String label : miniBatchMap_.get(d).keySet()) {
                    float[] ElogBeta_label;
                    if (ElogBeta.containsKey(label)) {
                        ElogBeta_label = ElogBeta.get(label);
                    } else {
                        ElogBeta_label = new float[K_];
                        Arrays.fill(ElogBeta_label, 0.f);
                    }

                    ElogBeta_label[k] = (float) (Gamma.digamma(lambda_.get(label)[k]) - Gamma.digamma(lambdaSum_k));
                    ElogBeta.put(label, ElogBeta_label);
                }
            }
        }

        // updating phi_ w/ normalization
        for (int d = 0; d < miniBatchSize_; d++) {
            for (String label : miniBatchMap_.get(d).keySet()) {
                float normalizer = 0.f;
                for (int k = 0; k < K_; k++) {
                    phi_.get(d).get(label)[k] = (float) Math.exp(ElogTheta[d][k]
                            + ElogBeta.get(label)[k]) + 1E-20f;
                    normalizer += phi_.get(d).get(label)[k];
                }

                // normalize
                for (int k = 0; k < K_; k++) {
                    phi_.get(d).get(label)[k] /= normalizer;
                }
            }
        }

        // 2) Updating gamma_
        for (int d = 0; d < miniBatchSize_; d++) {
            for (int k = 0; k < K_; k++) {
                float gamma_dk = alpha_;
                for (String label : miniBatchMap_.get(d).keySet()) {
                    gamma_dk += phi_.get(d).get(label)[k] * miniBatchMap_.get(d).get(label);
                }
                gamma_[d][k] = gamma_dk;
            }
        }
    }

    private void stepM() {
        // calculate lambdaBar
        Map<String, float[]> lambdaBar = new HashMap<String, float[]>();

        float docRatio = (float) D_ / (float) miniBatchSize_;

        for (int d = 0; d < miniBatchSize_; d++) {
            for (String label : miniBatchMap_.get(d).keySet()) {
                float[] lambdaBar_label;
                if (lambdaBar.containsKey(label)) {
                    lambdaBar_label = lambdaBar.get(label);
                } else {
                    lambdaBar_label = new float[K_];
                    Arrays.fill(lambdaBar_label, eta_);
                }
                for (int k = 0; k < K_; k++) {
                    lambdaBar_label[k] += docRatio * phi_.get(d).get(label)[k];
                }
                lambdaBar.put(label, lambdaBar_label);
            }
        }

        // update lambda_
        for (Map.Entry<String, float[]> e : lambda_.entrySet()) {
            String label = e.getKey();
            float[] lambda_label = e.getValue();

            float[] lambdaBar_label;
            if (lambdaBar.containsKey(label)) {
                lambdaBar_label = lambdaBar.get(label);
            } else {
                lambdaBar_label = new float[K_];
                Arrays.fill(lambdaBar_label, eta_);
            }
            for (int k = 0; k < K_; k++) {
                lambda_label[k] = (float) ((1.d - rhot) * lambda_label[k] + rhot * lambdaBar_label[k]);
            }
            lambda_.put(label, lambda_label);
        }
    }

    private void getMiniBatchParams(String[][] miniBatch) {
        miniBatchSize_ = miniBatch.length;

        for (int d = 0; d < miniBatchSize_; d++) {
            accumWordCount_ += miniBatch[d].length;
        }
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

    private void updateSizeOfParameterForMiniBatch() {
        phi_ = new ArrayList<Map<String, float[]>>();
        gamma_ = new float[miniBatchSize_][];

        for (int d = 0; d < miniBatchSize_; d++) {
            float[] gammad = getRandomGammaArray();
            gamma_[d] = gammad;

            // phi_ not needed to be initialized
            Map<String, float[]> phid = new HashMap<String, float[]>();
            for (String label : miniBatchMap_.get(d).keySet()) {
                float[] tmpFArray = new float[K_];
                phid.put(label, tmpFArray);
            }
            phi_.add(phid);
        }

        // lambda
        for (int d = 0; d < miniBatchSize_; d++) {
            for (String label : miniBatchMap_.get(d).keySet()) {
                if (!lambda_.containsKey(label)) {
                    int tmpLambdaIdx = lambda_.size() % dummySize_;
                    float[] lambdaNW = getRandomGammaArray();
                    for (int k = 0; k < K_; k++) {
                        lambdaNW[k] *= dummyLambdas_[tmpLambdaIdx][k];
                    }
                    lambda_.put(label, lambdaNW);
                }
            }
        }
    }

    private float[] getRandomGammaArray() {
        double[] dret = gd_.sample(K_);
        float[] ret = new float[K_];

        for (int k = 0; k < ret.length; k++) {
            ret[k] = (float) dret[k];
        }

        return ret;
    }

    private boolean checkGammaDiff(float[][] lastGamma, float[][] nextGamma) {
        double diff = 0.d;
        for (int d = 0; d < miniBatchSize_; d++) {
            for (int k = 0; k < K_; k++) {
                diff += Math.abs(lastGamma[d][k] - nextGamma[d][k]);
            }
        }
        return (diff < DELTA * miniBatchSize_ * K_);
    }

    /**
     * Methods for debugging and checking convergence:
     */

    /**
     * Calculate approximate perplexity for the current mini-batch.
     */
    public float computePerplexity() {
        float bound = computeApproxBoundForMiniBatch();
        float perWordBound = bound / (float) accumWordCount_;
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
                    float ElogTheta_dk = (float) (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum[d]));
                    float ElogBeta_kw = 0.f;
                    if (lambda_.containsKey(d)) {
                        ElogBeta_kw = (float) (Gamma.digamma(lambda_.get(d)[k] - Gamma.digamma(lambdaSum[k])));
                    }

                    tmp += phi_.get(d).get(label)[k]
                            * (ElogTheta_dk + ElogBeta_kw - Math.log(phi_.get(d).get(label)[k]));
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
                tmp /= accumDocCount_;
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
                    + (((float) (Gamma.logGamma(W * eta_)) - (float) (-1.f * W * (Gamma.logGamma(eta_)))) / accumDocCount_);
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

    public SortedMap<Float, String> getTopicWords(int k) {
        return getTopicWords(k, lambda_.keySet().size());
    }

    public SortedMap<Float, String> getTopicWords(int k, int topN) {
        float lambdaSum = 0.f;
        SortedMap<Float, String> sortedLambda = new TreeMap<Float, String>(Collections.reverseOrder());

        for (String label : lambda_.keySet()) {
            float lambda = lambda_.get(label)[k];
            lambdaSum += lambda;
            sortedLambda.put(lambda, label);
        }

        SortedMap<Float, String> ret = new TreeMap<Float, String>(Collections.reverseOrder());

        topN = Math.min(topN, lambda_.keySet().size());
        int tt = 0;
        for (Map.Entry<Float, String> e : sortedLambda.entrySet()) {
            float lambda = e.getKey();
            String label = e.getValue();
            ret.put(lambda / lambdaSum, label);

            if (++tt == topN) {
                break;
            }
        }

        return ret;
    }

}
