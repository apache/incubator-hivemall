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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.special.Gamma;

public final class OnlineLDAModel {

    private static final boolean printLambda = false;
    private static final boolean printGamma = false;
    private static final boolean printPhi = false;

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
    private double tau0_ = 1020;

    // exponential decay rate (i.e., learning rate) which must be in (0.5, 1] to guarantee convergence
    private double kappa_ = 0.7;

    // random number generator
    GammaDistribution gd;
    private double SHAPE = 100d;
    private double SCALE = 1.d / SHAPE;

    // parameters
    ArrayList<HashMap<String, float[]>> phi_;
    float[][] gamma_;
    HashMap<String, float[]> lambda_;

    // check convergence in the expectation (E) step
    private double DELTA = 1E-5;

    // for update size of lambda_
    private int dummySize = 100;
    private float[][] dummyLambdas;

    private ArrayList<HashMap<String, Float>> miniBatchMap;
    private int miniBatchSize;

    private int accumDocCount = 0;
    private int accumWordCount = 0;

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
        gd = new GammaDistribution(SHAPE, SCALE);
        gd.reseedRandomGenerator(1001);

        // initialize the parameters
        lambda_ = new HashMap<String, float[]>();

        setDummyLambda();
    }

    private void setDummyLambda() {
        dummyLambdas = new float[dummySize][];
        double[] tmpDArray;
        for (int b = 0; b < dummySize; b++) {
            float[] tmpDummyLambda = new float[K_];
            tmpDArray = gd.sample(K_);
            for (int k = 0; k < K_; k++) {
                tmpDummyLambda[k] = (float) tmpDArray[k];
            }
            dummyLambdas[b] = tmpDummyLambda;
        }
    }

    public void train(String[][] miniBatch, int time) {
        miniBatchSize = miniBatch.length;

        rhot = Math.pow(tau0_ + time, -kappa_);

        if (printLambda) {
            System.out.println("lambda:");
            for (String key : lambda_.keySet()) {
                System.out.println(Arrays.toString(lambda_.get(key)));
            }
        }

        // get the number of words(Nd) for each documents
        getMiniBatchParams(miniBatch);
        accumDocCount += miniBatchSize;

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

        if (printGamma) {
            System.out.println("gamma:");
            for (int d = 0; d < miniBatchSize; d++) {
                System.out.println(Arrays.toString(gamma_[d]));
            }
        }

        if (printPhi) {
            System.out.println("phi");
            for (int d = 0; d < miniBatchSize; d++) {
                for (String label : miniBatchMap.get(d).keySet()) {
                    System.out.println(Arrays.toString(phi_.get(d).get(label)));
                }
            }
        }
    }

    private void stepE() {
        // 1) Updating phi_

        // dirichlet_expectation_2d(gamma_)
        float[][] ElogTheta = new float[miniBatchSize][K_];
        for (int d = 0; d < miniBatchSize; d++) {
            float gammaSum_d = 0.f;
            for (int k = 0; k < K_; k++) {
                gammaSum_d += gamma_[d][k];
            }
            for (int k = 0; k < K_; k++) {
                ElogTheta[d][k] = (float) (Gamma.digamma(gamma_[d][k]) - Gamma.digamma(gammaSum_d));
            }
        }

        // dirichlet_expectation_2d(lambda_)
        HashMap<String, float[]> ElogBeta = new HashMap<String, float[]>();
        for (int k = 0; k < K_; k++) {
            float lambdaSum_k = 0.f;
            for (String label : lambda_.keySet()) {
                lambdaSum_k += lambda_.get(label)[k];
            }
            for (int d = 0; d < miniBatchSize; d++) {
                for (String label : miniBatchMap.get(d).keySet()) {
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
        for (int d = 0; d < miniBatchSize; d++) {
            for (String label : miniBatchMap.get(d).keySet()) {
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
        for (int d = 0; d < miniBatchSize; d++) {
            for (int k = 0; k < K_; k++) {
                float gamma_dk = alpha_;
                for (String label : miniBatchMap.get(d).keySet()) {
                    gamma_dk += phi_.get(d).get(label)[k] * miniBatchMap.get(d).get(label);
                }
                gamma_[d][k] = gamma_dk;
            }
        }
    }

    private void stepM() {
        // calculate lambdaBar
        HashMap<String, float[]> lambdaBar = new HashMap<String, float[]>();

        float docRatio = (float) D_ / (float) miniBatchSize;

        for (int d = 0; d < miniBatchSize; d++) {
            for (String label : miniBatchMap.get(d).keySet()) {
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
        for (String label : lambda_.keySet()) {
            float[] lambda_label = lambda_.get(label);
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
        miniBatchSize = miniBatch.length;

        for (int d = 0; d < miniBatchSize; d++) {
            accumWordCount = miniBatch[d].length;
        }
    }

    private void makeMiniBatchMap(String[][] miniBatch) {
        miniBatchMap = new ArrayList<HashMap<String, Float>>(); // initialize

        // parse document
        for (int d = 0; d < miniBatchSize; d++) {
            HashMap<String, Float> docMap = new HashMap<String, Float>();

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

            miniBatchMap.add(docMap);
        }
    }

    private void updateSizeOfParameterForMiniBatch() {
        phi_ = new ArrayList<HashMap<String, float[]>>();
        gamma_ = new float[miniBatchSize][];

        for (int d = 0; d < miniBatchSize; d++) {
            float[] gammad = getRandomGammaArray();
            gamma_[d] = gammad;

            // phi_ not needed to be initialized
            HashMap<String, float[]> phid = new HashMap<String, float[]>();
            for (String label : miniBatchMap.get(d).keySet()) {
                float[] tmpFArray = new float[K_];
                phid.put(label, tmpFArray);
            }
            phi_.add(phid);
        }

        // lambda
        for (int d = 0; d < miniBatchSize; d++) {
            for (String label : miniBatchMap.get(d).keySet()) {
                if (!lambda_.containsKey(label)) {
                    int tmpLambdaIdx = lambda_.size() % dummySize;
                    float[] lambdaNW = getRandomGammaArray();
                    for (int k = 0; k < K_; k++) {
                        lambdaNW[k] *= dummyLambdas[tmpLambdaIdx][k];
                    }
                    lambda_.put(label, lambdaNW);
                }
            }
        }
    }

    private float[] getRandomGammaArray() {
        double[] dret = gd.sample(K_);
        float[] ret = new float[K_];

        for (int k = 0; k < ret.length; k++) {
            ret[k] = (float) dret[k];
        }

        return ret;
    }

    private boolean checkGammaDiff(float[][] lastGamma, float[][] nextGamma) {
        double diff = 0.d;
        for (int d = 0; d < miniBatchSize; d++) {
            for (int k = 0; k < K_; k++) {
                diff += Math.abs(lastGamma[d][k] - nextGamma[d][k]);
            }
        }
        return (diff < DELTA * miniBatchSize * K_);
    }

    /**
     * Methods for debugging and checking convergence:
     */

    /**
     * Calculate approximate perplexity for the current mini-batch.
     */
    public float computePerplexity() {
        float bound = computeApproxBoundForMiniBatch();
        float perWordBound = bound / (float) accumWordCount;
        return (float) Math.exp(-1.f * perWordBound);
    }

    /**
     * Estimates the variational bound over all documents using only the documents passed as mini-batch.
     */
    private float computeApproxBoundForMiniBatch() {
        float score = 0.f;
        float tmp;

        // prepare
        float[] gammaSum = new float[miniBatchSize];
        Arrays.fill(gammaSum, 0.f);
        for (int d = 0; d < miniBatchSize; d++) {
            for (int k = 0; k < K_; k++) {
                gammaSum[d] += gamma_[d][k];
            }
        }
        float[] lambdaSum = new float[K_];
        Arrays.fill(lambdaSum, 0.f);
        for (int k = 0; k < K_; k++) {
            for (String label : lambda_.keySet()) {
                lambdaSum[k] = lambda_.get(label)[k];
            }
        }

        // E[log p(docs | theta, beta)]
        for (int d = 0; d < miniBatchSize; d++) {

            // for each word in the document
            for (String label : miniBatchMap.get(d).keySet()) {
                float wordCount = miniBatchMap.get(d).get(label);

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
                tmp /= accumDocCount;
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
            score += (tmp / miniBatchSize);

            float W = lambda_.size();
            tmp = (float) (Gamma.logGamma(K_ * alpha_))
                    - (float) (K_ * (Gamma.logGamma(alpha_)))
                    + (((float) (Gamma.logGamma(W * eta_)) - (float) (-1.f * W * (Gamma.logGamma(eta_)))) / accumDocCount);
            score += tmp;
        }

        return score;
    }

    public void showTopicWords() {
        System.out.println("SHOW TOPIC WORDS:");
        System.out.println("WORD SIZE:" + lambda_.size());
        for (int k = 0; k < K_; k++) {

            float lambdaSum = 0;
            for (String label : lambda_.keySet()) {
                lambdaSum += lambda_.get(label)[k];
            }

            System.out.print("Topic:" + k);

            System.out.println("===================================");
            ArrayList<String> sortedWords = getSortedLambda(k);
            System.out.println("k:" + k + " sortedWords.size():" + sortedWords.size());
            int topN = Math.min(50, lambda_.keySet().size());
            for (int tt = 0; tt < topN; tt++) {
                String label = sortedWords.get(tt);
                System.out.println("No." + tt + "\t" + label + "[" + label.length() + "]" + ":\t"
                        + lambda_.get(label)[k] / lambdaSum);
            }
            System.out.println("==========================================");
        }
    }

    private ArrayList<String> getSortedLambda(int k) {
        ArrayList<String> ret = new ArrayList<String>();
        ArrayList<LabelValueTuple> compareList = new ArrayList<LabelValueTuple>();

        for (String label : lambda_.keySet()) {
            float tmpValue = lambda_.get(label)[k];
            compareList.add(new LabelValueTuple(label, tmpValue));
        }

        Collections.sort(compareList, new LabelValueTupleComparator());

        for (int w = 0, W = compareList.size(); w < W; w++) {
            String label = compareList.get(w).getLabel();
            ret.add(label);
        }
        return ret;
    }

}
