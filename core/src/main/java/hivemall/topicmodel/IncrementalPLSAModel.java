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

import java.util.List;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Map;
import java.util.HashMap;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.Random;

import javax.annotation.Nonnull;

public final class IncrementalPLSAModel {

    // number of topics
    private int K_;

    // control how much P(w|z) updating is affected by the last value
    private float alpha_ = 0.5f;

    // check convergence of P(w|z) for a document
    private double delta_ = 1E-5d;

    // random number generator
    private final Random rng_;

    // optimized in the E step
    private List<Map<String, float[]>> p_dwz_; // P(z|d,w) probability of topics for each document-word pair

    // optimized in the M step
    private List<float[]> p_dz_; // P(z|d) probability of topics for documents
    private Map<String, float[]> p_zw_; // P(w|z) probability of words for each topic

    private List<Map<String, Float>> miniBatchMap_;
    private int miniBatchSize_;

    public IncrementalPLSAModel(int K, float alpha, double delta) {
        this.K_ = K;
        this.alpha_ = alpha;
        this.delta_ = delta;

        this.rng_ = new Random(1001);

        this.p_zw_ = new HashMap<String, float[]>();
    }

    public void train(@Nonnull String[][] miniBatch) {
        miniBatchSize_ = miniBatch.length;

        makeMiniBatchMap(miniBatch);

        initParams();

        Map<String, float[]> p_zw_prev = new HashMap<String, float[]>();

        for (int d = 0; d < miniBatchSize_; d++) {
            do {
                p_zw_prev.clear();
                p_zw_prev.putAll(p_zw_);

                // Expectation
                stepE(d);

                // Maximization
                stepM(d);
            } while (!checkPzwDiff(d, p_zw_prev, p_zw_));
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

    private void initParams() {
        p_dz_ = new ArrayList<float[]>();
        p_dwz_ = new ArrayList<Map<String, float[]>>();

        for (int d = 0; d < miniBatchSize_; d++) {
            // init P(z|d)
            float[] p_dz_d = new float[K_];
            float p_dz_dSum = 0.f;
            for (int z = 0; z < K_; z++) {
                p_dz_d[z] = rng_.nextFloat();
                p_dz_dSum += p_dz_d[z];
            }
            for (int z = 0; z < K_; z++) { // normalize
                p_dz_d[z] /= p_dz_dSum;
            }
            p_dz_.add(p_dz_d);

            Map<String, Float> docMap = miniBatchMap_.get(d);

            // init P(z|d,w)
            Map<String, float[]> p_dwz_d = new HashMap<String, float[]>();
            for (String word : docMap.keySet()) {
                float[] p_dwz_dw = new float[K_];
                float p_dwz_dwSum = 0.f;
                for (int z = 0; z < K_; z++) {
                    p_dwz_dw[z] = rng_.nextFloat();
                    p_dwz_dwSum += p_dwz_dw[z];
                }
                for (int z = 0; z < K_; z++) { // normalize
                    p_dwz_dw[z] /= p_dwz_dwSum;
                }
                p_dwz_d.put(word, p_dwz_dw);
            }
            p_dwz_.add(p_dwz_d);


            // insert new words to P(w|z)
            for (String word : docMap.keySet()) {
                float[] p_zw_w = new float[K_];
                float p_zw_wSum = 0.f;
                for (int z = 0; z < K_; z++) {
                    if (!p_zw_.containsKey(word)) {
                        p_zw_w[z] = rng_.nextFloat();
                    } else {
                        p_zw_w[z] = p_zw_.get(word)[z];
                    }
                    p_zw_wSum += p_zw_w[z];
                }
                for (int z = 0; z < K_; z++) { // normalize
                    p_zw_w[z] /= p_zw_wSum;
                }
                p_zw_.put(word, p_zw_w);
            }
        }
    }

    private void stepE(int d) {
        // updating P(z|d,w)
        Map<String, float[]> p_dwz_d = new HashMap<String, float[]>();
        for (String word : miniBatchMap_.get(d).keySet()) {
            float[] p_dwz_dw = p_dwz_.get(d).get(word);
            float p_dwz_dwSum = 0.f;
            for (int z = 0; z < K_; z++) {
                p_dwz_dw[z] = p_dz_.get(d)[z] * p_zw_.get(word)[z];
                p_dwz_dwSum += p_dwz_dw[z];
            }
            for (int z = 0; z < K_; z++) { // normalize over the topics
                p_dwz_dw[z] /= p_dwz_dwSum;
            }
            p_dwz_d.put(word, p_dwz_dw);
        }
        p_dwz_.set(d, p_dwz_d);
    }

    private void stepM(int d) {
        Map<String, Float> docMap = miniBatchMap_.get(d);

        // updating P(z|d)
        float p_dz_dSum = 0.f;
        float[] p_dz_d = new float[K_];
        for (int z = 0; z < K_; z++) {
            p_dz_d[z] = 0.f;
            for (Map.Entry<String, Float> e : docMap.entrySet()) {
                String word = e.getKey();
                float n = e.getValue();
                p_dz_d[z] += n * p_dwz_.get(d).get(word)[z];
            }
            p_dz_dSum += p_dz_d[z];
        }
        for (int z = 0; z < K_; z++) { // normalize over the topics
            p_dz_d[z] /= p_dz_dSum;
        }
        p_dz_.set(d, p_dz_d);

        // updating P(w|z)
        for (int z = 0; z < K_; z++) {
            float p_zw_wSumInDoc = 0.f; // over the words in the document
            float p_zw_wSumAll = 0.f; // over the all existing words

            for (Map.Entry<String, float[]> e : p_zw_.entrySet()) {
                String word = e.getKey();
                float[] p_zw_w = e.getValue();

                p_zw_wSumAll += p_zw_w[z];

                p_zw_w[z] *= alpha_;

                if (docMap.containsKey(word)) {
                    float term = docMap.get(word) * p_dwz_.get(d).get(word)[z];
                    p_zw_w[z] += term;
                    p_zw_wSumInDoc += term;
                }
            }

            float normalizer = p_zw_wSumInDoc + alpha_ * p_zw_wSumAll; // normalize over the all words
            for (Map.Entry<String, float[]> e : p_zw_.entrySet()) {
                String word = e.getKey();
                float[] p_zw_w = p_zw_.get(word);
                p_zw_w[z] /= normalizer;
                p_zw_.put(word, p_zw_w);
            }
        }
    }

    private boolean checkPzwDiff(int d, @Nonnull Map<String, float[]> p_zw_prev, @Nonnull Map<String, float[]> p_zw) {
        double diff = 0.d;
        for (int z = 0; z < K_; z++) {
            for (String word : miniBatchMap_.get(d).keySet()) {
                diff += Math.abs(p_zw_prev.get(word)[z] - p_zw.get(word)[z]);
            }
        }
        return (diff / K_) < delta_;
    }

    public float computePerplexity() {
        float numer = 0.f;
        float denom = 0.f;
        for (int d = 0; d < miniBatchSize_; d++) {
            for (Map.Entry<String, Float> e : miniBatchMap_.get(d).entrySet()) {
                String word = e.getKey();
                float value = e.getValue();

                float p_dw = 0.f;
                float[] p_zw_w = p_zw_.get(word);
                for (int z = 0; z < K_; z++) {
                    p_dw += p_zw_w[z] * p_dz_.get(d)[z];
                }

                denom += value;
                numer += value * (float) Math.log(p_dw);
            }
        }

        return (float) Math.exp(-1.f * (numer / denom));
    }

    public SortedMap<Float, List<String>> getTopicWords(int z) {
        SortedMap<Float, List<String>> res = new TreeMap<Float, List<String>>(Collections.reverseOrder());
        for (Map.Entry<String, float[]> e : p_zw_.entrySet()) {
            String word = e.getKey();
            float[] probs = e.getValue();

            List<String> words = new ArrayList<String>();
            if (res.containsKey(probs[z])) {
                words = res.get(probs[z]);
            }
            words.add(word);
            res.put(probs[z], words);
        }
        return res;
    }

    public float[] getTopicDistribution(@Nonnull String[] doc) {
        train(new String[][] {doc});
        return p_dz_.get(0);
    }

    public float getProbability(String word, int z) {
        return p_zw_.get(word)[z];
    }

    public void setProbability(String word, int z, float prob) {
        float[] prob_word;

        if (!p_zw_.containsKey(word)) {
            prob_word = new float[K_];
            for (int k = 0; k < K_; k++) {
                prob_word[k] = rng_.nextFloat();
            }
        } else {
            prob_word = p_zw_.get(word);
        }

        prob_word[z] = prob;

        // normalize
        float sum = 0.f;
        for (int k = 0; k < K_; k++) {
            sum += prob_word[k];
        }
        for (int k = 0; k < K_; k++) {
            prob_word[k] /= sum;
        }

        p_zw_.put(word, prob_word);
    }
}
