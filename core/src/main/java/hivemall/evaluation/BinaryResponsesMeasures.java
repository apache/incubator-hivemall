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
package hivemall.evaluation;

import hivemall.utils.lang.Preconditions;
import hivemall.utils.math.MathUtils;

import java.util.List;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Binary responses measures for item recommendation (i.e. ranking problems)
 *
 * References: B. McFee and G. R. Lanckriet. "Metric Learning to Rank" ICML 2010.
 */
public final class BinaryResponsesMeasures {

    private BinaryResponsesMeasures() {}

    /**
     * Computes binary nDCG (i.e. relevance score is 0 or 1)
     *
     * @param rankedList a list of ranked item IDs (first item is highest-ranked)
     * @param groundTruth a collection of positive/correct item IDs
     * @param recommendSize top-`recommendSize` items in `rankedList` are recommended
     * @return nDCG
     */
    public static double nDCG(@Nonnull final List<?> rankedList,
            @Nonnull final List<?> groundTruth, @Nonnegative final int recommendSize) {
        Preconditions.checkArgument(recommendSize > 0);

        double dcg = 0.d;

        final int k = Math.min(rankedList.size(), recommendSize);
        for (int i = 0; i < k; i++) {
            Object item_id = rankedList.get(i);
            if (!groundTruth.contains(item_id)) {
                continue;
            }
            int rank = i + 1;
            dcg += 1.d / MathUtils.log2(rank + 1);
        }

        final double idcg = IDCG(Math.min(groundTruth.size(), k));
        if (idcg == 0.d) {
            return 0.d;
        }
        return dcg / idcg;
    }

    /**
     * Computes the ideal DCG
     * 
     * @param n the number of positive items
     * @return ideal DCG
     */
    public static double IDCG(@Nonnegative final int n) {
        Preconditions.checkArgument(n >= 0);

        double idcg = 0.d;
        for (int i = 0; i < n; i++) {
            idcg += 1.d / MathUtils.log2(i + 2);
        }
        return idcg;
    }

    /**
     * Computes Precision@`recommendSize`
     *
     * @param rankedList a list of ranked item IDs (first item is highest-ranked)
     * @param groundTruth a collection of positive/correct item IDs
     * @param recommendSize top-`recommendSize` items in `rankedList` are recommended
     * @return Precision
     */
    public static double Precision(@Nonnull final List<?> rankedList,
            @Nonnull final List<?> groundTruth, @Nonnegative final int recommendSize) {
        if (rankedList.isEmpty()) {
            if (groundTruth.isEmpty()) {
                return 1.d;
            }
            return 0.d;
        }

        Preconditions.checkArgument(recommendSize > 0); // can be zero when groundTruth is empty

        int nTruePositive = 0;
        final int k = Math.min(rankedList.size(), recommendSize);
        for (int i = 0; i < k; i++) {
            Object item_id = rankedList.get(i);
            if (groundTruth.contains(item_id)) {
                nTruePositive++;
            }
        }

        return ((double) nTruePositive) / k;
    }

    /**
     * Computes Recall@`recommendSize`
     *
     * @param rankedList a list of ranked item IDs (first item is highest-ranked)
     * @param groundTruth a collection of positive/correct item IDs
     * @param recommendSize top-`recommendSize` items in `rankedList` are recommended
     * @return Recall
     */
    public static double Recall(@Nonnull final List<?> rankedList,
            @Nonnull final List<?> groundTruth, @Nonnegative final int recommendSize) {
        if (groundTruth.isEmpty()) {
            if (rankedList.isEmpty()) {
                return 1.d;
            }
            return 0.d;
        }

        return ((double) TruePositives(rankedList, groundTruth, recommendSize))
                / groundTruth.size();
    }

    /**
     * Counts the number of true positives
     *
     * @param rankedList a list of ranked item IDs (first item is highest-ranked)
     * @param groundTruth a collection of positive/correct item IDs
     * @param recommendSize top-`recommendSize` items in `rankedList` are recommended
     * @return number of true positives
     */
    public static int TruePositives(final List<?> rankedList, final List<?> groundTruth,
            @Nonnegative final int recommendSize) {
        Preconditions.checkArgument(recommendSize > 0);

        int nTruePositive = 0;

        final int k = Math.min(rankedList.size(), recommendSize);
        for (int i = 0; i < k; i++) {
            Object item_id = rankedList.get(i);
            if (groundTruth.contains(item_id)) {
                nTruePositive++;
            }
        }

        return nTruePositive;
    }

    /**
     * Computes Reciprocal Rank
     *
     * @param rankedList a list of ranked item IDs (first item is highest-ranked)
     * @param groundTruth a collection of positive/correct item IDs
     * @param recommendSize top-`recommendSize` items in `rankedList` are recommended
     * @return Reciprocal Rank
     * @link https://en.wikipedia.org/wiki/Mean_reciprocal_rank
     */
    public static double ReciprocalRank(@Nonnull final List<?> rankedList,
            @Nonnull final List<?> groundTruth, @Nonnegative final int recommendSize) {
        Preconditions.checkArgument(recommendSize > 0);

        final int k = Math.min(rankedList.size(), recommendSize);
        for (int i = 0; i < k; i++) {
            Object item_id = rankedList.get(i);
            if (groundTruth.contains(item_id)) {
                return 1.d / (i + 1);
            }
        }

        return 0.d;
    }

    /**
     * Computes Average Precision (AP)
     *
     * @param rankedList a list of ranked item IDs (first item is highest-ranked)
     * @param groundTruth a collection of positive/correct item IDs
     * @param recommendSize top-`recommendSize` items in `rankedList` are recommended
     * @return AveragePrecision
     */
    public static double AveragePrecision(@Nonnull final List<?> rankedList,
            @Nonnull final List<?> groundTruth, @Nonnegative final int recommendSize) {
        Preconditions.checkArgument(recommendSize > 0);

        if (groundTruth.isEmpty()) {
            if (rankedList.isEmpty()) {
                return 1.d;
            }
            return 0.d;
        }

        int nTruePositive = 0;
        double sumPrecision = 0.d;

        // accumulate precision@1 to @recommendSize
        final int k = Math.min(rankedList.size(), recommendSize);
        for (int i = 0; i < k; i++) {
            Object item_id = rankedList.get(i);
            if (groundTruth.contains(item_id)) {
                nTruePositive++;
                sumPrecision += nTruePositive / (i + 1.d);
            }
        }

        if (nTruePositive == 0) {
            return 0.d;
        }
        return sumPrecision / nTruePositive;
    }

    /**
     * Computes the area under the ROC curve (AUC)
     *
     * @param rankedList a list of ranked item IDs (first item is highest-ranked)
     * @param groundTruth a collection of positive/correct item IDs
     * @param recommendSize top-`recommendSize` items in `rankedList` are recommended
     * @return AUC
     */
    public static double AUC(@Nonnull final List<?> rankedList, @Nonnull final List<?> groundTruth,
            @Nonnegative final int recommendSize) {
        Preconditions.checkArgument(recommendSize > 0);

        int nTruePositive = 0, nCorrectPairs = 0;

        // count # of pairs of items that are ranked in the correct order (i.e. TP > FP)
        final int k = Math.min(rankedList.size(), recommendSize);
        for (int i = 0; i < k; i++) {
            Object item_id = rankedList.get(i);
            if (groundTruth.contains(item_id)) {
                // # of true positives which are ranked higher position than i-th recommended item
                nTruePositive++;
            } else {
                // for each FP item, # of correct ordered <TP, FP> pairs equals to # of TPs at i-th position
                nCorrectPairs += nTruePositive;
            }
        }

        // # of all possible <TP, FP> pairs
        int nPairs = nTruePositive * (recommendSize - nTruePositive);

        // if there is no TP or no FP, it's meaningless for this metric (i.e., AUC=0.5)
        if (nPairs == 0) {
            return 0.5d;
        }

        // AUC can equivalently be calculated by counting the portion of correctly ordered pairs
        return ((double) nCorrectPairs) / nPairs;
    }

}
