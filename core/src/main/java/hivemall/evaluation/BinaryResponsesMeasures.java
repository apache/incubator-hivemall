/*
 * Hivemall: Hive scalable Machine Learning Library
 *
 * Copyright (C) 2015 Makoto YUI
 * Copyright (C) 2013-2015 National Institute of Advanced Industrial Science and Technology (AIST)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package hivemall.evaluation;

import hivemall.utils.lang.Preconditions;

import java.util.List;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

/**
 * Utility class of various measures.
 *
 * See http://recsyswiki.com/wiki/Discounted_Cumulative_Gain
 */
public final class BinaryResponsesMeasures {

    private BinaryResponsesMeasures() {}

    public static double nDCG(@Nonnull final List<?> rankedList, @Nonnull final List<?> groundTruth) {
        double dcg = 0.d;
        double idcg = IDCG(groundTruth.size());

        for (int i = 0, n = rankedList.size(); i < n; i++) {
            Object item_id = rankedList.get(i);
            if (!groundTruth.contains(item_id)) {
                continue;
            }
            int rank = i + 1;
            dcg += Math.log(2) / Math.log(rank + 1);
        }

        return dcg / idcg;
    }

    /**
     * Computes the ideal DCG
     *
     * @param n the number of positive items
     * @return ideal DCG
     */
    public static double IDCG(final int n) {
        double idcg = 0.d;
        for (int i = 0; i < n; i++) {
            idcg += Math.log(2) / Math.log(i + 2);
        }
        return idcg;
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
