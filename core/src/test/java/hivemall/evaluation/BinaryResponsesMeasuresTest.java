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

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class BinaryResponsesMeasuresTest {

    @Test
    public void testNDCG() {
        List<Integer> rankedList = Arrays.asList(1, 3, 2, 6);
        List<Integer> groundTruth = Arrays.asList(1, 2, 4);

        double actual = BinaryResponsesMeasures.nDCG(rankedList, groundTruth, rankedList.size());
        Assert.assertEquals(0.7039180890341348d, actual, 0.0001d);

        actual = BinaryResponsesMeasures.nDCG(rankedList, groundTruth, 2);
        Assert.assertEquals(0.6131471927654585d, actual, 0.0001d);
    }

    @Test
    public void testNDCG2() {
        List<Integer> rankedList = Arrays.asList(3, 2, 1, 6);
        List<Integer> groundTruth = Arrays.asList(1);

        double actual = BinaryResponsesMeasures.nDCG(rankedList, groundTruth, 2);
        Assert.assertEquals(0.d, actual, 0.0001d);

        actual = BinaryResponsesMeasures.nDCG(rankedList, groundTruth, 3);
        Assert.assertEquals(0.5d, actual, 0.0001d);
    }

    @Test
    public void testRecall() {
        List<Integer> rankedList = Arrays.asList(1, 3, 2, 6);
        List<Integer> groundTruth = Arrays.asList(1, 2, 4);

        double actual = BinaryResponsesMeasures.Recall(rankedList, groundTruth, rankedList.size());
        Assert.assertEquals(0.6666666666666666d, actual, 0.0001d);

        actual = BinaryResponsesMeasures.Recall(rankedList, groundTruth, 2);
        Assert.assertEquals(0.3333333333333333d, actual, 0.0001d);
    }

    @Test
    public void testRecallEmpty() {
        Assert.assertEquals(1.d,
            BinaryResponsesMeasures.Recall(Collections.emptyList(), Collections.emptyList(), 2),
            0.d);

        Assert.assertEquals(0.d,
            BinaryResponsesMeasures.Recall(Arrays.asList(1, 3, 2), Collections.emptyList(), 2), 0.d);
    }

    @Test
    public void testPrecision() {
        List<Integer> rankedList = Arrays.asList(1, 3, 2, 6);
        List<Integer> groundTruth = Arrays.asList(1, 2, 4);

        double actual = BinaryResponsesMeasures.Precision(rankedList, groundTruth,
            rankedList.size());
        Assert.assertEquals(0.5d, actual, 0.0001d);

        actual = BinaryResponsesMeasures.Precision(rankedList, groundTruth, 2);
        Assert.assertEquals(0.5d, actual, 0.0001d);
    }

    @Test
    public void testPrecisionEmpty() {
        Assert.assertEquals(1.d,
            BinaryResponsesMeasures.Precision(Collections.emptyList(), Collections.emptyList(), 2),
            0.d);

        Assert.assertEquals(0.d,
            BinaryResponsesMeasures.Precision(Arrays.asList(1, 3, 2), Collections.emptyList(), 2),
            0.d);
    }

    @Test
    public void testRR() {
        List<Integer> rankedList = Arrays.asList(1, 3, 2, 6);
        List<Integer> groundTruth = Arrays.asList(1, 2, 4);

        double actual = BinaryResponsesMeasures.ReciprocalRank(rankedList, groundTruth,
            rankedList.size());
        Assert.assertEquals(1.0d, actual, 0.0001d);

        Collections.reverse(rankedList);

        actual = BinaryResponsesMeasures.ReciprocalRank(rankedList, groundTruth, rankedList.size());
        Assert.assertEquals(0.5d, actual, 0.0001d);

        actual = BinaryResponsesMeasures.ReciprocalRank(rankedList, groundTruth, 1);
        Assert.assertEquals(0.0d, actual, 0.0001d);
    }

    @Test
    public void testAP() {
        List<Integer> rankedList = Arrays.asList(1, 3, 2, 6);
        List<Integer> groundTruth = Arrays.asList(1, 2, 4);

        double actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth,
            rankedList.size());
        Assert.assertEquals(1.0 / 2.0 * (1.0 / 1.0 + 2.0 / 3.0), actual, 0.0001d);

        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 4);
        Assert.assertEquals(1.0 / 2.0 * (1.0 / 1.0 + 2.0 / 3.0), actual, 0.0001d);

        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 3);
        Assert.assertEquals(1.0 / 2.0 * (1.0 / 1.0 + 2.0 / 3.0), actual, 0.0001d);

        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 2);
        Assert.assertEquals(1.0 / 1.0 * (1.0 / 1.0), actual, 0.0001d);

        rankedList = Arrays.asList(3, 1, 2, 6);
        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 2);
        Assert.assertEquals(1.0 / 1.0 * (1.0 / 2.0), actual, 0.0001d);

        groundTruth = Arrays.asList(1, 2, 3);
        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 2);
        Assert.assertEquals(1.0 / 2.0 * (1.0 / 1.0 + 2.0 / 2.0), actual, 0.0001d);

        rankedList = Arrays.asList(3, 1);
        groundTruth = Arrays.asList(1, 2);
        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 2);
        Assert.assertEquals(1.0 / 1.0 * (1.0 / 2.0), actual, 0.0001d);
    }

    @Test
    public void testAPString() {
        List<String> rankedList = Arrays.asList("a", "b", "c", "d", "e", "f", "g");
        List<String> groundTruth = Arrays.asList("a", "x", "x", "d", "x", "x");

        double actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 6);
        Assert.assertEquals(0.75d, actual, 0.0001d);
    }

    @Test
    public void testAPString10() {
        List<String> rankedList = Arrays.asList("a", "b", "c", "d", "e", "f", "g", "h", "i", "j");
        List<String> groundTruth = Arrays.asList("a", "x", "c", "x", "e", "f");

        double actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 10);
        Assert.assertEquals(1.0 / 4.0 * (1.0 / 1.0 + 2.0 / 3.0 + 3.0 / 5.0 + 4.0 / 6.0), actual,
            0.0001d);

        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 5);
        Assert.assertEquals(1.0 / 3.0 * (1.0 / 1.0 + 2.0 / 3.0 + 3.0 / 5.0), actual, 0.0001d);

        groundTruth = Arrays.asList("a", "x", "c", "x", "e", "f", "x", "x", "x", "x");
        actual = BinaryResponsesMeasures.AveragePrecision(rankedList, groundTruth, 10);
        Assert.assertEquals(1.0 / 4.0 * (1.0 / 1.0 + 2.0 / 3.0 + 3.0 / 5.0 + 4.0 / 6.0), actual,
            0.0001d);
    }

    @Test
    public void testAUC() {
        List<Integer> rankedList = Arrays.asList(1, 3, 2, 6);
        List<Integer> groundTruth = Arrays.asList(1, 2, 4);

        double actual = BinaryResponsesMeasures.AUC(rankedList, groundTruth, rankedList.size());
        Assert.assertEquals(0.75d, actual, 0.0001d);

        actual = BinaryResponsesMeasures.AUC(rankedList, groundTruth, 2);
        Assert.assertEquals(1.0d, actual, 0.0001d);

        // meaningless case I: all TPs
        List<Integer> groundTruthAllTruePositive = Arrays.asList(1, 3, 2, 6);
        actual = BinaryResponsesMeasures.AUC(rankedList, groundTruthAllTruePositive,
            rankedList.size());
        Assert.assertEquals(0.5d, actual, 0.0001d);

        // meaningless case II: all FPs
        List<Integer> groundTruthAllFalsePositive = Arrays.asList(7, 8, 9, 10);
        actual = BinaryResponsesMeasures.AUC(rankedList, groundTruthAllFalsePositive,
            rankedList.size());
        Assert.assertEquals(0.5d, actual, 0.0001d);
    }

}
