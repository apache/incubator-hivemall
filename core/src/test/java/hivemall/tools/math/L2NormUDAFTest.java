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
package hivemall.tools.math;

import org.apache.hadoop.hive.serde2.io.DoubleWritable;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class L2NormUDAFTest {

    L2NormUDAF.Evaluator evaluator;
    double[] x;
    double expected;

    @Before
    public void setUp() throws Exception {
        this.evaluator = new L2NormUDAF.Evaluator();
        this.x = new double[] {1.d, 2.d, 3.d, 4.d, 5.d, 6.d};
        this.expected = 9.5393920141694561;
    }

    @Test
    public void test() throws Exception {
        evaluator.init();

        for (double xi : x) {
            evaluator.iterate(new DoubleWritable(xi));
        }

        Assert.assertEquals(expected, evaluator.terminate(), 1e-5d);
    }

    @Test
    public void testMerge() throws Exception {
        L2NormUDAF.PartialResult[] partials = new L2NormUDAF.PartialResult[3];

        // bin #1
        evaluator.init();
        evaluator.iterate(new DoubleWritable(x[0]));
        evaluator.iterate(new DoubleWritable(x[1]));
        partials[0] = evaluator.terminatePartial();

        // bin #2
        evaluator.init();
        evaluator.iterate(new DoubleWritable(x[2]));
        evaluator.iterate(new DoubleWritable(x[3]));
        partials[1] = evaluator.terminatePartial();

        // bin #3
        evaluator.init();
        evaluator.iterate(new DoubleWritable(x[4]));
        evaluator.iterate(new DoubleWritable(x[5]));
        partials[2] = evaluator.terminatePartial();

        // merge in a different order; e.g., <bin0, bin1>, <bin1, bin0> should return same value
        final int[][] orders =
                new int[][] {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 1, 0}, {2, 0, 1}};
        for (int i = 0; i < orders.length; i++) {
            evaluator.init();

            evaluator.merge(partials[orders[i][0]]);
            evaluator.merge(partials[orders[i][1]]);
            evaluator.merge(partials[orders[i][2]]);

            Assert.assertEquals(expected, evaluator.terminate(), 1e-5d);
        }
    }
}
