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
package hivemall.ftvec.binning;

import static java.lang.Double.NEGATIVE_INFINITY;
import static java.lang.Double.NaN;
import static java.lang.Double.POSITIVE_INFINITY;
import static org.junit.Assert.assertEquals;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Test;

public class FeatureBinningUDFTest {

    @Test
    public void testNaN() throws HiveException {
        // If num_bins = 3, the bins become something like  [-Inf, 1], (1, 10], (10, Inf]. 
        final double[] bin = new double[] {NEGATIVE_INFINITY, 1.d, 10.d, POSITIVE_INFINITY};
        assertEquals(2, FeatureBinningUDF.findBin(bin, POSITIVE_INFINITY));
        assertEquals(3, FeatureBinningUDF.findBin(bin, NaN));
    }

    @Test
    public void test3Bins() throws HiveException {
        // If num_bins = 3, the bins become something like  [-Inf, 1], (1, 10], (10, Inf]. 
        final double[] bin = new double[] {NEGATIVE_INFINITY, 1.d, 10.d, POSITIVE_INFINITY};
        assertEquals(0, FeatureBinningUDF.findBin(bin, NEGATIVE_INFINITY));
        assertEquals(0, FeatureBinningUDF.findBin(bin, 1.d));

        assertEquals(1, FeatureBinningUDF.findBin(bin, 1.1d));
        assertEquals(1, FeatureBinningUDF.findBin(bin, 10.d));

        assertEquals(2, FeatureBinningUDF.findBin(bin, 10.1d));
        assertEquals(2, FeatureBinningUDF.findBin(bin, POSITIVE_INFINITY));
    }

    @Test
    public void test4Bins() throws HiveException {
        // If num_bins = 4, the bins become something like [-Inf, 0.111], (0.111, 0.222], (0.222, 0.333], (0.333, Inf]. 
        final double[] bin =
                new double[] {NEGATIVE_INFINITY, 0.111d, 0.222d, 0.333d, POSITIVE_INFINITY};
        assertEquals(0, FeatureBinningUDF.findBin(bin, NEGATIVE_INFINITY));
        assertEquals(0, FeatureBinningUDF.findBin(bin, -1.d));
        assertEquals(0, FeatureBinningUDF.findBin(bin, 0.110d));
        assertEquals(0, FeatureBinningUDF.findBin(bin, 0.111d));

        assertEquals(1, FeatureBinningUDF.findBin(bin, 0.112d));
        assertEquals(1, FeatureBinningUDF.findBin(bin, 0.2d));
        assertEquals(1, FeatureBinningUDF.findBin(bin, 0.222d));
        assertEquals(1, FeatureBinningUDF.findBin(bin, 0.2220d));

        assertEquals(2, FeatureBinningUDF.findBin(bin, 0.223d));
        assertEquals(2, FeatureBinningUDF.findBin(bin, 0.3d));
        assertEquals(2, FeatureBinningUDF.findBin(bin, 0.332d));
        assertEquals(2, FeatureBinningUDF.findBin(bin, 0.333d));

        assertEquals(3, FeatureBinningUDF.findBin(bin, 0.334d));
        assertEquals(3, FeatureBinningUDF.findBin(bin, 0.4d));
        assertEquals(3, FeatureBinningUDF.findBin(bin, 10000d));
        assertEquals(3, FeatureBinningUDF.findBin(bin, POSITIVE_INFINITY));

        assertEquals(4, FeatureBinningUDF.findBin(bin, NaN));
    }

}
