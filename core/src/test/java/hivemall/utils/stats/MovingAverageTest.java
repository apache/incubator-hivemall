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
package hivemall.utils.stats;

import hivemall.utils.stats.MovingAverage;

import org.junit.Assert;
import org.junit.Test;

public class MovingAverageTest {

    @Test
    public void testAdd() {
        MovingAverage movingAvg = new MovingAverage(3);
        Assert.assertEquals(0.d, movingAvg.get(), 0.d);
        Assert.assertEquals(1.d, movingAvg.add(1.d), 0.d);
        Assert.assertEquals(1.5d, movingAvg.add(2.d), 0.d); // (1+2)/2 = 1.5
        Assert.assertEquals(2.d, movingAvg.add(3.d), 0.d); // (1+2+3)/3 = 2
        Assert.assertEquals(3.d, movingAvg.add(4.d), 0.d); // (2+3+4)/3 = 3
        Assert.assertEquals(4.d, movingAvg.add(5.d), 0.d); // (3+4+5)/3 = 4
        Assert.assertEquals(5.d, movingAvg.add(6.d), 0.d); // (4+5+6)/3 = 5
        Assert.assertEquals(6.d, movingAvg.add(7.d), 0.d); // (5+6+7)/3 = 6
    }

    @Test(expected = IllegalArgumentException.class)
    public void testNaN() {
        MovingAverage movingAvg = new MovingAverage(3);
        movingAvg.add(Double.NaN);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testInfinity() {
        MovingAverage movingAvg = new MovingAverage(3);
        movingAvg.add(Double.POSITIVE_INFINITY);
    }

}
