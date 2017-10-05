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
package hivemall.math.matrix.sparse;

import hivemall.math.matrix.sparse.floats.DoKFloatMatrix;

import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class DoKFloatMatrixTest {

    @Test
    public void testGetSet() {
        DoKFloatMatrix matrix = new DoKFloatMatrix();
        Random rnd = new Random(43);

        for (int i = 0; i < 1000; i++) {
            int row = Math.abs(rnd.nextInt());
            int col = Math.abs(rnd.nextInt());
            double v = rnd.nextDouble();
            matrix.set(row, col, v);
            Assert.assertEquals(v, matrix.get(row, col), 0.00001d);
        }

    }

}
