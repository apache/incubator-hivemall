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

import hivemall.math.vector.VectorProcedure;
import hivemall.utils.lang.Primitives;

import java.util.HashSet;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class DoKMatrixTest {

    @Test
    public void testGetSet() {
        DoKMatrix matrix = new DoKMatrix();
        Random rnd = new Random(43);

        for (int i = 0; i < 1000; i++) {
            int row = Math.abs(rnd.nextInt());
            int col = Math.abs(rnd.nextInt());
            double v = rnd.nextDouble();
            matrix.set(row, col, v);
            Assert.assertEquals(v, matrix.get(row, col), 0.00001d);
        }
    }

    @Test
    public void testNumRowsNumCols() {
        DoKMatrix matrix = new DoKMatrix();
        Random rnd = new Random(43);
        HashSet<Long> bitset = new HashSet<>(100000);

        int numRows = -1, numCols = -1;
        for (int i = 0; i < 100000; i++) {
            int row = Math.abs(rnd.nextInt());
            int col = Math.abs(rnd.nextInt());
            numRows = Math.max(row + 1, numRows);
            numCols = Math.max(col + 1, numCols);
            double v = rnd.nextDouble();
            if (v >= 0.8) {
                v = 0.d;
            }
            matrix.getAndSet(row, col, v);
            bitset.add(Primitives.toLong(row, col));
            Assert.assertEquals(v, matrix.get(row, col), 0.00001d);
        }

        Assert.assertEquals(numRows, matrix.numRows());
        Assert.assertEquals(numCols, matrix.numColumns());
        Assert.assertEquals(bitset.size(), matrix.nnz());
    }

    @Test
    public void testEmpty() {
        DoKMatrix matrix = new DoKMatrix();
        matrix.eachNonZeroCell(new VectorProcedure() {
            @Override
            public void apply(int i, int j, double value) {
                Assert.fail("should not be called");
            }
        });
    }

}
