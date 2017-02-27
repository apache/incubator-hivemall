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
package hivemall.matrix;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public interface Matrix {

    public boolean readOnly();

    public boolean swappable();

    public void setDefaultValue(double value);

    @Nonnegative
    public int numRows();

    @Nonnegative
    public int numColumns();

    @Nonnegative
    public int numColumns(@Nonnegative int row);

    @Nonnull
    public double[] row();

    @Nonnull
    public double[] getRow(@Nonnegative int index);

    /**
     * @return returns dst
     */
    @Nonnull
    public double[] getRow(@Nonnegative int index, @Nonnull double[] dst);

    /**
     * @throws IndexOutOfBoundsException
     */
    public double get(@Nonnegative int row, @Nonnegative int col);

    /**
     * @throws IndexOutOfBoundsException
     */
    public double get(@Nonnegative int row, @Nonnegative int col, double defaultValue);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public void set(@Nonnegative int row, @Nonnegative int col, double value);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public double getAndSet(@Nonnegative int row, @Nonnegative int col, double value);

    public void swap(@Nonnegative int row1, @Nonnegative int row2);

    public void eachInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure);

    public void eachNonZeroInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure);

    @Nonnull
    public MatrixBuilder builder();

}
