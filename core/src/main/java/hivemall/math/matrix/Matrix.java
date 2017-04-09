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
package hivemall.math.matrix;

import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.math.vector.Vector;
import hivemall.math.vector.VectorProcedure;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.concurrent.NotThreadSafe;

/**
 * Double matrix.
 */
@NotThreadSafe
public interface Matrix {

    public boolean isSparse();

    public boolean isRowMajorMatrix();

    public boolean isColumnMajorMatrix();

    public boolean readOnly();

    public boolean swappable();

    /** The Number of Non-Zeros */
    public int nnz();

    @Nonnegative
    public int numRows();

    @Nonnegative
    public int numColumns();

    @Nonnegative
    public int numColumns(@Nonnegative int row);

    @Nonnull
    public double[] row();

    @Nonnull
    public Vector rowVector();

    @Nonnull
    public double[] getRow(@Nonnegative int index);

    /**
     * @return returns dst
     */
    @Nonnull
    public double[] getRow(@Nonnegative int index, @Nonnull double[] dst);

    public void getRow(@Nonnegative int index, @Nonnull Vector row);

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

    public void eachInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure,
            boolean nullOutput);

    public void eachNonNullInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure);

    public void eachNonZeroInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure);

    public void eachColumnIndexInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure);

    public void eachInColumn(@Nonnegative int col, @Nonnull VectorProcedure procedure);

    public void eachInColumn(@Nonnegative int col, @Nonnull VectorProcedure procedure,
            boolean nullOutput);

    public void eachNonNullInColumn(@Nonnegative int col, @Nonnull VectorProcedure procedure);

    public void eachNonZeroInColumn(@Nonnegative int col, @Nonnull VectorProcedure procedure);

    @Nonnull
    public RowMajorMatrix toRowMajorMatrix();

    @Nonnull
    public ColumnMajorMatrix toColumnMajorMatrix();

    @Nonnull
    public MatrixBuilder builder();

}
