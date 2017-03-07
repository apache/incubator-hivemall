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
package hivemall.matrix.ints;

import hivemall.matrix.VectorProcedure;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public interface IntMatrix {

    public boolean isSparse();

    public boolean readOnly();

    public void setDefaultValue(int value);

    @Nonnegative
    public int numRows();

    @Nonnegative
    public int numColumns();

    @Nonnull
    public int[] row();

    @Nonnull
    public int[] getRow(@Nonnegative int index);

    /**
     * @return returns dst
     */
    @Nonnull
    public int[] getRow(@Nonnegative int index, @Nonnull int[] dst);

    /**
     * @throws IndexOutOfBoundsException
     */
    public int get(@Nonnegative int row, @Nonnegative int col);

    /**
     * @throws IndexOutOfBoundsException
     */
    public int get(@Nonnegative int row, @Nonnegative int col, int defaultValue);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public void set(@Nonnegative int row, @Nonnegative int col, int value);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public int getAndSet(@Nonnegative int row, @Nonnegative int col, int value);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public void incr(@Nonnegative int row, @Nonnegative int col);

    /**
     * @throws IndexOutOfBoundsException
     * @throws UnsupportedOperationException
     */
    public void incr(@Nonnegative int row, @Nonnegative int col, int delta);

    public void eachInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure);

    public void eachNonZeroInRow(@Nonnegative int row, @Nonnull VectorProcedure procedure);

    public void eachInColumn(@Nonnegative int col, @Nonnull VectorProcedure procedure);

    public void eachInNonZeroColumn(@Nonnegative int col, @Nonnull VectorProcedure procedure);

}
