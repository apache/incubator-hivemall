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

public abstract class MatrixBuilder {

    protected final boolean readOnly;
    
    public MatrixBuilder(boolean readOnly) {
        this.readOnly = readOnly;
    }    
    
    public void nextRow(@Nonnull final double[] row) {
        for (int col = 0; col < row.length; col++) {
            nextColumn(col, row[col]);
        }
        nextRow();
    }

    public void nextRow(@Nonnull final String[] row) {
        for (String col : row) {
            if (col == null) {
                continue;
            }
            nextColumn(col);
        }
        nextRow();
    }

    @Nonnull
    public abstract MatrixBuilder nextRow();

    @Nonnull
    public abstract MatrixBuilder nextColumn(@Nonnegative int col, double value);

    /**
     * @throws IllegalArgumentException
     * @throws NumberFormatException
     */
    @Nonnull
    public MatrixBuilder nextColumn(@Nonnull final String col) {
        final int pos = col.indexOf(':');
        if (pos == 0) {
            throw new IllegalArgumentException("Invalid feature value representation: " + col);
        }

        final String feature;
        final double value;
        if (pos > 0) {
            feature = col.substring(0, pos);
            String s2 = col.substring(pos + 1);
            value = Double.parseDouble(s2);
        } else {
            feature = col;
            value = 1.d;
        }

        if (feature.indexOf(':') != -1) {
            throw new IllegalArgumentException("Invaliad feature format `<index>:<value>`: " + col);
        }

        int colIndex = Integer.parseInt(feature);
        if (colIndex < 0) {
            throw new IllegalArgumentException("Col index MUST be greather than or equals to 0: "
                    + colIndex);
        }

        return nextColumn(colIndex, value);
    }

    @Nonnull
    public abstract Matrix buildMatrix();

}
