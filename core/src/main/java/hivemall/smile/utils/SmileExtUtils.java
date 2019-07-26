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
package hivemall.smile.utils;

import hivemall.annotations.VisibleForTesting;
import hivemall.math.matrix.ColumnMajorMatrix;
import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.MatrixUtils;
import hivemall.math.matrix.ints.ColumnMajorDenseIntMatrix2d;
import hivemall.math.matrix.ints.ColumnMajorIntMatrix;
import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.math.vector.VectorProcedure;
import hivemall.smile.classification.DecisionTree.SplitRule;
import hivemall.utils.collections.lists.DoubleArrayList;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.math.MathUtils;
import smile.data.NominalAttribute;
import smile.data.NumericAttribute;
import smile.sort.QuickSort;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.roaringbitmap.RoaringBitmap;

public final class SmileExtUtils {
    public static final byte NUMERIC = (byte) 1;
    public static final byte NOMINAL = (byte) 2;

    private SmileExtUtils() {}

    /**
     * Q for {@link NumericAttribute}, C for {@link NominalAttribute}.
     */
    @Nonnull
    public static RoaringBitmap resolveAttributes(@Nullable final String opt)
            throws UDFArgumentException {
        final RoaringBitmap attr = new RoaringBitmap();
        if (opt == null) {
            return attr;
        }
        final String[] opts = opt.split(",");
        final int size = opts.length;
        int numCount = 0, qcCount = 0;
        for (int i = 0; i < size; i++) {
            final String type = opts[i];
            if ("Q".equals(type)) {
                qcCount++;
                continue;
            } else if ("C".equals(type)) {
                qcCount++;
                attr.add(i);
            } else if (NumberUtils.isDigits(type)) {
                numCount++;
                int index = NumberUtils.parseInt(type);
                attr.add(index);
            } else {
                throw new UDFArgumentException("Unsupported attribute type: " + type);
            }
        }
        if (qcCount != size && numCount != size) {
            throw new UDFArgumentException("Invalid attribute expression: " + opt);
        }
        return attr;
    }

    @VisibleForTesting
    @Nonnull
    public static RoaringBitmap convertAttributeTypes(
            @Nonnull final smile.data.Attribute[] original) {
        final int size = original.length;
        final RoaringBitmap nominalAttrs = new RoaringBitmap();
        for (int i = 0; i < size; i++) {
            smile.data.Attribute o = original[i];
            switch (o.type) {
                case NOMINAL: {
                    nominalAttrs.add(i);
                    break;
                }
                case NUMERIC: {
                    break;
                }
                default:
                    throw new UnsupportedOperationException("Unsupported type: " + o.type);
            }
        }
        return nominalAttrs;
    }

    @Nonnull
    public static ColumnMajorIntMatrix sort(@Nonnull final RoaringBitmap attributes,
            @Nonnull final Matrix x) {
        final int n = x.numRows();
        final int p = x.numColumns();

        final int[][] index = new int[p][];
        if (x.isSparse()) {
            int initSize = n / 10;
            final DoubleArrayList dlist = new DoubleArrayList(initSize);
            final IntArrayList ilist = new IntArrayList(initSize);
            final VectorProcedure proc = new VectorProcedure() {
                @Override
                public void apply(final int i, final double v) {
                    dlist.add(v);
                    ilist.add(i);
                }
            };

            final ColumnMajorMatrix x2 = x.toColumnMajorMatrix();
            for (int j = 0; j < p; j++) {
                if (attributes.contains(j)) {
                    continue; // nop for categorical columns
                }
                // sort only numerical columns
                x2.eachNonNullInColumn(j, proc);
                if (ilist.isEmpty()) {
                    continue;
                }
                int[] indexJ = ilist.toArray();
                QuickSort.sort(dlist.array(), indexJ, indexJ.length);
                index[j] = indexJ;
                dlist.clear();
                ilist.clear();
            }
        } else {
            final double[] a = new double[n];
            for (int j = 0; j < p; j++) {
                if (attributes.contains(j)) {
                    continue; // nop for categorical columns
                }
                // sort only numerical columns
                for (int i = 0; i < n; i++) {
                    a[i] = x.get(i, j);
                }
                index[j] = QuickSort.sort(a);
            }
        }

        return new ColumnMajorDenseIntMatrix2d(index, n);
    }

    @Nonnull
    public static int[] classLabels(@Nonnull final int[] y) throws HiveException {
        final int[] labels = smile.math.Math.unique(y);
        Arrays.sort(labels);

        if (labels.length < 2) {
            throw new HiveException("Only one class.");
        }
        for (int i = 0; i < labels.length; i++) {
            if (labels[i] < 0) {
                throw new HiveException("Negative class label: " + labels[i]);
            }
            if (i > 0 && (labels[i] - labels[i - 1]) > 1) {
                throw new HiveException("Missing class: " + (labels[i - 1] + 1));
            }
        }

        return labels;
    }

    @Nonnull
    public static SplitRule resolveSplitRule(@Nullable String ruleName) {
        if ("gini".equalsIgnoreCase(ruleName)) {
            return SplitRule.GINI;
        } else if ("entropy".equalsIgnoreCase(ruleName)) {
            return SplitRule.ENTROPY;
        } else if ("classification_error".equalsIgnoreCase(ruleName)) {
            return SplitRule.CLASSIFICATION_ERROR;
        } else {
            return SplitRule.GINI;
        }
    }

    public static int computeNumInputVars(final float numVars, @Nonnull final Matrix x) {
        final int numInputVars;
        if (numVars <= 0.f) {
            int dims = x.numColumns();
            numInputVars = (int) Math.ceil(Math.sqrt(dims));
        } else if (numVars > 0.f && numVars <= 1.f) {
            numInputVars = (int) (numVars * x.numColumns());
        } else {
            numInputVars = (int) numVars;
        }
        return numInputVars;
    }

    public static long generateSeed() {
        return Thread.currentThread().getId() * System.nanoTime();
    }

    public static void shuffle(@Nonnull final int[] x, @Nonnull final PRNG rnd) {
        for (int i = x.length; i > 1; i--) {
            int j = rnd.nextInt(i);
            swap(x, i - 1, j);
        }
    }

    @Nonnull
    public static Matrix shuffle(@Nonnull final Matrix x, @Nonnull final int[] y, long seed) {
        final int numRows = x.numRows();
        if (numRows != y.length) {
            throw new IllegalArgumentException(
                "x.length (" + numRows + ") != y.length (" + y.length + ')');
        }
        if (seed == -1L) {
            seed = generateSeed();
        }

        final PRNG rnd = RandomNumberGeneratorFactory.createPRNG(seed);
        if (x.swappable()) {
            for (int i = numRows; i > 1; i--) {
                int j = rnd.nextInt(i);
                int k = i - 1;
                x.swap(k, j);
                swap(y, k, j);
            }
            return x;
        } else {
            final int[] indices = MathUtils.permutation(numRows);
            for (int i = numRows; i > 1; i--) {
                int j = rnd.nextInt(i);
                int k = i - 1;
                swap(indices, k, j);
                swap(y, k, j);
            }
            return MatrixUtils.shuffle(x, indices);
        }
    }

    @Nonnull
    public static Matrix shuffle(@Nonnull final Matrix x, @Nonnull final double[] y,
            @Nonnull long seed) {
        final int numRows = x.numRows();
        if (numRows != y.length) {
            throw new IllegalArgumentException(
                "x.length (" + numRows + ") != y.length (" + y.length + ')');
        }
        if (seed == -1L) {
            seed = generateSeed();
        }

        final PRNG rnd = RandomNumberGeneratorFactory.createPRNG(seed);
        if (x.swappable()) {
            for (int i = numRows; i > 1; i--) {
                int j = rnd.nextInt(i);
                int k = i - 1;
                x.swap(k, j);
                swap(y, k, j);
            }
            return x;
        } else {
            final int[] indices = MathUtils.permutation(numRows);
            for (int i = numRows; i > 1; i--) {
                int j = rnd.nextInt(i);
                int k = i - 1;
                swap(indices, k, j);
                swap(y, k, j);
            }
            return MatrixUtils.shuffle(x, indices);
        }
    }

    /**
     * Swap two elements of an array.
     */
    private static void swap(final int[] x, final int i, final int j) {
        int s = x[i];
        x[i] = x[j];
        x[j] = s;
    }

    /**
     * Swap two elements of an array.
     */
    private static void swap(final double[] x, final int i, final int j) {
        double s = x[i];
        x[i] = x[j];
        x[j] = s;
    }

    @Nonnull
    public static int[] bagsToSamples(@Nonnull final int[] bags) {
        int maxIndex = -1;
        for (int e : bags) {
            if (e > maxIndex) {
                maxIndex = e;
            }
        }
        return bagsToSamples(bags, maxIndex + 1);
    }

    @Nonnull
    public static int[] bagsToSamples(@Nonnull final int[] bags, final int samplesLength) {
        final int[] samples = new int[samplesLength];
        for (int i = 0, size = bags.length; i < size; i++) {
            samples[bags[i]]++;
        }
        return samples;
    }

    public static boolean containsNumericType(@Nonnull final Matrix x,
            final RoaringBitmap attributes) {
        int numColumns = x.numColumns();
        int numCategoricalCols = attributes.getCardinality();
        return numColumns != numCategoricalCols; // contains at least one numerical column
    }

    @Nonnull
    public static String resolveFeatureName(final int index, @Nullable final String[] names) {
        if (names == null) {
            return "feature#" + index;
        }
        if (index >= names.length) {
            return "feature#" + index;
        }
        return names[index];
    }

    @Nonnull
    public static String resolveName(final int index, @Nullable final String[] names) {
        if (names == null) {
            return String.valueOf(index);
        }
        if (index >= names.length) {
            return String.valueOf(index);
        }
        return names[index];
    }

    /**
     * Generates an evenly distributed range of hue values in the HSV color scale.
     *
     * @return colors
     */
    public static double[] getColorBrew(@Nonnegative int n) {
        Preconditions.checkArgument(n >= 1);

        final double hue_step = 360.d / n;

        final double[] colors = new double[n];
        for (int i = 0; i < n; i++) {
            colors[i] = i * hue_step / 360.d;
        }
        return colors;
    }

}
