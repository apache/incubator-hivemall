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
package hivemall.utils.lang;

import static hivemall.utils.lang.Preconditions.checkElementIndex;

import hivemall.utils.random.PRNG;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.math3.distribution.GammaDistribution;

public final class ArrayUtils {

    /**
     * The index value when an element is not found in a list or array: <code>-1</code>. This value
     * is returned by methods in this class and can also be used in comparisons with values returned
     * by various method from {@link java.util.List}.
     */
    public static final int INDEX_NOT_FOUND = -1;

    private ArrayUtils() {}

    @SuppressWarnings("unchecked")
    public static <T> T[] newInstance(@Nonnull final T[] a, @Nonnegative final int newLength) {
        return (T[]) newInstance(a.getClass(), newLength);
    }

    @SuppressWarnings("unchecked")
    public static <T> T[] newInstance(@Nonnull final Class<? extends T[]> newType,
            @Nonnegative final int newLength) {
        if ((Object) newType == (Object) Object[].class) {
            return (T[]) new Object[newLength];
        } else {
            return (T[]) Array.newInstance(newType.getComponentType(), newLength);
        }
    }

    @Nonnull
    public static int get(final int[] a, final int index) {
        return a[checkElementIndex(index, a.length)];
    }

    @Nonnull
    public static double get(final double[] a, final int index) {
        return a[checkElementIndex(index, a.length)];
    }

    @Nonnull
    public static <T> T get(final T[] a, final int index) {
        return a[checkElementIndex(index, a.length)];
    }

    @Nonnull
    public static double[] set(@Nonnull double[] src, final int index, final double value) {
        if (index >= src.length) {
            src = Arrays.copyOf(src, src.length * 2);
        }
        src[index] = value;
        return src;
    }

    @Nonnull
    public static <T> T[] set(@Nonnull T[] src, final int index, final T value) {
        if (index >= src.length) {
            src = Arrays.copyOf(src, src.length * 2);
        }
        src[index] = value;
        return src;
    }

    @Nonnull
    public static int[] sortedArraySet(@Nonnull final int[] sorted,
            @Nonnegative final int element) {
        final int i = Arrays.binarySearch(sorted, element);
        if (i >= 0) {// found element
            return sorted;
        } else {
            return insert(sorted, ~i, element);
        }
    }

    public static boolean contains(@Nonnull final int[] sorted, @Nonnegative final int element) {
        int i = Arrays.binarySearch(sorted, element);
        return i >= 0;
    }

    @Nonnull
    public static float[] toArray(@Nonnull final List<Float> lst) {
        final int ndim = lst.size();
        final float[] ary = new float[ndim];
        int i = 0;
        for (float f : lst) {
            ary[i++] = f;
        }
        return ary;
    }

    @Nonnull
    public static Integer[] toObject(@Nonnull final int[] array) {
        final Integer[] result = new Integer[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    @Nonnull
    public static List<Integer> toList(@Nonnull final int[] array) {
        Integer[] v = toObject(array);
        return Arrays.asList(v);
    }

    @Nonnull
    public static Long[] toObject(@Nonnull final long[] array) {
        final Long[] result = new Long[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    @Nonnull
    public static List<Long> toList(@Nonnull final long[] array) {
        Long[] v = toObject(array);
        return Arrays.asList(v);
    }

    @Nonnull
    public static Float[] toObject(@Nonnull final float[] array) {
        final Float[] result = new Float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    @Nonnull
    public static List<Float> toList(@Nonnull final float[] array) {
        Float[] v = toObject(array);
        return Arrays.asList(v);
    }

    @Nonnull
    public static Double[] toObject(@Nonnull final double[] array) {
        final Double[] result = new Double[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }

    @Nonnull
    public static List<Double> toList(@Nonnull final double[] array) {
        Double[] v = toObject(array);
        return Arrays.asList(v);
    }

    @Nonnull
    public static int[] slice(@Nonnull final int[] a, @Nonnull final int... indexes) {
        final int size = indexes.length;
        final int[] ret = new int[size];
        for (int i = 0; i < size; i++) {
            ret[i] = get(a, indexes[i]);
        }
        return ret;
    }

    @Nonnull
    public static double[] slice(@Nonnull final double[] a, @Nonnull final int... indexes) {
        final int size = indexes.length;
        final double[] ret = new double[size];
        for (int i = 0; i < size; i++) {
            ret[i] = get(a, indexes[i]);
        }
        return ret;
    }

    @Nonnull
    public static <T> T[] slice(@Nonnull final T[] a, @Nonnull final int... indexes) {
        final int size = indexes.length;
        final T[] ret = newInstance(a, size);
        for (int i = 0; i < size; i++) {
            ret[i] = get(a, indexes[i]);
        }
        return ret;
    }

    @Nonnull
    public static <T> T[] shuffle(@Nonnull final T[] array) {
        shuffle(array, array.length);
        return array;
    }

    @Nonnull
    public static <T> T[] shuffle(@Nonnull final T[] array, final Random rnd) {
        shuffle(array, array.length, rnd);
        return array;
    }

    @Nonnull
    public static <T> T[] shuffle(@Nonnull final T[] array, final int size) {
        Random rnd = new Random();
        shuffle(array, size, rnd);
        return array;
    }

    /**
     * Fisher–Yates shuffle
     * 
     * @link http://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
     */
    @Nonnull
    public static <T> T[] shuffle(@Nonnull final T[] array, final int size,
            @Nonnull final Random rnd) {
        for (int i = size; i > 1; i--) {
            int randomPosition = rnd.nextInt(i);
            swap(array, i - 1, randomPosition);
        }
        return array;
    }

    @Nonnull
    public static int[] shuffle(@Nonnull final int[] array, @Nonnull final Random rnd) {
        for (int i = array.length; i > 1; i--) {
            int randomPosition = rnd.nextInt(i);
            swap(array, i - 1, randomPosition);
        }
        return array;
    }

    public static void swap(@Nonnull final Object[] arr, final int i, final int j) {
        Object tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    public static void swap(@Nonnull final int[] arr, final int i, final int j) {
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    public static void swap(@Nonnull final long[] arr, final int i, final int j) {
        long tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    public static void swap(@Nonnull final float[] arr, final int i, final int j) {
        float tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    public static void swap(@Nonnull final double[] arr, final int i, final int j) {
        double tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    @Nullable
    public static Object[] subarray(@Nullable final Object[] array, int startIndexInclusive,
            int endIndexExclusive) {
        if (array == null) {
            return null;
        }
        if (startIndexInclusive < 0) {
            startIndexInclusive = 0;
        }
        if (endIndexExclusive > array.length) {
            endIndexExclusive = array.length;
        }
        int newSize = endIndexExclusive - startIndexInclusive;
        Class<?> type = array.getClass().getComponentType();
        if (newSize <= 0) {
            return (Object[]) Array.newInstance(type, 0);
        }
        Object[] subarray = (Object[]) Array.newInstance(type, newSize);
        System.arraycopy(array, startIndexInclusive, subarray, 0, newSize);
        return subarray;
    }

    public static void fill(@Nonnull final float[] a, @Nonnull final Random rand) {
        for (int i = 0, len = a.length; i < len; i++) {
            a[i] = rand.nextFloat();
        }
    }

    public static int indexOf(@Nullable final int[] array, final int valueToFind,
            final int startIndex, final int endIndex) {
        if (array == null) {
            return INDEX_NOT_FOUND;
        }
        final int til = Math.min(endIndex, array.length);
        if (startIndex < 0 || startIndex > til) {
            throw new IllegalArgumentException("Illegal startIndex: " + startIndex);
        }
        for (int i = startIndex; i < til; i++) {
            if (valueToFind == array[i]) {
                return i;
            }
        }
        return INDEX_NOT_FOUND;
    }

    public static int lastIndexOf(@Nullable final int[] array, final int valueToFind,
            int startIndex) {
        if (array == null) {
            return INDEX_NOT_FOUND;
        }
        return lastIndexOf(array, valueToFind, startIndex, array.length);
    }

    /**
     * @param startIndex inclusive start index
     * @param endIndex exclusive end index
     */
    public static int lastIndexOf(@Nullable final int[] array, final int valueToFind,
            int startIndex, int endIndex) {
        if (array == null) {
            return INDEX_NOT_FOUND;
        }
        if (startIndex < 0) {
            throw new IllegalArgumentException("startIndex out of bound: " + startIndex);
        }
        if (endIndex >= array.length) {
            throw new IllegalArgumentException("endIndex out of bound: " + endIndex);
        }
        for (int i = endIndex - 1; i >= startIndex; i--) {
            if (valueToFind == array[i]) {
                return i;
            }
        }
        return INDEX_NOT_FOUND;
    }

    public static int insertionPoint(@Nonnull final int[] a, final int key) {
        final int pos = Arrays.binarySearch(a, key);
        if (pos < 0) {
            return ~pos;
        } else {
            return pos;
        }
    }

    public static int insertionPoint(@Nonnull final int[] a, final int size, final int key) {
        final int pos = Arrays.binarySearch(a, 0, size, key);
        if (pos < 0) {
            return ~pos;
        } else {
            return pos;
        }
    }

    @Nonnull
    public static byte[] copyOf(@Nonnull final byte[] original, final int newLength) {
        final byte[] copy = new byte[newLength];
        System.arraycopy(original, 0, copy, 0, Math.min(original.length, newLength));
        return copy;
    }

    public static int[] copyOf(@Nonnull final int[] src) {
        int len = src.length;
        int[] dest = new int[len];
        System.arraycopy(src, 0, dest, 0, len);
        return dest;
    }

    public static void copy(@Nonnull final int[] src, @Nonnull final int[] dest) {
        if (src.length != dest.length) {
            throw new IllegalArgumentException(
                "src.legnth '" + src.length + "' != dest.length '" + dest.length + "'");
        }
        System.arraycopy(src, 0, dest, 0, src.length);
    }

    @Nonnull
    public static int[] append(@Nonnull final int[] array, final int element) {
        int size = array.length;
        final int[] newArray = new int[size + 1];
        System.arraycopy(array, 0, newArray, 0, size);
        newArray[size] = element;
        return newArray;
    }

    @Nonnull
    public static int[] append(@Nonnull int[] array, final int currentSize, final int element) {
        if (currentSize + 1 > array.length) {
            int[] newArray = new int[currentSize * 2];
            System.arraycopy(array, 0, newArray, 0, currentSize);
            array = newArray;
        }
        array[currentSize] = element;
        return array;
    }

    @Nonnull
    public static float[] append(@Nonnull float[] array, final int currentSize,
            final float element) {
        if (currentSize + 1 > array.length) {
            float[] newArray = new float[currentSize * 2];
            System.arraycopy(array, 0, newArray, 0, currentSize);
            array = newArray;
        }
        array[currentSize] = element;
        return array;
    }

    @Nonnull
    public static double[] append(@Nonnull double[] array, final int currentSize,
            final double element) {
        if (currentSize + 1 > array.length) {
            double[] newArray = new double[currentSize * 2];
            System.arraycopy(array, 0, newArray, 0, currentSize);
            array = newArray;
        }
        array[currentSize] = element;
        return array;
    }

    @Nonnull
    public static int[] concat(@Nonnegative final int[] array1, @Nonnegative final int... array2) {
        final int[] joinedArray = new int[array1.length + array2.length];
        System.arraycopy(array1, 0, joinedArray, 0, array1.length);
        System.arraycopy(array2, 0, joinedArray, array1.length, array2.length);
        return joinedArray;
    }

    @Nonnull
    public static int[] concat(@Nonnegative final int[] array1, @Nonnegative final int[] array2,
            final int offset, final int length) {
        final int[] joinedArray = new int[array1.length + length];
        System.arraycopy(array1, 0, joinedArray, 0, array1.length);
        System.arraycopy(array2, offset, joinedArray, array1.length, length);
        return joinedArray;
    }

    @Nonnull
    public static int[] concat(@Nonnegative final int[] array1, final int offset1,
            final int length1, @Nonnegative final int[] array2, final int offset2,
            final int length2) {
        final int[] joinedArray = new int[length1 + length2];
        System.arraycopy(array1, offset1, joinedArray, 0, length1);
        System.arraycopy(array2, offset2, joinedArray, length1, length2);
        return joinedArray;
    }

    @Nonnull
    public static int[] insert(@Nonnull final int[] array, final int index, final int element) {
        final int size = array.length;
        if (index > size) {
            throw new IllegalArgumentException(String.format(
                "index should be less than or equals to array.length: index=%d, array.length=%d",
                index, array.length));
        }
        final int[] newArray = new int[size + 1];
        System.arraycopy(array, 0, newArray, 0, Math.min(index, size));
        newArray[index] = element;
        if (index != size) {
            System.arraycopy(array, index, newArray, index + 1, size - index);
        }
        return newArray;
    }

    @Nonnull
    public static int[] insert(@Nonnull final int[] array, final int currentSize, final int index,
            final int element) {
        if (currentSize + 1 <= array.length) {
            System.arraycopy(array, index, array, index + 1, currentSize - index);
            array[index] = element;
            return array;
        }
        final int[] newArray = new int[currentSize * 2];
        System.arraycopy(array, 0, newArray, 0, index);
        newArray[index] = element;
        System.arraycopy(array, index, newArray, index + 1, array.length - index);
        return newArray;
    }

    @Nonnull
    public static float[] insert(@Nonnull final float[] array, final int currentSize,
            final int index, final float element) {
        if (currentSize + 1 <= array.length) {
            System.arraycopy(array, index, array, index + 1, currentSize - index);
            array[index] = element;
            return array;
        }
        final float[] newArray = new float[currentSize * 2];
        System.arraycopy(array, 0, newArray, 0, index);
        newArray[index] = element;
        System.arraycopy(array, index, newArray, index + 1, array.length - index);
        return newArray;
    }

    @Nonnull
    public static double[] insert(@Nonnull final double[] array, final int currentSize,
            final int index, final double element) {
        if (currentSize + 1 <= array.length) {
            System.arraycopy(array, index, array, index + 1, currentSize - index);
            array[index] = element;
            return array;
        }
        final double[] newArray = new double[currentSize * 2];
        System.arraycopy(array, 0, newArray, 0, index);
        newArray[index] = element;
        System.arraycopy(array, index, newArray, index + 1, array.length - index);
        return newArray;
    }

    /**
     * Removes from {@code array} all of the elements whose index is between {@code fromIndex},
     * inclusive, and {@code toIndex}, exclusive.
     *
     * @param fromIndex index of first element to be removed
     * @param toIndex index after last element to be removed
     * @throws IndexOutOfBoundsException if {@code fromIndex} or {@code toIndex} is out of range
     *         ({@code fromIndex < 0 ||
     *          fromIndex >= size() ||
     *          toIndex > size() ||
     *          toIndex < fromIndex})
     */
    public static void clearRange(@Nonnull final int[] array, @Nonnegative final int fromIndex,
            @Nonnegative final int toIndex, final int fillVal) {
        final int size = array.length;
        if (fromIndex < 0 || fromIndex >= size || toIndex > size || toIndex < fromIndex) {
            throw new IllegalArgumentException(String.format(
                "fromIndex: %d, toIndex: %d, array.length=%d", fromIndex, toIndex, size));
        }

        System.arraycopy(array, toIndex, array, fromIndex, array.length - toIndex);
        Arrays.fill(array, toIndex, array.length, fillVal);
    }

    public static boolean equals(@Nonnull final float[] array, final float value) {
        for (int i = 0, size = array.length; i < size; i++) {
            if (array[i] != value) {
                return false;
            }
        }
        return true;
    }

    public static boolean almostEquals(@Nonnull final float[] array, final float expected) {
        return equals(array, expected, 1E-15f);
    }

    public static boolean equals(@Nonnull final float[] array, final float expected,
            final float delta) {
        for (int i = 0, size = array.length; i < size; i++) {
            float actual = array[i];
            if (Math.abs(expected - actual) > delta) {
                return false;
            }
        }
        return true;
    }

    public static void copy(@Nonnull final float[] src, @Nonnull final double[] dst) {
        final int size = Math.min(src.length, dst.length);
        for (int i = 0; i < size; i++) {
            dst[i] = src[i];
        }
    }

    public static void sort(final long[] arr, final double[] brr) {
        sort(arr, brr, arr.length);
    }

    public static void sort(final long[] arr, final double[] brr, final int n) {
        final int NSTACK = 64;
        final int M = 7;
        final int[] istack = new int[NSTACK];

        int jstack = -1;
        int l = 0;
        int ir = n - 1;

        int i, j, k;
        long a;
        double b;
        for (;;) {
            if (ir - l < M) {
                for (j = l + 1; j <= ir; j++) {
                    a = arr[j];
                    b = brr[j];
                    for (i = j - 1; i >= l; i--) {
                        if (arr[i] <= a) {
                            break;
                        }
                        arr[i + 1] = arr[i];
                        brr[i + 1] = brr[i];
                    }
                    arr[i + 1] = a;
                    brr[i + 1] = b;
                }
                if (jstack < 0) {
                    break;
                }
                ir = istack[jstack--];
                l = istack[jstack--];
            } else {
                k = (l + ir) >> 1;
                swap(arr, k, l + 1);
                swap(brr, k, l + 1);
                if (arr[l] > arr[ir]) {
                    swap(arr, l, ir);
                    swap(brr, l, ir);
                }
                if (arr[l + 1] > arr[ir]) {
                    swap(arr, l + 1, ir);
                    swap(brr, l + 1, ir);
                }
                if (arr[l] > arr[l + 1]) {
                    swap(arr, l, l + 1);
                    swap(brr, l, l + 1);
                }
                i = l + 1;
                j = ir;
                a = arr[l + 1];
                b = brr[l + 1];
                for (;;) {
                    do {
                        i++;
                    } while (arr[i] < a);
                    do {
                        j--;
                    } while (arr[j] > a);
                    if (j < i) {
                        break;
                    }
                    swap(arr, i, j);
                    swap(brr, i, j);
                }
                arr[l + 1] = arr[j];
                arr[j] = a;
                brr[l + 1] = brr[j];
                brr[j] = b;
                jstack += 2;

                if (jstack >= NSTACK) {
                    throw new IllegalStateException("NSTACK too small in sort.");
                }

                if (ir - i + 1 >= j - l) {
                    istack[jstack] = ir;
                    istack[jstack - 1] = i;
                    ir = j - 1;
                } else {
                    istack[jstack] = j - 1;
                    istack[jstack - 1] = l;
                    l = i;
                }
            }
        }
    }

    public static void sort(@Nonnull final int[] arr, @Nonnull final int[] brr,
            @Nonnull final double[] crr) {
        sort(arr, brr, crr, arr.length);
    }

    public static void sort(@Nonnull final int[] arr, @Nonnull final int[] brr,
            @Nonnull final double[] crr, final int n) {
        Preconditions.checkArgument(arr.length >= n);
        Preconditions.checkArgument(brr.length >= n);
        Preconditions.checkArgument(crr.length >= n);

        final int NSTACK = 64;
        final int M = 7;
        final int[] istack = new int[NSTACK];

        int jstack = -1;
        int l = 0;
        int ir = n - 1;

        int i, j, k;
        int a, b;
        double c;
        for (;;) {
            if (ir - l < M) {
                for (j = l + 1; j <= ir; j++) {
                    a = arr[j];
                    b = brr[j];
                    c = crr[j];
                    for (i = j - 1; i >= l; i--) {
                        if (arr[i] <= a) {
                            break;
                        }
                        arr[i + 1] = arr[i];
                        brr[i + 1] = brr[i];
                        crr[i + 1] = crr[i];
                    }
                    arr[i + 1] = a;
                    brr[i + 1] = b;
                    crr[i + 1] = c;
                }
                if (jstack < 0) {
                    break;
                }
                ir = istack[jstack--];
                l = istack[jstack--];
            } else {
                k = (l + ir) >> 1;
                swap(arr, k, l + 1);
                swap(brr, k, l + 1);
                swap(crr, k, l + 1);
                if (arr[l] > arr[ir]) {
                    swap(arr, l, ir);
                    swap(brr, l, ir);
                    swap(crr, l, ir);
                }
                if (arr[l + 1] > arr[ir]) {
                    swap(arr, l + 1, ir);
                    swap(brr, l + 1, ir);
                    swap(crr, l + 1, ir);
                }
                if (arr[l] > arr[l + 1]) {
                    swap(arr, l, l + 1);
                    swap(brr, l, l + 1);
                    swap(crr, l, l + 1);
                }
                i = l + 1;
                j = ir;
                a = arr[l + 1];
                b = brr[l + 1];
                c = crr[l + 1];
                for (;;) {
                    do {
                        i++;
                    } while (arr[i] < a);
                    do {
                        j--;
                    } while (arr[j] > a);
                    if (j < i) {
                        break;
                    }
                    swap(arr, i, j);
                    swap(brr, i, j);
                    swap(crr, i, j);
                }
                arr[l + 1] = arr[j];
                arr[j] = a;
                brr[l + 1] = brr[j];
                brr[j] = b;
                crr[l + 1] = crr[j];
                crr[j] = c;
                jstack += 2;

                if (jstack >= NSTACK) {
                    throw new IllegalStateException("NSTACK too small in sort.");
                }

                if (ir - i + 1 >= j - l) {
                    istack[jstack] = ir;
                    istack[jstack - 1] = i;
                    ir = j - 1;
                } else {
                    istack[jstack] = j - 1;
                    istack[jstack - 1] = l;
                    l = i;
                }
            }
        }
    }

    public static void sort(@Nonnull final int[] arr, @Nonnull final int[] brr,
            @Nonnull final float[] crr) {
        sort(arr, brr, crr, arr.length);
    }

    public static void sort(@Nonnull final int[] arr, @Nonnull final int[] brr,
            @Nonnull final float[] crr, final int n) {
        Preconditions.checkArgument(arr.length >= n);
        Preconditions.checkArgument(brr.length >= n);
        Preconditions.checkArgument(crr.length >= n);

        final int NSTACK = 64;
        final int M = 7;
        final int[] istack = new int[NSTACK];

        int jstack = -1;
        int l = 0;
        int ir = n - 1;

        int i, j, k;
        int a, b;
        float c;
        for (;;) {
            if (ir - l < M) {
                for (j = l + 1; j <= ir; j++) {
                    a = arr[j];
                    b = brr[j];
                    c = crr[j];
                    for (i = j - 1; i >= l; i--) {
                        if (arr[i] <= a) {
                            break;
                        }
                        arr[i + 1] = arr[i];
                        brr[i + 1] = brr[i];
                        crr[i + 1] = crr[i];
                    }
                    arr[i + 1] = a;
                    brr[i + 1] = b;
                    crr[i + 1] = c;
                }
                if (jstack < 0) {
                    break;
                }
                ir = istack[jstack--];
                l = istack[jstack--];
            } else {
                k = (l + ir) >> 1;
                swap(arr, k, l + 1);
                swap(brr, k, l + 1);
                swap(crr, k, l + 1);
                if (arr[l] > arr[ir]) {
                    swap(arr, l, ir);
                    swap(brr, l, ir);
                    swap(crr, l, ir);
                }
                if (arr[l + 1] > arr[ir]) {
                    swap(arr, l + 1, ir);
                    swap(brr, l + 1, ir);
                    swap(crr, l + 1, ir);
                }
                if (arr[l] > arr[l + 1]) {
                    swap(arr, l, l + 1);
                    swap(brr, l, l + 1);
                    swap(crr, l, l + 1);
                }
                i = l + 1;
                j = ir;
                a = arr[l + 1];
                b = brr[l + 1];
                c = crr[l + 1];
                for (;;) {
                    do {
                        i++;
                    } while (arr[i] < a);
                    do {
                        j--;
                    } while (arr[j] > a);
                    if (j < i) {
                        break;
                    }
                    swap(arr, i, j);
                    swap(brr, i, j);
                    swap(crr, i, j);
                }
                arr[l + 1] = arr[j];
                arr[j] = a;
                brr[l + 1] = brr[j];
                brr[j] = b;
                crr[l + 1] = crr[j];
                crr[j] = c;
                jstack += 2;

                if (jstack >= NSTACK) {
                    throw new IllegalStateException("NSTACK too small in sort.");
                }

                if (ir - i + 1 >= j - l) {
                    istack[jstack] = ir;
                    istack[jstack - 1] = i;
                    ir = j - 1;
                } else {
                    istack[jstack] = j - 1;
                    istack[jstack - 1] = l;
                    l = i;
                }
            }
        }
    }

    @Nonnull
    public static int[] argsort(@Nonnull final int[] a) {
        final int size = a.length;
        final Integer[] indexes = new Integer[size];
        for (int i = 0; i < size; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return Integer.compare(a[i], a[j]);
            }
        });

        final int[] ret = new int[size];
        for (int i = 0; i < size; i++) {
            ret[i] = indexes[i].intValue();
        }
        return ret;
    }

    @Nonnull
    public static int[] argsort(@Nonnull final double[] a) {
        final int size = a.length;
        final Integer[] indexes = new Integer[size];
        for (int i = 0; i < size; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return Double.compare(a[i], a[j]);
            }
        });

        final int[] ret = new int[size];
        for (int i = 0; i < size; i++) {
            ret[i] = indexes[i].intValue();
        }
        return ret;
    }

    @Nonnull
    public static <T extends Comparable<T>> int[] argsort(@Nonnull final T[] a) {
        final int size = a.length;
        final Integer[] indexes = new Integer[size];
        for (int i = 0; i < size; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                T ai = a[i.intValue()];
                T aj = a[j.intValue()];
                return ai.compareTo(aj);
            }
        });

        final int[] ret = new int[size];
        for (int i = 0; i < size; i++) {
            ret[i] = indexes[i].intValue();
        }
        return ret;
    }

    @Nonnull
    public static <T> int[] argsort(@Nonnull final T[] a, @Nonnull final Comparator<? super T> c) {
        final int size = a.length;
        final Integer[] indexes = new Integer[size];
        for (int i = 0; i < size; i++) {
            indexes[i] = i;
        }
        Arrays.sort(indexes, new Comparator<Integer>() {
            @Override
            public int compare(Integer i, Integer j) {
                return c.compare(a[i.intValue()], a[j.intValue()]);
            }
        });

        final int[] ret = new int[size];
        for (int i = 0; i < size; i++) {
            ret[i] = indexes[i].intValue();
        }
        return ret;
    }

    public static int[] argrank(@Nonnull final int[] a) {
        return argsort(argsort(a));
    }

    public static int[] argrank(@Nonnull final double[] a) {
        return argsort(argsort(a));
    }

    @Nonnull
    public static <T> int[] argrank(@Nonnull final T[] a, @Nonnull final Comparator<? super T> c) {
        return argsort(argsort(a, c));
    }

    public static int argmin(@Nonnull final double[] a) {
        final int size = a.length;
        if (size == 0) {
            throw new IllegalArgumentException("attempt to get argmin of an empty array");
        }

        int minIdx = 0;
        double minValue = Double.POSITIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            final double v = a[i];
            if (v < minValue) {
                minIdx = i;
                minValue = v;
            }
        }
        return minIdx;
    }

    public static <T extends Comparable<T>> int argmin(@Nonnull final T[] a) {
        final int size = a.length;
        if (size == 0) {
            throw new IllegalArgumentException("attempt to get argmin of an empty array");
        }

        int minIdx = 0;
        T minValue = null;
        for (int i = 0; i < size; i++) {
            final T v = a[i];
            if (v == null) {
                continue;
            }
            if (minValue == null || v.compareTo(minValue) < 0) {
                minIdx = i;
                minValue = v;
            }
        }
        return minIdx;
    }

    @Nonnull
    public static <T> int argmin(@Nonnull final T[] a, @Nonnull final Comparator<? super T> c) {
        final int size = a.length;
        if (size == 0) {
            throw new IllegalArgumentException("attempt to get argmin of an empty array");
        }
        if (size == 1) {
            return 0;
        }

        int minIdx = 0;
        T minValue = a[0];
        for (int i = 1; i < size; i++) {
            final T v = a[i];
            if (c.compare(v, minValue) < 0) {
                minIdx = i;
                minValue = v;
            }
        }
        return minIdx;
    }

    public static int argmax(@Nonnull final double[] a) {
        final int size = a.length;
        if (size == 0) {
            throw new IllegalArgumentException("attempt to get argmin of an empty array");
        }

        int maxIdx = 0;
        double maxVaue = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < size; i++) {
            final double v = a[i];
            if (v > maxVaue) {
                maxIdx = i;
                maxVaue = v;
            }
        }
        return maxIdx;
    }

    public static <T extends Comparable<T>> int argmax(@Nonnull final T[] a) {
        final int size = a.length;
        if (size == 0) {
            throw new IllegalArgumentException("attempt to get argmax of an empty array");
        }

        int maxIdx = 0;
        T maxValue = null;
        for (int i = 0; i < size; i++) {
            final T v = a[i];
            if (v == null) {
                continue;
            }
            if (maxValue == null || v.compareTo(maxValue) > 0) {
                maxIdx = i;
                maxValue = v;
            }
        }
        return maxIdx;
    }

    @Nonnull
    public static <T> int argmax(@Nonnull final T[] a, @Nonnull final Comparator<? super T> c) {
        final int size = a.length;
        if (size == 0) {
            throw new IllegalArgumentException("attempt to get argmax of an empty array");
        }
        if (size == 1) {
            return 0;
        }

        int maxIdx = 0;
        T maxValue = a[0]; // consideration for null
        for (int i = 1; i < size; i++) {
            final T v = a[i];
            if (c.compare(v, maxValue) > 0) {
                maxIdx = i;
                maxValue = v;
            }
        }
        return maxIdx;
    }

    public static int[] range(final int stop) {
        return range(0, stop);
    }

    public static int[] range(final int start, final int stop) {
        final int count = stop - start;
        final int[] r;
        if (count < 0) {
            r = new int[-count];
            for (int i = 0; i < r.length; i++) {
                r[i] = start - i;
            }
        } else {
            r = new int[count];
            for (int i = 0; i < r.length; i++) {
                r[i] = start + i;
            }
        }
        return r;
    }

    /**
     * Return evenly spaced values within a given interval.
     *
     * @param start inclusive index of the start
     * @param stop exclusive index of the end
     * @param step positive interval value
     */
    public static int[] range(final int start, final int stop, @Nonnegative final int step) {
        if (step <= 0) {
            throw new IllegalArgumentException("Invalid step value: " + step);
        }

        final int[] r;
        final int diff = stop - start;
        if (diff < 0) {
            final int count = divideAndRoundUp(-diff, step);
            r = new int[count];
            for (int i = 0, value = start; i < r.length; i++, value -= step) {
                r[i] = value;
            }
        } else {
            final int count = divideAndRoundUp(diff, step);
            r = new int[count];
            for (int i = 0, value = start; i < r.length; i++, value += step) {
                r[i] = value;
            }
        }
        return r;
    }

    public static int divideAndRoundUp(@Nonnegative final int num, @Nonnegative final int divisor) {
        return (num + divisor - 1) / divisor;
    }

    public static int count(@Nonnull final int[] values, final int valueToFind) {
        int cnt = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] == valueToFind) {
                cnt++;
            }
        }
        return cnt;
    }

    @Nonnull
    public static float[] newFloatArray(@Nonnegative int size, float filledValue) {
        final float[] a = new float[size];
        Arrays.fill(a, filledValue);
        return a;
    }

    @Nonnull
    public static float[] newRandomFloatArray(@Nonnegative final int size,
            @Nonnull final GammaDistribution gd) {
        final float[] ret = new float[size];
        for (int i = 0; i < size; i++) {
            ret[i] = (float) gd.sample();
        }
        return ret;
    }

    @Nonnull
    public static float[] newRandomFloatArray(@Nonnegative final int size,
            @Nonnull final PRNG rnd) {
        final float[] ret = new float[size];
        for (int i = 0; i < size; i++) {
            ret[i] = (float) rnd.nextDouble();
        }
        return ret;
    }

    /**
     * Workaround for org.apache.hive.com.esotericsoftware.kryo.KryoException
     */
    @Nonnull
    public static List<String> asKryoSerializableList(@Nonnull final String[] array) {
        return new ArrayList<>(Arrays.asList(array));
    }

}
