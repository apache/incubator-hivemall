package hivemall.matrix.ints;

import hivemall.utils.lang.mutable.MutableInt;
import hivemall.vector.VectorProcedure;

import org.junit.Assert;
import org.junit.Test;

public class IntMatrixTest {

    @Test
    public void testDoKMatrixRowMajor() {
        DoKIntMatrix matrix = DoKIntMatrix.build(rowMajorData(), true, true);

        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());

        Assert.assertEquals(11, matrix.get(0, 0));
        Assert.assertEquals(12, matrix.get(0, 1));
        Assert.assertEquals(13, matrix.get(0, 2));
        Assert.assertEquals(14, matrix.get(0, 3));
        Assert.assertEquals(22, matrix.get(1, 1));
        Assert.assertEquals(23, matrix.get(1, 2));
        Assert.assertEquals(33, matrix.get(2, 2));
        Assert.assertEquals(34, matrix.get(2, 3));
        Assert.assertEquals(35, matrix.get(2, 4));
        Assert.assertEquals(36, matrix.get(2, 5));
        Assert.assertEquals(44, matrix.get(3, 3));
        Assert.assertEquals(45, matrix.get(3, 4));
        Assert.assertEquals(56, matrix.get(4, 5));
        Assert.assertEquals(66, matrix.get(5, 5));

        Assert.assertEquals(0, matrix.get(5, 4));
        Assert.assertEquals(0, matrix.get(1, 0));
        Assert.assertEquals(0, matrix.get(1, 3));
        Assert.assertEquals(-1, matrix.get(1, 0, -1));
    }

    @Test
    public void testDoKMatrixColumnMajor() {
        DoKIntMatrix matrix = DoKIntMatrix.build(columnMajorData(), false, true);

        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());

        Assert.assertEquals(11, matrix.get(0, 0));
        Assert.assertEquals(12, matrix.get(0, 1));
        Assert.assertEquals(13, matrix.get(0, 2));
        Assert.assertEquals(14, matrix.get(0, 3));
        Assert.assertEquals(22, matrix.get(1, 1));
        Assert.assertEquals(23, matrix.get(1, 2));
        Assert.assertEquals(33, matrix.get(2, 2));
        Assert.assertEquals(34, matrix.get(2, 3));
        Assert.assertEquals(35, matrix.get(2, 4));
        Assert.assertEquals(36, matrix.get(2, 5));
        Assert.assertEquals(44, matrix.get(3, 3));
        Assert.assertEquals(45, matrix.get(3, 4));
        Assert.assertEquals(56, matrix.get(4, 5));
        Assert.assertEquals(66, matrix.get(5, 5));

        Assert.assertEquals(0, matrix.get(5, 4));
        Assert.assertEquals(0, matrix.get(1, 0));
        Assert.assertEquals(0, matrix.get(1, 3));
        Assert.assertEquals(-1, matrix.get(1, 0, -1));
    }

    @Test
    public void testDoKMatrixColumnMajorNonZeroOnlyFalse() {
        DoKIntMatrix matrix = DoKIntMatrix.build(columnMajorData(), false, false);

        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());

        Assert.assertEquals(0, matrix.get(5, 4));
        Assert.assertEquals(0, matrix.get(1, 0));
        Assert.assertEquals(0, matrix.get(1, 3));
        Assert.assertEquals(0, matrix.get(1, 3, -1));
        Assert.assertEquals(-1, matrix.get(1, 0, -1));

        matrix.setDefaultValue(-1);
        Assert.assertEquals(-1, matrix.get(5, 4));
        Assert.assertEquals(-1, matrix.get(1, 0));
        Assert.assertEquals(0, matrix.get(1, 3));
        Assert.assertEquals(0, matrix.get(1, 0, 0));
    }

    @Test
    public void testColumnMajorDenseMatrix() {
        ColumnMajorDenseIntMatrix2d matrix = new ColumnMajorDenseIntMatrix2d(columnMajorData(), 6);
        Assert.assertEquals(6, matrix.numRows());
        Assert.assertEquals(6, matrix.numColumns());

        Assert.assertEquals(11, matrix.get(0, 0));
        Assert.assertEquals(12, matrix.get(0, 1));
        Assert.assertEquals(13, matrix.get(0, 2));
        Assert.assertEquals(14, matrix.get(0, 3));
        Assert.assertEquals(22, matrix.get(1, 1));
        Assert.assertEquals(23, matrix.get(1, 2));
        Assert.assertEquals(33, matrix.get(2, 2));
        Assert.assertEquals(34, matrix.get(2, 3));
        Assert.assertEquals(35, matrix.get(2, 4));
        Assert.assertEquals(36, matrix.get(2, 5));
        Assert.assertEquals(44, matrix.get(3, 3));
        Assert.assertEquals(45, matrix.get(3, 4));
        Assert.assertEquals(56, matrix.get(4, 5));
        Assert.assertEquals(66, matrix.get(5, 5));

        Assert.assertEquals(0, matrix.get(5, 4));
        Assert.assertEquals(0, matrix.get(1, 0));
        Assert.assertEquals(0, matrix.get(1, 3));
        Assert.assertEquals(-1, matrix.get(1, 0, -1));
    }

    @Test
    public void testColumnMajorDenseMatrixEachColumn() {
        ColumnMajorDenseIntMatrix2d matrix = new ColumnMajorDenseIntMatrix2d(columnMajorData(), 6);
        matrix.setDefaultValue(-1);

        final MutableInt count = new MutableInt(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 4 + 4 + 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            }, true);
        }
        Assert.assertEquals(6 * 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInNonZeroColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 3 + 2 + 3, count.getValue());

        // change default value to zero
        matrix.setDefaultValue(0);

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 4 + 4 + 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            }, true);
        }
        Assert.assertEquals(6 * 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInNonZeroColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 3 + 2 + 3, count.getValue());
    }

    @Test
    public void testDoKMatrixColumnMajorNonZeroOnlyFalseEachColumn() {
        DoKIntMatrix matrix = DoKIntMatrix.build(columnMajorData(), false, false);
        matrix.setDefaultValue(-1);

        final MutableInt count = new MutableInt(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 4 + 4 + 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            }, true);
        }
        Assert.assertEquals(6 * 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInNonZeroColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 3 + 2 + 3, count.getValue());

        // change default value to zero
        matrix.setDefaultValue(0);

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 4 + 4 + 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            }, true);
        }
        Assert.assertEquals(6 * 6, count.getValue());

        count.setValue(0);
        for (int j = 0; j < 6; j++) {
            matrix.eachInNonZeroColumn(j, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(1 + 2 + 3 + 3 + 2 + 3, count.getValue());
    }

    @Test
    public void testDoKMatrixRowMajorNonZeroOnlyFalseEachColumn() {
        DoKIntMatrix matrix = DoKIntMatrix.build(rowMajorData(), true, false);
        matrix.setDefaultValue(-1);

        final MutableInt count = new MutableInt(0);
        for (int i = 0; i < 6; i++) {
            matrix.eachInRow(i, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(4 + 3 + 6 + 5 + 6 + 6, count.getValue());

        count.setValue(0);
        for (int i = 0; i < 6; i++) {
            matrix.eachInRow(i, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            }, true);
        }
        Assert.assertEquals(6 * 6, count.getValue());

        count.setValue(0);
        for (int i = 0; i < 6; i++) {
            matrix.eachNonZeroInRow(i, new VectorProcedure() {
                @Override
                public void apply(int i, int value) {
                    count.addValue(1);
                }
            });
        }
        Assert.assertEquals(4 + 2 + 4 + 2 + 1 + 1, count.getValue());
    }

    private static int[][] rowMajorData() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        int[][] data = new int[6][];
        data[0] = new int[] {11, 12, 13, 14};
        data[1] = new int[] {0, 22, 23};
        data[2] = new int[] {0, 0, 33, 34, 35, 36};
        data[3] = new int[] {0, 0, 0, 44, 45};
        data[4] = new int[] {0, 0, 0, 0, 0, 56};
        data[5] = new int[] {0, 0, 0, 0, 0, 66};
        return data;
    }

    private static int[][] columnMajorData() {
        /*
        11  12  13  14  0   0
        0   22  23  0   0   0
        0   0   33  34  35  36
        0   0   0   44  45  0
        0   0   0   0   0   56
        0   0   0   0   0   66
        */
        int[][] data = new int[6][];
        data[0] = new int[] {11};
        data[1] = new int[] {12, 22};
        data[2] = new int[] {13, 23, 33};
        data[3] = new int[] {14, 0, 34, 44};
        data[4] = new int[] {0, 0, 35, 45};
        data[5] = new int[] {0, 0, 36, 0, 56, 66};
        return data;
    }

}
