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
package hivemall.recommend;

import hivemall.UDTFWithOptions;
import hivemall.common.ConversionState;
import hivemall.math.matrix.sparse.DoKMatrix;
import hivemall.math.vector.VectorProcedure;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.NioStatefullSegment;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;
import hivemall.utils.lang.mutable.MutableDouble;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.*;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.*;


public class SlimUDTF extends UDTFWithOptions {
    private double l1;
    private double l2;
    private int numIterations;
    private int previousItemId = -2147483648;

    private final DoKMatrix W = new DoKMatrix(); // item-item weight matrix
    private final DoKMatrix A = new DoKMatrix(); // item-user matrix

    private PrimitiveObjectInspector itemIOI;
    private PrimitiveObjectInspector itemJOI;
    private MapObjectInspector RiOI;
    private MapObjectInspector RjOI;

    private MapObjectInspector KNNiOI;
    private PrimitiveObjectInspector KNNiKeyOI;
    private MapObjectInspector KNNiValueOI;
    private PrimitiveObjectInspector KNNiValueKeyOI;
    private PrimitiveObjectInspector KNNiValueValueOI;

    private PrimitiveObjectInspector RiKeyOI;
    private PrimitiveObjectInspector RiValueOI;

    private PrimitiveObjectInspector RjKeyOI;
    private PrimitiveObjectInspector RjValueOI;

    // Used for iterations
    private NioStatefullSegment fileIO;
    private ByteBuffer inputBuf;

    protected ConversionState cvState;

    public SlimUDTF() {}

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;
        if (numArgs != 5 && numArgs != 6) {
            throw new UDFArgumentException(
                "_FUNC_ takes arguments: int i, map<int, double> r_i, map<int, map<int, double>> topKRatesOfI, int j, map<int, double> r_j, [, constant string options]");
        }

        this.itemIOI = HiveUtils.asIntCompatibleOI(argOIs[0]);

        this.RiOI = HiveUtils.asMapOI(argOIs[1]);
        this.RiKeyOI = HiveUtils.asIntCompatibleOI((this.RiOI.getMapKeyObjectInspector()));
        this.RiValueOI = HiveUtils.asDoubleCompatibleOI((this.RiOI.getMapValueObjectInspector()));

        this.KNNiOI = HiveUtils.asMapOI(argOIs[2]);
        this.KNNiKeyOI = HiveUtils.asIntCompatibleOI(KNNiOI.getMapKeyObjectInspector());
        this.KNNiValueOI = HiveUtils.asMapOI(KNNiOI.getMapValueObjectInspector());
        this.KNNiValueKeyOI = HiveUtils.asIntCompatibleOI(KNNiValueOI.getMapKeyObjectInspector());
        this.KNNiValueValueOI = HiveUtils.asDoubleCompatibleOI(KNNiValueOI.getMapValueObjectInspector());

        this.itemJOI = HiveUtils.asIntCompatibleOI(argOIs[3]);

        this.RjOI = HiveUtils.asMapOI(argOIs[4]);
        this.RjKeyOI = HiveUtils.asIntCompatibleOI((this.RjOI.getMapKeyObjectInspector()));
        this.RjValueOI = HiveUtils.asDoubleCompatibleOI((this.RjOI.getMapValueObjectInspector()));

        processOptions(argOIs);

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        // initialize temporary file to save knn for iterative training
        if (mapredContext != null && numIterations > 1) {
            // invoke only at task node (initialize is also invoked in compilation)
            final File file;
            try {
                file = File.createTempFile("hivemall_slim", ".sgmt"); // A, Knn and R
                file.deleteOnExit();
                if (!file.canWrite()) {
                    throw new UDFArgumentException("Cannot write a temporary file: "
                            + file.getAbsolutePath());
                }
            } catch (IOException ioe) {
                throw new UDFArgumentException(ioe);
            }

            this.fileIO = new NioStatefullSegment(file, false);
        }

        fieldNames.add("i");
        fieldNames.add("j");
        fieldNames.add("wij");

        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("l1", "l1coefficient", true,
            "Coefficient for l1 regularizer [default: 0.01]");
        opts.addOption("l2", "l2coefficient", true,
            "Coefficient for l2 regularizer [default: 0.01]");
        opts.addOption("numIterations", "iteration", true,
            "The number of iterations for coordinate descent [default: 40]");
        opts.addOption("disable_cv", "disable_cvtest", false,
            "Whether to disable convergence check [default: enabled]");
        opts.addOption("cv_rate", "convergence_rate", true,
            "Threshold to determine convergence [default: 0.005]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        double l1 = 0.01d;
        double l2 = 0.01d;
        int numIterations = 3;
        boolean conversionCheck = true;
        double cv_rate = 0.005d;

        if (argOIs.length >= 6) {
            String rawArgs = HiveUtils.getConstString(argOIs[5]);
            cl = parseOptions(rawArgs);

            l1 = Primitives.parseDouble(cl.getOptionValue("l1"), l1);
            if (l1 < 0.d || l1 > 1.d) {
                throw new UDFArgumentException("Argument `double l1` must be within [0., 1.]: "
                        + l1);
            }

            l2 = Primitives.parseDouble(cl.getOptionValue("l2"), l2);
            if (l2 < 0.d || l2 > 1.d) {
                throw new UDFArgumentException("Argument `double l2` must be within [0., 1.]: "
                        + l2);
            }

            numIterations = Primitives.parseInt(cl.getOptionValue("numIterations"), numIterations);
            if (numIterations <= 0) {
                throw new UDFArgumentException(
                    "Argument `int numIterations` must be greater than 0: " + numIterations);
            }

            conversionCheck = !cl.hasOption("disable_cvtest");

            cv_rate = Primitives.parseDouble(cl.getOptionValue("cv_rate"), cv_rate);
            if (cv_rate <= 0) {
                throw new UDFArgumentException(
                    "Argument `double cv_rate` must be greater than 0.0: " + cv_rate);
            }
        }

        this.l1 = l1;
        this.l2 = l2;
        this.numIterations = numIterations;
        this.cvState = new ConversionState(conversionCheck, cv_rate);

        return cl;
    }

    @Override
    public void process(Object[] args) throws HiveException {
        int itemI = PrimitiveObjectInspectorUtils.getInt(args[0], itemIOI);
        Map Ri = this.RiOI.getMap(args[1]);
        Map KNNi = this.KNNiOI.getMap(args[2]);
        int itemJ = PrimitiveObjectInspectorUtils.getInt(args[3], itemJOI);
        Map Rj = this.RjOI.getMap(args[4]);
        train(itemI, Ri, KNNi, itemJ, Rj);

        if (this.numIterations == 1) {
            return;
        }

        if (this.previousItemId != itemI) {
            this.previousItemId = itemI;

            // store Ri
            for (Map.Entry<?, ?> ruiEntry : ((Map<?, ?>) Ri).entrySet()) {
                int user = PrimitiveObjectInspectorUtils.getInt(ruiEntry.getKey(), this.RiKeyOI);
                double rui = PrimitiveObjectInspectorUtils.getDouble(ruiEntry.getValue(),
                    this.RiValueOI);
                this.A.unsafeSet(itemI, user, rui); // need optimize
            }

            recordTrainingInput(itemI, this.KNNiOI.getMap(KNNi));
        }
    }

    private void recordTrainingInput(int itemI, Map<?, ?> KNNi) throws HiveException {
        // count element size size: i, numKNN, [[u, numKNNu, [[item, rate], ...], ...]
        int numElementOfKNNi = 0;
        for (Map.Entry<?, ?> RuEntry : KNNi.entrySet()) {
            numElementOfKNNi += this.KNNiValueOI.getMap(RuEntry.getValue()).size();
        }

        int recordBytes = SizeOf.INT + SizeOf.INT + SizeOf.INT * 2 * KNNi.size()
                + (SizeOf.DOUBLE + SizeOf.INT) * numElementOfKNNi;
        int requiredBytes = SizeOf.INT + recordBytes; // need to allocate space for "recordBytes" itself

        if (this.inputBuf == null) {
            this.inputBuf = ByteBuffer.allocateDirect(requiredBytes);
        }

        ByteBuffer buf = inputBuf;
        NioStatefullSegment dst = fileIO;

        int remain = buf.remaining();
        if (remain < requiredBytes) {
            writeBuffer(buf, dst);
        }

        buf.putInt(itemI);
        buf.putInt(KNNi.size());
        for (Map.Entry<?, ?> RuEntry : this.KNNiOI.getMap(KNNi).entrySet()) {
            int user = PrimitiveObjectInspectorUtils.getInt(RuEntry.getKey(), this.KNNiKeyOI);
            Map<?, ?> Ru = this.KNNiValueOI.getMap(RuEntry.getValue());

            buf.putInt(user);
            buf.putInt(Ru.size());

            for (Map.Entry<?, ?> rukEntry : Ru.entrySet()) {
                int itemK = PrimitiveObjectInspectorUtils.getInt(rukEntry.getKey(),
                    this.KNNiValueKeyOI);
                double ruk = PrimitiveObjectInspectorUtils.getDouble(rukEntry.getValue(),
                    this.KNNiValueValueOI);

                buf.putInt(itemK);
                buf.putDouble(ruk);
            }
        }
    }


    private static void writeBuffer(@Nonnull ByteBuffer srcBuf, @Nonnull NioStatefullSegment dst)
            throws HiveException {
        srcBuf.flip();
        try {
            dst.write(srcBuf);
        } catch (IOException e) {
            throw new HiveException("Exception causes while writing a buffer to file", e);
        }
        srcBuf.clear();
    }

    @Override
    public void close() throws HiveException {

        runIterativeTraining();

        int numItem = Math.max(this.W.numRows(), this.W.numColumns());
        for (int i = 0; i < numItem; i++) {
            for (int j = 0; j < numItem; j++) {
                if (this.W.unsafeGet(i, j, 0.d) != 0.d) {
                    Object[] res = new Object[3];
                    res[0] = new IntWritable(i);
                    res[1] = new IntWritable(j);
                    res[2] = new DoubleWritable(this.W.get(i, j));
                    forward(res);
                }
            }
        }
    }

    protected double predict(int user, int itemI, Map<?, ?> KNNi, int excludeIndex) {
        if (!KNNi.containsKey(user)) {
            return 0.d;
        }
        double pred = 0.d;
        for (Map.Entry<?, ?> rukEntry : this.KNNiValueOI.getMap(KNNi.get(user)).entrySet()) {
            int itemK = PrimitiveObjectInspectorUtils.getInt(rukEntry.getKey(), this.KNNiValueKeyOI);
            if (itemK == excludeIndex) {
                continue;
            }
            double ruk = PrimitiveObjectInspectorUtils.getDouble(rukEntry.getValue(),
                this.KNNiValueValueOI);
            pred += ruk * this.W.unsafeGet(itemI, itemK, 0.d);
        }
        return pred;
    }

    protected double predict(int user, int itemI, Map<?, ?> KNNi) {
        if (!KNNi.containsKey(user)) {
            return 0.d;
        }

        double pred = 0.d;
        for (Map.Entry<?, ?> rukEntry : this.KNNiValueOI.getMap(KNNi.get(user)).entrySet()) {
            int itemK = PrimitiveObjectInspectorUtils.getInt(rukEntry.getKey(), this.KNNiValueKeyOI);
            double ruk = PrimitiveObjectInspectorUtils.getDouble(rukEntry.getValue(),
                this.KNNiValueValueOI);

            pred += ruk * this.W.unsafeGet(itemI, itemK, 0.d);
        }
        return pred;
    }

    protected double predict4Iterative(int user, int itemI,
            Map<Integer, Map<Integer, Double>> KNNi, int excludeIndex) {
        if (!KNNi.containsKey(user)) {
            return 0.d;
        }

        double pred = 0.d;
        for (Map.Entry<Integer, Double> rukEntry : KNNi.get(user).entrySet()) {
            int itemK = rukEntry.getKey();
            if (itemK == excludeIndex)
                continue;
            double ruk = rukEntry.getValue();

            pred += ruk * this.W.unsafeGet(itemI, itemK, 0.d);
        }

        return pred;
    }

    private void train(int itemI, Map<?, ?> Ri, Map<?, ?> KNNi, int itemJ, Map<?, ?> Rj) {
        int N = Rj.size();
        double gradSum = 0.d;
        double rateSum = 0.d;

        for (Map.Entry<?, ?> rujEntry : Rj.entrySet()) {
            int user = PrimitiveObjectInspectorUtils.getInt(rujEntry.getKey(), this.RjKeyOI);
            double ruj = PrimitiveObjectInspectorUtils.getDouble(rujEntry.getValue(),
                this.RjValueOI);
            double rui = 0.d;
            if (Ri.containsKey(user)) {
                rui = PrimitiveObjectInspectorUtils.getDouble(Ri.get(user), this.RiValueOI);
            }

            double eui = rui - predict(user, itemI, KNNi, itemJ);
            gradSum += ruj * eui;
            rateSum += ruj * ruj;

            if (this.numIterations > 1) {
                this.A.unsafeSet(itemJ, user, ruj);
            }
        }

        gradSum /= N;
        rateSum /= N;

        this.W.unsafeSet(itemI, itemJ, getUpdateTerm(gradSum, rateSum, this.l1, this.l2));
    }

    private double train(final int itemI, final Map<Integer, Map<Integer, Double>> KNNi,
            final int itemJ) {

        int N = this.A.numColumns(itemJ);
        final MutableDouble mutableGradSum = new MutableDouble(0.d);
        final MutableDouble mutableRateSum = new MutableDouble(0.d);
        final MutableDouble mutableLossSum = new MutableDouble(0.d);

        this.A.eachNonZeroInRow(itemJ, new VectorProcedure() {
            @Override
            public void apply(int user, double ruj) {
                double rui = A.get(itemI, user, 0.d);
                double eui = rui - predict4Iterative(itemI, user, KNNi, itemJ);

                mutableGradSum.addValue(ruj * eui);
                mutableRateSum.addValue(ruj * ruj);
                mutableLossSum.addValue(eui * eui);
            }
        });

        double gradSum = mutableGradSum.getValue() / N;
        double rateSum = mutableRateSum.getValue() / N;
        double wij = W.unsafeGet(itemI, itemJ, 0.d);
        double loss = mutableLossSum.getValue() / N + 0.5 * this.l2 * wij * wij + this.l1 * wij;

        this.W.unsafeSet(itemI, itemJ, getUpdateTerm(gradSum, rateSum, this.l1, this.l2));

        return loss;
    }

    private static double getUpdateTerm(final double gradSum, final double rateSum,
            final double l1, final double l2) {
        double update = 0.d;
        if (l1 < Math.abs(gradSum)) {
            if (gradSum > 0.d) {
                update = (gradSum - l1) / (rateSum + l2);
            } else {
                update = (gradSum + l1) / (rateSum + l2);
            }
            if (update < 0.d) { // non-negativity constraints
                update = 0.d;
            }
        }
        return update;
    }

    private void runIterativeTraining() throws HiveException {
        final ByteBuffer buf = this.inputBuf;
        final NioStatefullSegment dst = this.fileIO;
        assert (buf != null);
        assert (dst != null);

        try {
            if (dst.getPosition() == 0L) {// run iterations w/o temporary file
                if (buf.position() == 0) {
                    return; // no training example
                }
                buf.flip();
                int iter = 2;
                for (; iter < this.numIterations; iter++) {
                    cvState.next();

                    while (buf.remaining() > 0) {
                        final int itemI = buf.getInt();
                        int knnSize = buf.getInt();
                        final Map<Integer, Map<Integer, Double>> KNNi = new HashMap<>();
                        for (int i = 0; i < knnSize; i++) {
                            int user = buf.getInt();
                            int RuSize = buf.getInt();
                            Map<Integer, Double> Ru = new HashMap<>();
                            for (int j = 0; j < RuSize; j++) {
                                int itemK = buf.getInt();
                                double ruk = buf.getDouble();
                                Ru.put(itemK, ruk);
                            }
                            KNNi.put(user, Ru);
                        }

                        this.W.eachNonZeroInRow(itemI, new VectorProcedure() {
                            @Override
                            public void apply(int itemJ, double wij) {
                                cvState.incrLoss(train(itemI, KNNi, itemJ));
                            }
                        });
                    }
                    buf.rewind();
                }
            } else {

            }
        } catch (Throwable e) {
            throw new HiveException("Exception caused in the iterative training", e);
        } finally {
            // delete the temporary file and release resources
            try {
                dst.close(true);
            } catch (IOException e) {
                throw new HiveException("Failed to close a file: "
                        + dst.getFile().getAbsolutePath(), e);
            }
            this.inputBuf = null;
            this.fileIO = null;
        }
    }
}
