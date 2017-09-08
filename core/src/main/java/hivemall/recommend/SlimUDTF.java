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
import hivemall.annotations.VisibleForTesting;
import hivemall.common.ConversionState;
import hivemall.math.matrix.sparse.DoKMatrix;
import hivemall.math.vector.VectorProcedure;
import hivemall.utils.collections.maps.Int2FloatOpenHashTable;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.FileUtils;
import hivemall.utils.io.NioStatefullSegment;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;
import hivemall.utils.lang.mutable.MutableDouble;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.*;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.*;


public class SlimUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(SlimUDTF.class);

    private double l1;
    private double l2;
    private int numIterations;
    private int previousItemId;

    private transient DoKMatrix weightMatrix; // item-item weight matrix
    private transient DoKMatrix dataMatrix; // item-user matrix to get the number of nnz values in column

    private PrimitiveObjectInspector itemIOI;
    private PrimitiveObjectInspector itemJOI;
    private MapObjectInspector riOI;
    private MapObjectInspector rjOI;

    private MapObjectInspector knnItemsOI;
    private PrimitiveObjectInspector knnItemsKeyOI;
    private MapObjectInspector knnItemsValueOI;
    private PrimitiveObjectInspector knnItemsValueKeyOI;
    private PrimitiveObjectInspector knnItemsValueValueOI;

    private PrimitiveObjectInspector riKeyOI;
    private PrimitiveObjectInspector riValueOI;

    private PrimitiveObjectInspector rjKeyOI;
    private PrimitiveObjectInspector rjValueOI;

    // used to store KNN data into temporary file for iterative training
    private NioStatefullSegment fileIO;
    private ByteBuffer inputBuf;

    private ConversionState cvState;
    private long observedTrainingExamples;

    public SlimUDTF() {}

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;
        if (numArgs != 5 && numArgs != 6) {
            throw new UDFArgumentException(
                "_FUNC_ takes arguments: int i, map<int, double> r_i, map<int, map<int, double>> topKRatesOfI, int j, map<int, double> r_j, [, constant string options]");
        }

        this.itemIOI = HiveUtils.asIntCompatibleOI(argOIs[0]);

        this.riOI = HiveUtils.asMapOI(argOIs[1]);
        this.riKeyOI = HiveUtils.asIntCompatibleOI((this.riOI.getMapKeyObjectInspector()));
        this.riValueOI = HiveUtils.asPrimitiveObjectInspector((this.riOI.getMapValueObjectInspector()));

        this.knnItemsOI = HiveUtils.asMapOI(argOIs[2]);
        this.knnItemsKeyOI = HiveUtils.asIntCompatibleOI(knnItemsOI.getMapKeyObjectInspector());
        this.knnItemsValueOI = HiveUtils.asMapOI(knnItemsOI.getMapValueObjectInspector());
        this.knnItemsValueKeyOI = HiveUtils.asIntCompatibleOI(knnItemsValueOI.getMapKeyObjectInspector());
        this.knnItemsValueValueOI = HiveUtils.asDoubleCompatibleOI(knnItemsValueOI.getMapValueObjectInspector());

        this.itemJOI = HiveUtils.asIntCompatibleOI(argOIs[3]);

        this.rjOI = HiveUtils.asMapOI(argOIs[4]);
        this.rjKeyOI = HiveUtils.asIntCompatibleOI((this.rjOI.getMapKeyObjectInspector()));
        this.rjValueOI = HiveUtils.asPrimitiveObjectInspector((this.rjOI.getMapValueObjectInspector()));

        processOptions(argOIs);

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("j");
        fieldNames.add("nn");
        fieldNames.add("w");

        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        this.observedTrainingExamples = 0L;
        this.previousItemId = -2147483648;

        this.dataMatrix = null;
        this.weightMatrix = null;

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("l1", "l1coefficient", true,
            "Coefficient for l1 regularizer [default: 0.001]");
        opts.addOption("l2", "l2coefficient", true,
            "Coefficient for l2 regularizer [default: 0.0005]");
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
        double l1 = 0.001d;
        double l2 = 0.0005d;
        int numIterations = 40;
        boolean conversionCheck = true;
        double cv_rate = 0.005d;

        if (argOIs.length >= 6) {
            String rawArgs = HiveUtils.getConstString(argOIs[5]);
            cl = parseOptions(rawArgs);

            l1 = Primitives.parseDouble(cl.getOptionValue("l1"), l1);
            if (l1 < 0.d) {
                throw new UDFArgumentException("Argument `double l1` must be non-negative: " + l1);
            }

            l2 = Primitives.parseDouble(cl.getOptionValue("l2"), l2);
            if (l2 < 0.d) {
                throw new UDFArgumentException("Argument `double l2` must be non-negative: " + l2);
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

    @SuppressWarnings("unchecked")
    @Override
    public void process(Object[] args) throws HiveException {
        if (this.dataMatrix == null) {
            this.dataMatrix = new DoKMatrix();
        }

        if (this.weightMatrix == null) {
            this.weightMatrix = new DoKMatrix();
        }

        int itemI = PrimitiveObjectInspectorUtils.getInt(args[0], itemIOI);
        Int2FloatOpenHashTable ri = map2Int2FloatOpenHashTable(this.riOI.getMap(args[1]), riKeyOI,
            riValueOI);


        Map<Integer, Int2FloatOpenHashTable> knnItems = new HashMap<>();
        for (Map.Entry<?, ?> entry : this.knnItemsOI.getMap(args[2]).entrySet()) {
            int user = PrimitiveObjectInspectorUtils.getInt(entry.getKey(), this.knnItemsKeyOI);
            Int2FloatOpenHashTable ru = map2Int2FloatOpenHashTable(
                this.knnItemsValueOI.getMap(entry.getValue()), knnItemsValueKeyOI,
                knnItemsValueValueOI);
            knnItems.put(user, ru);
        }

        int itemJ = PrimitiveObjectInspectorUtils.getInt(args[3], itemJOI);
        Int2FloatOpenHashTable rj = map2Int2FloatOpenHashTable(this.rjOI.getMap(args[4]), rjKeyOI,
            rjValueOI);

        train(itemI, ri, knnItems, itemJ, rj);
        observedTrainingExamples++;

        if (this.numIterations == 1) {
            return;
        }

        if (this.previousItemId != itemI) {
            this.previousItemId = itemI;

            // store Ri
            final Int2FloatOpenHashTable.IMapIterator itor = ri.entries();
            while (itor.next() != -1) {
                this.dataMatrix.unsafeSet(itemI, itor.getKey(), itor.getValue());
            }

            recordTrainingInput(itemI, knnItems);
        }
    }

    private void recordTrainingInput(int itemI, Map<Integer, Int2FloatOpenHashTable> knnItems)
            throws HiveException {

        // initialize temporary file to save knn for iterative training
        ByteBuffer buf = inputBuf;
        NioStatefullSegment dst = fileIO;

        if (buf == null) {
            // invoke only at task node (initialize is also invoked in compilation)
            final File file;
            try {
                file = File.createTempFile("hivemall_slim", ".sgmt"); // to save KNN data
                file.deleteOnExit();
                if (!file.canWrite()) {
                    throw new UDFArgumentException("Cannot write a temporary file: "
                            + file.getAbsolutePath());
                }
            } catch (IOException ioe) {
                throw new UDFArgumentException(ioe);
            }

            this.inputBuf = buf = ByteBuffer.allocateDirect(8 * 1024 * 1024); // 8MB
            this.fileIO = dst = new NioStatefullSegment(file, false);
        }

        // count element size size: i, numKNN, [[u, numKNNu, [[item, rate], ...], ...]
        int numElementOfKNNItems = 0;
        for (Int2FloatOpenHashTable c : knnItems.values()) {
            numElementOfKNNItems += c.size();
        }

        int recordBytes = SizeOf.INT + SizeOf.INT + SizeOf.INT * 2 * knnItems.size()
                + (SizeOf.INT + SizeOf.FLOAT) * numElementOfKNNItems;
        int requiredBytes = SizeOf.INT + recordBytes; // need to allocate space for "recordBytes" itself


        int remain = buf.remaining();
        if (remain < requiredBytes) {
            writeBuffer(buf, dst);
        }

        buf.putInt(recordBytes);
        buf.putInt(itemI);
        buf.putInt(knnItems.size());
        for (Map.Entry<Integer, Int2FloatOpenHashTable> ruEntry : knnItems.entrySet()) {
            int user = ruEntry.getKey();
            Int2FloatOpenHashTable ru = ruEntry.getValue();

            buf.putInt(user);
            buf.putInt(ru.size());

            final Int2FloatOpenHashTable.IMapIterator itor = ru.entries();
            while (itor.next() != -1) {
                buf.putInt(itor.getKey());
                buf.putFloat(itor.getValue());
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
        forwardModel();
    }

    protected float predict(int user, int itemI, Map<Integer, Int2FloatOpenHashTable> knnItems,
            int excludeIndex) {
        if (!knnItems.containsKey(user)) {
            return 0.f;
        }
        float pred = 0.f;
        final Int2FloatOpenHashTable.IMapIterator itor = knnItems.get(user).entries();
        while (itor.next() != -1) {
            int itemK = itor.getKey();
            if (itemK == excludeIndex) {
                continue;
            }
            float ruk = itor.getValue();
            pred += ruk * this.weightMatrix.unsafeGet(itemI, itemK, 0.f);
        }
        return pred;
    }

    private void train(int itemI, Int2FloatOpenHashTable ri,
            Map<Integer, Int2FloatOpenHashTable> knnItems, int itemJ, Int2FloatOpenHashTable rj) {
        int N = rj.size();
        double gradSum = 0.d;
        double rateSum = 0.d;
        double lossSum = 0.d;

        final Int2FloatOpenHashTable.IMapIterator itor = rj.entries();
        while (itor.next() != -1) {
            int user = itor.getKey();
            double ruj = itor.getValue();
            double rui = 0.d;
            if (ri.containsKey(user)) {
                rui = ri.get(user);
            }

            double eui = rui - predict(user, itemI, knnItems, itemJ);
            gradSum += ruj * eui;
            rateSum += ruj * ruj;
            lossSum += eui * eui;

            if (this.numIterations > 1) {
                this.dataMatrix.unsafeSet(itemJ, user, ruj);
            }
        }

        gradSum /= N;
        rateSum /= N;
        double wij = weightMatrix.unsafeGet(itemI, itemJ, 0.d);
        double loss = lossSum / N + 0.5 * this.l2 * wij * wij + this.l1 * wij;
        cvState.incrLoss(loss);

        this.weightMatrix.unsafeSet(itemI, itemJ, getUpdateTerm(gradSum, rateSum, this.l1, this.l2));
    }

    private void train(final int itemI, final Map<Integer, Int2FloatOpenHashTable> knnItems,
            final int itemJ) {

        final int N = this.dataMatrix.numColumns(itemJ);
        final MutableDouble mutableGradSum = new MutableDouble(0.d);
        final MutableDouble mutableRateSum = new MutableDouble(0.d);
        final MutableDouble mutableLossSum = new MutableDouble(0.d);


        this.dataMatrix.eachNonZeroInRow(itemJ, new VectorProcedure() {
            @Override
            public void apply(int user, double ruj) {
                double rui = dataMatrix.unsafeGet(itemI, user, 0.d);
                double eui = rui - predict(user, itemI, knnItems, itemJ);

                mutableGradSum.addValue(ruj * eui);
                mutableRateSum.addValue(ruj * ruj);
                mutableLossSum.addValue(eui * eui);
            }
        });

        double gradSum = mutableGradSum.getValue() / N;
        double rateSum = mutableRateSum.getValue() / N;
        double wij = this.weightMatrix.unsafeGet(itemI, itemJ, 0.d);
        double loss = mutableLossSum.getValue() / N + 0.5 * this.l2 * wij * wij + this.l1 * wij;
        cvState.incrLoss(loss);

        this.weightMatrix.unsafeSet(itemI, itemJ, getUpdateTerm(gradSum, rateSum, this.l1, this.l2));
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
            // non-negativity constraints
            if (update < 0.d) {
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

        final Reporter reporter = getReporter();
        final Counters.Counter iterCounter = (reporter == null) ? null : reporter.getCounter(
            "hivemall.recommend.slim$Counter", "iteration");

        try {
            if (dst.getPosition() == 0L) {// run iterations w/o temporary file
                if (buf.position() == 0) {
                    return; // no training example
                }
                buf.flip();
                int iter = 2;
                for (; iter < this.numIterations; iter++) {
                    cvState.next();
                    reportProgress(reporter);
                    setCounterValue(iterCounter, iter);

                    while (buf.remaining() > 0) {
                        int recordBytes = buf.getInt();
                        assert (recordBytes > 0) : recordBytes;
                        trainFromBuffer(buf);
                    }
                    buf.rewind();
                    if (cvState.isConverged(observedTrainingExamples)) {
                        break;
                    }

                }
                logger.info("Performed "
                        + cvState.getCurrentIteration()
                        + " iterations of "
                        + NumberUtils.formatNumber(observedTrainingExamples)
                        + " training examples on memory (thus "
                        + NumberUtils.formatNumber(observedTrainingExamples
                                * cvState.getCurrentIteration()) + " training updates in total) ");

            } else { // read training examples in the temporary file and invoke train for each example
                // write KNNi in buffer to a temporary file
                if (buf.remaining() > 0) {
                    writeBuffer(buf, dst);
                }

                try {
                    dst.flush();
                } catch (IOException e) {
                    throw new HiveException("Failed to flush a file: "
                            + dst.getFile().getAbsolutePath(), e);
                }

                if (logger.isInfoEnabled()) {
                    File tmpFile = dst.getFile();
                    logger.info("Wrote KNN data of item i record to a temporary file for iterative training: "
                            + tmpFile.getAbsolutePath()
                            + " ("
                            + FileUtils.prettyFileSize(tmpFile)
                            + ")");
                }

                // run iterations
                int iter = 2;
                for (; iter < this.numIterations; iter++) {
                    cvState.next();
                    setCounterValue(iterCounter, iter);

                    buf.clear();
                    dst.resetPosition();
                    while (true) {
                        reportProgress(reporter);
                        // load a KNNi to a buffer in the temporary file
                        final int bytesRead;
                        try {
                            bytesRead = dst.read(buf);
                        } catch (IOException e) {
                            throw new HiveException("Failed to read a file: "
                                    + dst.getFile().getAbsolutePath(), e);
                        }
                        if (bytesRead == 0) { // reached file EOF
                            break;
                        }
                        assert (bytesRead > 0) : bytesRead;

                        // reads training examples from a buffer
                        buf.flip();
                        int remain = buf.remaining();
                        if (remain < SizeOf.INT) {
                            throw new HiveException("Illegal file format was detected");
                        }
                        while (remain >= SizeOf.INT) {
                            int pos = buf.position();
                            int recordBytes = buf.getInt();
                            remain -= SizeOf.INT;
                            if (remain < recordBytes) {
                                buf.position(pos);
                                break;
                            }

                            trainFromBuffer(buf);
                            remain -= recordBytes;
                        }
                        buf.compact();
                    }
                    if (cvState.isConverged(observedTrainingExamples)) {
                        break;
                    }
                }
                logger.info("Performed "
                        + cvState.getCurrentIteration()
                        + " iterations of "
                        + NumberUtils.formatNumber(observedTrainingExamples)
                        + " training examples on memory and KNNi data on secondary storage (thus "
                        + NumberUtils.formatNumber(observedTrainingExamples
                                * cvState.getCurrentIteration()) + " training updates in total) ");

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

    private void trainFromBuffer(ByteBuffer buf) {
        final int itemI = buf.getInt();
        int knnSize = buf.getInt();
        final Map<Integer, Int2FloatOpenHashTable> knnItems = new HashMap<>();
        Set<Integer> pairItems = new HashSet<>();

        for (int i = 0; i < knnSize; i++) {
            int user = buf.getInt();
            int ruSize = buf.getInt();
            Int2FloatOpenHashTable ru = new Int2FloatOpenHashTable(ruSize);
            ru.defaultReturnValue(0.f);

            for (int j = 0; j < ruSize; j++) {
                int itemK = buf.getInt();
                pairItems.add(itemK);
                float ruk = buf.getFloat();
                ru.put(itemK, ruk);
            }
            knnItems.put(user, ru);
        }

        for (int itemJ : pairItems) {
            train(itemI, knnItems, itemJ);
        }
    }

    private void forwardModel() throws HiveException {
        for (int i = 0; i < this.weightMatrix.numRows(); i++) {
            for (int j = 0; j < this.weightMatrix.numColumns(); j++) {
                if (this.weightMatrix.unsafeGet(i, j, 0.d) != 0.d) {
                    Object[] res = new Object[3];
                    res[0] = new IntWritable(i);
                    res[1] = new IntWritable(j);
                    res[2] = new DoubleWritable(this.weightMatrix.get(i, j));
                    forward(res);
                }
            }
        }
        logger.info("Forwarded Slim's weights matrix");
    }

    @VisibleForTesting
    void finalizeTraining() throws HiveException {
        if (this.numIterations > 1) {
            runIterativeTraining();
        }
    }

    private static Int2FloatOpenHashTable map2Int2FloatOpenHashTable(Map<?, ?> map,
            PrimitiveObjectInspector keyOI, PrimitiveObjectInspector valueOI) {
        Int2FloatOpenHashTable result = new Int2FloatOpenHashTable(map.size());
        result.defaultReturnValue(0.f);

        for (Map.Entry<?, ?> entry : map.entrySet()) {
            int k = PrimitiveObjectInspectorUtils.getInt(entry.getKey(), keyOI);
            float v = PrimitiveObjectInspectorUtils.getFloat(entry.getValue(), valueOI);
            result.put(k, v);
        }

        return result;
    }
}
