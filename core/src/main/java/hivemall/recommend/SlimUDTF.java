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
import matrix4j.matrix.FloatMatrix;
import matrix4j.matrix.sparse.floats.DoKFloatMatrix;
import matrix4j.vector.VectorProcedure;
import hivemall.utils.collections.Fastutil;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.FileUtils;
import hivemall.utils.io.NioStatefulSegment;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;
import hivemall.utils.lang.mutable.MutableDouble;
import hivemall.utils.lang.mutable.MutableInt;
import hivemall.utils.lang.mutable.MutableObject;
import it.unimi.dsi.fastutil.ints.Int2FloatMap;
import it.unimi.dsi.fastutil.ints.Int2FloatOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;
import it.unimi.dsi.fastutil.ints.Int2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.ints.IntOpenHashSet;
import it.unimi.dsi.fastutil.ints.IntSet;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

/**
 * Sparse Linear Methods (SLIM) for Top-N Recommender Systems.
 *
 * <pre>
 * Xia Ning and George Karypis, SLIM: Sparse Linear Methods for Top-N Recommender Systems, Proc. ICDM, 2011.
 * </pre>
 */
@Description(name = "train_slim",
        value = "_FUNC_( int i, map<int, double> r_i, map<int, map<int, double>> topKRatesOfI, int j, map<int, double> r_j [, constant string options]) "
                + "- Returns row index, column index and non-zero weight value of prediction model")
public class SlimUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(SlimUDTF.class);

    //--------------------------------------------
    // input OIs

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

    //--------------------------------------------
    // hyperparameters

    private double l1;
    private double l2;
    private int numIterations;

    //--------------------------------------------
    // model parameters and else

    /** item-item weight matrix */
    private transient DoKFloatMatrix _weightMatrix;

    //--------------------------------------------
    // caching for each item i

    private int _previousItemId;

    @Nullable
    private transient Int2FloatMap _ri;
    @Nullable
    private transient Int2ObjectMap<Int2FloatMap> _kNNi;
    /** The number of elements in kNNi */
    @Nullable
    private transient MutableInt _nnzKNNi;

    //--------------------------------------------
    // variables for iteration supports

    /** item-user matrix holding the input data */
    @Nullable
    private transient FloatMatrix _dataMatrix;

    // used to store KNN data into temporary file for iterative training
    private transient NioStatefulSegment _fileIO;
    private transient ByteBuffer _inputBuf;

    private ConversionState _cvState;
    private long _observedTrainingExamples;

    //--------------------------------------------

    public SlimUDTF() {}

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;

        if (numArgs == 1 && HiveUtils.isConstString(argOIs[0])) {// for -help option
            String rawArgs = HiveUtils.getConstString(argOIs[0]);
            parseOptions(rawArgs);
        }

        if (numArgs != 5 && numArgs != 6) {
            throw new UDFArgumentException(
                "_FUNC_ takes 5 or 6 arguments: int i, map<int, double> r_i, map<int, map<int, double>> topKRatesOfI, int j, map<int, double> r_j [, constant string options]: "
                        + Arrays.toString(argOIs));
        }

        this.itemIOI = HiveUtils.asIntCompatibleOI(argOIs[0]);

        this.riOI = HiveUtils.asMapOI(argOIs[1]);
        this.riKeyOI = HiveUtils.asIntCompatibleOI((riOI.getMapKeyObjectInspector()));
        this.riValueOI = HiveUtils.asPrimitiveObjectInspector((riOI.getMapValueObjectInspector()));

        this.knnItemsOI = HiveUtils.asMapOI(argOIs[2]);
        this.knnItemsKeyOI = HiveUtils.asIntCompatibleOI(knnItemsOI.getMapKeyObjectInspector());
        this.knnItemsValueOI = HiveUtils.asMapOI(knnItemsOI.getMapValueObjectInspector());
        this.knnItemsValueKeyOI =
                HiveUtils.asIntCompatibleOI(knnItemsValueOI.getMapKeyObjectInspector());
        this.knnItemsValueValueOI =
                HiveUtils.asDoubleCompatibleOI(knnItemsValueOI.getMapValueObjectInspector());

        this.itemJOI = HiveUtils.asIntCompatibleOI(argOIs[3]);

        this.rjOI = HiveUtils.asMapOI(argOIs[4]);
        this.rjKeyOI = HiveUtils.asIntCompatibleOI((rjOI.getMapKeyObjectInspector()));
        this.rjValueOI = HiveUtils.asPrimitiveObjectInspector((rjOI.getMapValueObjectInspector()));

        processOptions(argOIs);

        this._observedTrainingExamples = 0L;
        this._previousItemId = Integer.MIN_VALUE;
        this._weightMatrix = null;
        this._dataMatrix = null;

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("j");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("nn");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("w");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("l1", "l1coefficient", true,
            "Coefficient for l1 regularizer [default: 0.001]");
        opts.addOption("l2", "l2coefficient", true,
            "Coefficient for l2 regularizer [default: 0.0005]");
        opts.addOption("iters", "iterations", true,
            "The number of iterations for coordinate descent [default: 30]");
        opts.addOption("disable_cv", "disable_cvtest", false,
            "Whether to disable convergence check [default: enabled]");
        opts.addOption("cv_rate", "convergence_rate", true,
            "Threshold to determine convergence [default: 0.005]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        CommandLine cl = null;
        double l1 = 0.001d;
        double l2 = 0.0005d;
        int numIterations = 30;
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

            numIterations = Primitives.parseInt(cl.getOptionValue("iters"), numIterations);
            if (numIterations <= 0) {
                throw new UDFArgumentException(
                    "Argument `int iters` must be greater than 0: " + numIterations);
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
        this._cvState = new ConversionState(conversionCheck, cv_rate);

        return cl;
    }

    @Override
    public void process(@Nonnull Object[] args) throws HiveException {
        if (_weightMatrix == null) {// initialize variables
            this._weightMatrix = new DoKFloatMatrix();
            if (numIterations >= 2) {
                this._dataMatrix = new DoKFloatMatrix();
            }
            this._nnzKNNi = new MutableInt();
        }

        final int itemI = PrimitiveObjectInspectorUtils.getInt(args[0], itemIOI);

        if (itemI != _previousItemId || _ri == null) {
            // cache Ri and kNNi
            this._ri =
                    int2floatMap(itemI, riOI.getMap(args[1]), riKeyOI, riValueOI, _dataMatrix, _ri);
            this._kNNi = kNNentries(args[2], knnItemsOI, knnItemsKeyOI, knnItemsValueOI,
                knnItemsValueKeyOI, knnItemsValueValueOI, _kNNi, _nnzKNNi);

            final int numKNNItems = _nnzKNNi.getValue();
            if (numIterations >= 2 && numKNNItems >= 1) {
                recordTrainingInput(itemI, _kNNi, numKNNItems);
            }
            this._previousItemId = itemI;
        }

        int itemJ = PrimitiveObjectInspectorUtils.getInt(args[3], itemJOI);
        Int2FloatMap rj =
                int2floatMap(itemJ, rjOI.getMap(args[4]), rjKeyOI, rjValueOI, _dataMatrix);

        train(itemI, _ri, _kNNi, itemJ, rj);
        _observedTrainingExamples++;
    }

    private void recordTrainingInput(final int itemI,
            @Nonnull final Int2ObjectMap<Int2FloatMap> knnItems, final int numKNNItems)
            throws HiveException {
        ByteBuffer buf = this._inputBuf;
        NioStatefulSegment dst = this._fileIO;

        if (buf == null) {
            // invoke only at task node (initialize is also invoked in compilation)
            final File file;
            try {
                file = File.createTempFile("hivemall_slim", ".sgmt"); // to save KNN data
                file.deleteOnExit();
                if (!file.canWrite()) {
                    throw new UDFArgumentException(
                        "Cannot write a temporary file: " + file.getAbsolutePath());
                }
            } catch (IOException ioe) {
                throw new UDFArgumentException(ioe);
            }

            this._inputBuf = buf = ByteBuffer.allocateDirect(8 * 1024 * 1024); // 8MB
            this._fileIO = dst = new NioStatefulSegment(file, false);
        }

        int recordBytes = SizeOf.INT + SizeOf.INT + SizeOf.INT * 2 * knnItems.size()
                + (SizeOf.INT + SizeOf.FLOAT) * numKNNItems;
        int requiredBytes = SizeOf.INT + recordBytes; // need to allocate space for "recordBytes" itself

        int remain = buf.remaining();
        if (remain < requiredBytes) {
            writeBuffer(buf, dst);
        }

        buf.putInt(recordBytes);
        buf.putInt(itemI);
        buf.putInt(knnItems.size());

        for (Int2ObjectMap.Entry<Int2FloatMap> e1 : Fastutil.fastIterable(knnItems)) {
            int user = e1.getIntKey();
            buf.putInt(user);

            Int2FloatMap ru = e1.getValue();
            buf.putInt(ru.size());
            for (Int2FloatMap.Entry e2 : Fastutil.fastIterable(ru)) {
                buf.putInt(e2.getIntKey());
                buf.putFloat(e2.getFloatValue());
            }
        }
    }

    private static void writeBuffer(@Nonnull final ByteBuffer srcBuf,
            @Nonnull final NioStatefulSegment dst) throws HiveException {
        srcBuf.flip();
        try {
            dst.write(srcBuf);
        } catch (IOException e) {
            throw new HiveException("Exception causes while writing a buffer to file", e);
        }
        srcBuf.clear();
    }

    private void train(final int itemI, @Nonnull final Int2FloatMap ri,
            @Nonnull final Int2ObjectMap<Int2FloatMap> kNNi, final int itemJ,
            @Nonnull final Int2FloatMap rj) {
        final FloatMatrix W = _weightMatrix;

        final int N = rj.size();
        if (N == 0) {
            return;
        }

        double gradSum = 0.d;
        double rateSum = 0.d;
        double lossSum = 0.d;

        for (Int2FloatMap.Entry e : Fastutil.fastIterable(rj)) {
            int user = e.getIntKey();
            double ruj = e.getFloatValue();
            double rui = ri.get(user); // ri.getOrDefault(user, 0.f);

            double eui = rui - predict(user, itemI, kNNi, itemJ, W);
            gradSum += ruj * eui;
            rateSum += ruj * ruj;
            lossSum += eui * eui;
        }

        gradSum /= N;
        rateSum /= N;

        double wij = W.get(itemI, itemJ, 0.d);
        double loss = lossSum / N + 0.5d * l2 * wij * wij + l1 * wij;
        _cvState.incrLoss(loss);

        W.set(itemI, itemJ, getUpdateTerm(gradSum, rateSum, l1, l2));
    }

    private void train(final int itemI, @Nonnull final Int2ObjectMap<Int2FloatMap> knnItems,
            final int itemJ) {
        final FloatMatrix A = _dataMatrix;
        final FloatMatrix W = _weightMatrix;

        final int N = A.numColumns(itemJ);
        if (N == 0) {
            return;
        }

        final MutableDouble mutableGradSum = new MutableDouble(0.d);
        final MutableDouble mutableRateSum = new MutableDouble(0.d);
        final MutableDouble mutableLossSum = new MutableDouble(0.d);

        A.eachNonZeroInRow(itemJ, new VectorProcedure() {
            @Override
            public void apply(int user, double ruj) {
                double rui = A.get(itemI, user, 0.d);
                double eui = rui - predict(user, itemI, knnItems, itemJ, W);

                mutableGradSum.addValue(ruj * eui);
                mutableRateSum.addValue(ruj * ruj);
                mutableLossSum.addValue(eui * eui);
            }
        });

        double gradSum = mutableGradSum.getValue() / N;
        double rateSum = mutableRateSum.getValue() / N;

        double wij = W.get(itemI, itemJ, 0.d);
        double loss = mutableLossSum.getValue() / N + 0.5 * l2 * wij * wij + l1 * wij;
        _cvState.incrLoss(loss);

        W.set(itemI, itemJ, getUpdateTerm(gradSum, rateSum, l1, l2));
    }

    private static double predict(final int user, final int itemI,
            @Nonnull final Int2ObjectMap<Int2FloatMap> knnItems, final int excludeIndex,
            @Nonnull final FloatMatrix weightMatrix) {
        final Int2FloatMap kNNu = knnItems.get(user);
        if (kNNu == null) {
            return 0.d;
        }

        double pred = 0.d;
        for (Int2FloatMap.Entry e : Fastutil.fastIterable(kNNu)) {
            final int itemK = e.getIntKey();
            if (itemK == excludeIndex) {
                continue;
            }
            float ruk = e.getFloatValue();
            pred += ruk * weightMatrix.get(itemI, itemK, 0.d);
        }
        return pred;
    }

    private static double getUpdateTerm(final double gradSum, final double rateSum, final double l1,
            final double l2) {
        double update = 0.d;
        if (Math.abs(gradSum) > l1) {
            if (gradSum > 0.d) {
                update = (gradSum - l1) / (rateSum + l2);
            } else {
                update = (gradSum + l1) / (rateSum + l2);
            }
            // non-negative constraints
            if (update < 0.d) {
                update = 0.d;
            }
        }
        return update;
    }

    @Override
    public void close() throws HiveException {
        finalizeTraining();
        forwardModel();
        this._weightMatrix = null;
    }

    @VisibleForTesting
    void finalizeTraining() throws HiveException {
        if (numIterations > 1) {
            this._ri = null;
            this._kNNi = null;

            runIterativeTraining();

            this._dataMatrix = null;
        }
    }

    private void runIterativeTraining() throws HiveException {
        final ByteBuffer buf = this._inputBuf;
        final NioStatefulSegment dst = this._fileIO;
        assert (buf != null);
        assert (dst != null);

        final Reporter reporter = getReporter();
        final Counters.Counter iterCounter = (reporter == null) ? null
                : reporter.getCounter("hivemall.recommend.slim$Counter", "iteration");

        try {
            if (dst.getPosition() == 0L) {// run iterations w/o temporary file
                if (buf.position() == 0) {
                    return; // no training example
                }
                buf.flip();
                for (int iter = 2; iter < numIterations; iter++) {
                    _cvState.next();
                    reportProgress(reporter);
                    setCounterValue(iterCounter, iter);

                    while (buf.remaining() > 0) {
                        int recordBytes = buf.getInt();
                        assert (recordBytes > 0) : recordBytes;
                        replayTrain(buf);
                    }
                    buf.rewind();
                    if (_cvState.isConverged(_observedTrainingExamples)) {
                        break;
                    }
                }
                logger.info("Performed " + _cvState.getCurrentIteration() + " iterations of "
                        + NumberUtils.formatNumber(_observedTrainingExamples)
                        + " training examples on memory (thus "
                        + NumberUtils.formatNumber(
                            _observedTrainingExamples * _cvState.getCurrentIteration())
                        + " training updates in total) ");

            } else { // read training examples in the temporary file and invoke train for each example
                // write KNNi in buffer to a temporary file
                if (buf.remaining() > 0) {
                    writeBuffer(buf, dst);
                }

                try {
                    dst.flush();
                } catch (IOException e) {
                    throw new HiveException(
                        "Failed to flush a file: " + dst.getFile().getAbsolutePath(), e);
                }

                if (logger.isInfoEnabled()) {
                    File tmpFile = dst.getFile();
                    logger.info(
                        "Wrote KNN entries of axis items to a temporary file for iterative training: "
                                + tmpFile.getAbsolutePath() + " ("
                                + FileUtils.prettyFileSize(tmpFile) + ")");
                }

                // run iterations
                for (int iter = 2; iter < numIterations; iter++) {
                    _cvState.next();
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
                            throw new HiveException(
                                "Failed to read a file: " + dst.getFile().getAbsolutePath(), e);
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

                            replayTrain(buf);
                            remain -= recordBytes;
                        }
                        buf.compact();
                    }
                    if (_cvState.isConverged(_observedTrainingExamples)) {
                        break;
                    }
                }
                logger.info("Performed " + _cvState.getCurrentIteration() + " iterations of "
                        + NumberUtils.formatNumber(_observedTrainingExamples)
                        + " training examples on memory and KNNi data on secondary storage (thus "
                        + NumberUtils.formatNumber(
                            _observedTrainingExamples * _cvState.getCurrentIteration())
                        + " training updates in total) ");

            }
        } catch (Throwable e) {
            throw new HiveException("Exception caused in the iterative training", e);
        } finally {
            // delete the temporary file and release resources
            try {
                dst.close(true);
            } catch (IOException e) {
                throw new HiveException(
                    "Failed to close a file: " + dst.getFile().getAbsolutePath(), e);
            }
            this._inputBuf = null;
            this._fileIO = null;
        }
    }

    private void replayTrain(@Nonnull final ByteBuffer buf) {
        final int itemI = buf.getInt();
        final int knnSize = buf.getInt();

        final Int2ObjectMap<Int2FloatMap> knnItems = new Int2ObjectOpenHashMap<>(1024);
        final IntSet pairItems = new IntOpenHashSet();
        for (int i = 0; i < knnSize; i++) {
            int user = buf.getInt();
            int ruSize = buf.getInt();
            Int2FloatMap ru = new Int2FloatOpenHashMap(ruSize);
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
        final IntWritable f0 = new IntWritable(); // i
        final IntWritable f1 = new IntWritable(); // nn
        final FloatWritable f2 = new FloatWritable(); // w
        final Object[] forwardObj = new Object[] {f0, f1, f2};

        final MutableObject<HiveException> catched = new MutableObject<>();
        _weightMatrix.eachNonZeroCell(new VectorProcedure() {
            @Override
            public void apply(int i, int j, float value) {
                if (value == 0.f) {
                    return;
                }
                f0.set(i);
                f1.set(j);
                f2.set(value);
                try {
                    forward(forwardObj);
                } catch (HiveException e) {
                    catched.setIfAbsent(e);
                }
            }
        });
        HiveException ex = catched.get();
        if (ex != null) {
            throw ex;
        }
        logger.info("Forwarded SLIM's weights matrix");
    }

    @Nonnull
    private static Int2ObjectMap<Int2FloatMap> kNNentries(@Nonnull final Object kNNiObj,
            @Nonnull final MapObjectInspector knnItemsOI,
            @Nonnull final PrimitiveObjectInspector knnItemsKeyOI,
            @Nonnull final MapObjectInspector knnItemsValueOI,
            @Nonnull final PrimitiveObjectInspector knnItemsValueKeyOI,
            @Nonnull final PrimitiveObjectInspector knnItemsValueValueOI,
            @Nullable Int2ObjectMap<Int2FloatMap> knnItems, @Nonnull final MutableInt nnzKNNi) {
        if (knnItems == null) {
            knnItems = new Int2ObjectOpenHashMap<>(1024);
        } else {
            knnItems.clear();
        }

        int numElementOfKNNItems = 0;
        for (Map.Entry<?, ?> entry : knnItemsOI.getMap(kNNiObj).entrySet()) {
            int user = PrimitiveObjectInspectorUtils.getInt(entry.getKey(), knnItemsKeyOI);
            Int2FloatMap ru = int2floatMap(knnItemsValueOI.getMap(entry.getValue()),
                knnItemsValueKeyOI, knnItemsValueValueOI);
            knnItems.put(user, ru);
            numElementOfKNNItems += ru.size();
        }

        nnzKNNi.setValue(numElementOfKNNItems);
        return knnItems;
    }

    @Nonnull
    private static Int2FloatMap int2floatMap(@Nonnull final Map<?, ?> map,
            @Nonnull final PrimitiveObjectInspector keyOI,
            @Nonnull final PrimitiveObjectInspector valueOI) {
        final Int2FloatMap result = new Int2FloatOpenHashMap(map.size());
        result.defaultReturnValue(0.f);

        for (Map.Entry<?, ?> entry : map.entrySet()) {
            float v = PrimitiveObjectInspectorUtils.getFloat(entry.getValue(), valueOI);
            if (v == 0.f) {
                continue;
            }
            int k = PrimitiveObjectInspectorUtils.getInt(entry.getKey(), keyOI);
            result.put(k, v);
        }

        return result;
    }

    @Nonnull
    private static Int2FloatMap int2floatMap(final int item, @Nonnull final Map<?, ?> map,
            @Nonnull final PrimitiveObjectInspector keyOI,
            @Nonnull final PrimitiveObjectInspector valueOI,
            @Nullable final FloatMatrix dataMatrix) {
        return int2floatMap(item, map, keyOI, valueOI, dataMatrix, null);
    }

    @Nonnull
    private static Int2FloatMap int2floatMap(final int item, @Nonnull final Map<?, ?> map,
            @Nonnull final PrimitiveObjectInspector keyOI,
            @Nonnull final PrimitiveObjectInspector valueOI, @Nullable final FloatMatrix dataMatrix,
            @Nullable Int2FloatMap dst) {
        if (dst == null) {
            dst = new Int2FloatOpenHashMap(map.size());
            dst.defaultReturnValue(0.f);
        } else {
            dst.clear();
        }

        for (Map.Entry<?, ?> entry : map.entrySet()) {
            float rating = PrimitiveObjectInspectorUtils.getFloat(entry.getValue(), valueOI);
            if (rating == 0.f) {
                continue;
            }
            int user = PrimitiveObjectInspectorUtils.getInt(entry.getKey(), keyOI);
            dst.put(user, rating);
            if (dataMatrix != null) {
                dataMatrix.set(item, user, rating);
            }
        }

        return dst;
    }
}
