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
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.NioStatefullSegment;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;
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

    private final DoKMatrix W = new DoKMatrix();
    private final DoKMatrix A = new DoKMatrix();

    private PrimitiveObjectInspector itemIOI;
    private PrimitiveObjectInspector itemJOI;
    private MapObjectInspector itemIRatesOI;
    private MapObjectInspector itemJRatesOI;

    private MapObjectInspector topKRatesOfIOI;
    private PrimitiveObjectInspector topKRatesOfIKeyOI;
    private MapObjectInspector topKRatesOfIValueOI;
    private PrimitiveObjectInspector topKRatesOfIValueKeyOI;
    private PrimitiveObjectInspector topKRatesOfIValueValueOI;

    private PrimitiveObjectInspector itemIRateKeyOI;
    private PrimitiveObjectInspector itemIRateValueOI;

    private PrimitiveObjectInspector itemJRateKeyOI;
    private PrimitiveObjectInspector itemJRateValueOI;

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

        this.itemIRatesOI = HiveUtils.asMapOI(argOIs[1]);
        this.itemIRateKeyOI = HiveUtils.asIntCompatibleOI((this.itemIRatesOI.getMapKeyObjectInspector()));
        this.itemIRateValueOI = HiveUtils.asDoubleCompatibleOI((this.itemIRatesOI.getMapValueObjectInspector()));

        this.topKRatesOfIOI = HiveUtils.asMapOI(argOIs[2]);
        this.topKRatesOfIKeyOI = HiveUtils.asIntCompatibleOI(topKRatesOfIOI.getMapKeyObjectInspector());
        this.topKRatesOfIValueOI = HiveUtils.asMapOI(topKRatesOfIOI.getMapValueObjectInspector());
        this.topKRatesOfIValueKeyOI = HiveUtils.asIntCompatibleOI(topKRatesOfIValueOI.getMapKeyObjectInspector());
        this.topKRatesOfIValueValueOI = HiveUtils.asDoubleCompatibleOI(topKRatesOfIValueOI.getMapValueObjectInspector());

        this.itemJOI = HiveUtils.asIntCompatibleOI(argOIs[3]);

        this.itemJRatesOI = HiveUtils.asMapOI(argOIs[4]);
        this.itemJRateKeyOI = HiveUtils.asIntCompatibleOI((this.itemJRatesOI.getMapKeyObjectInspector()));
        this.itemJRateValueOI = HiveUtils.asDoubleCompatibleOI((this.itemJRatesOI.getMapValueObjectInspector()));

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
            } catch (Throwable e) {
                throw new UDFArgumentException(e);
            }
            this.fileIO = new NioStatefullSegment(file,false);
            this.inputBuf = ByteBuffer.allocateDirect(1024*1024); // 1MB
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
        double cv_rate = 0.005;

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
                throw new UDFArgumentException("Argument `int numIterations` must be greater than 0: "
                        + numIterations);
            }

            conversionCheck = !cl.hasOption("disable_cvtest");

            cv_rate = Primitives.parseDouble(cl.getOptionValue("cv_rate"), cv_rate);
            if (cv_rate <= 0) {
                throw new UDFArgumentException("Argument `double cv_rate` must be greater than 0.0: "
                        + cv_rate);
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
        int i = PrimitiveObjectInspectorUtils.getInt(args[0], itemIOI);
        Map Ri = this.itemIRatesOI.getMap(args[1]);
        Map topKRatesOfI = this.topKRatesOfIOI.getMap(args[2]);
        int j = PrimitiveObjectInspectorUtils.getInt(args[3], itemJOI);
        Map Rj = this.itemJRatesOI.getMap(args[4]);
        trainAndStore(i, Ri, topKRatesOfI, j, Rj);

        if (this.numIterations == 1) {
            return;
        }

        if (this.previousItemId != i){
            this.previousItemId = i;

            for (Map.Entry<?, ?> userRate : ((Map<?, ?>) Ri).entrySet()) {
                Object u = userRate.getKey();
                double rui = PrimitiveObjectInspectorUtils.getDouble(userRate.getValue(), this.itemIRateValueOI);
                this.A.unsafeSet((int) u, i, rui); // need optimize
            }

            // save KNNi
            // count element size size: i, numKNN, [[u, numKNNu, [[item, rate], ...], ...]
            ByteBuffer buf = inputBuf;
            NioStatefullSegment dst = fileIO;

            int numElementOfKNNi = 0;
            Map<?, ?> knn = this.topKRatesOfIOI.getMap(topKRatesOfI);
            for (Map.Entry<?, ?> ri : knn.entrySet()) {
                numElementOfKNNi += this.topKRatesOfIValueOI.getMap(ri.getValue()).size();
            }

            int recordBytes = SizeOf.INT + SizeOf.INT + SizeOf.INT * 2 * knn.size() + (SizeOf.DOUBLE+SizeOf.INT) * numElementOfKNNi;
            int requiredBytes = SizeOf.INT + recordBytes; // need to allocate space for "recordBytes" itself

            int remain = buf.remaining();
            if (remain < requiredBytes) {
                writeBuffer(buf, dst);
            }

            buf.putInt(i);
            buf.putInt(knn.size());
            for (Map.Entry<?, ?> ri : this.topKRatesOfIOI.getMap(topKRatesOfI).entrySet()){
                int user = PrimitiveObjectInspectorUtils.getInt(ri.getKey(), this.topKRatesOfIKeyOI);
                Map<?, ?> userKNN = this.topKRatesOfIValueOI.getMap(ri.getValue());

                buf.putInt(user);
                buf.putInt(userKNN.size());

                for (Map.Entry<?, ?> ratings : userKNN.entrySet()) {
                    int item = PrimitiveObjectInspectorUtils.getInt(ratings.getKey(), this.topKRatesOfIValueKeyOI);
                    double rating = PrimitiveObjectInspectorUtils.getDouble(ratings.getValue(), this.topKRatesOfIValueValueOI);

                    buf.putInt(item);
                    buf.putDouble(rating);
                }
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
                if (this.W.unsafeGet(i, j, 0.) != 0.) {
                    Object[] res = new Object[3];
                    res[0] = new IntWritable(i);
                    res[1] = new IntWritable(j);
                    res[2] = new DoubleWritable(this.W.get(i, j));
                    forward(res);
                }
            }
        }
    }

    protected double predict(Object u, int i, Map<?, ?> topKRatesOfI, int excludeIndex) {
        if (!topKRatesOfI.containsKey(u)) {
            return 0.;
        }
        double pred = 0.d;
        for (Map.Entry<?, ?> rating : this.topKRatesOfIValueOI.getMap(topKRatesOfI.get(u))
                                                              .entrySet()) {
            int k = PrimitiveObjectInspectorUtils.getInt(rating.getKey(),
                this.topKRatesOfIValueKeyOI);
            if (k == excludeIndex) {
                continue;
            }
            double rate = PrimitiveObjectInspectorUtils.getDouble(rating.getValue(),
                this.topKRatesOfIValueValueOI);
            pred += rate * this.W.unsafeGet(i, k, 0.d);
        }
        return pred;
    }

    protected double predict(Object u, int i, Map<?, ?> topKRatesOfI) {
        if (!topKRatesOfI.containsKey(u)) {
            return 0.;
        }

        double pred = 0.d;
        for (Map.Entry<?, ?> rating : this.topKRatesOfIValueOI.getMap(topKRatesOfI.get(u))
                                                              .entrySet()) {
            int k = PrimitiveObjectInspectorUtils.getInt(rating.getKey(),
                this.topKRatesOfIValueKeyOI);
            double rate = PrimitiveObjectInspectorUtils.getDouble(rating.getValue(),
                this.topKRatesOfIValueValueOI);

            pred += rate * this.W.unsafeGet(i, k, 0.d);
        }
        return pred;
    }


    private void trainAndStore(int i, Map<?, ?> Ri, Map<?, ?> topKRatesOfI, int j, Map<?, ?> Rj) {
        int N = Rj.size();
        double gradSum = 0.d;
        double rateSum = 0.d;

        for (Map.Entry<?, ?> userRate : Rj.entrySet()) {
            Object u = userRate.getKey();
            double ruj = PrimitiveObjectInspectorUtils.getDouble(userRate.getValue(),
                    this.itemJRateValueOI);
            double rui = 0.d;
            if (Ri.containsKey(u)) {
                rui = PrimitiveObjectInspectorUtils.getDouble(Ri.get(u), this.itemIRateValueOI);
            }

            double eui = rui - predict(u, i, topKRatesOfI, j);
            gradSum += ruj * eui;
            rateSum += ruj * ruj;

            if (this.numIterations > 1){
                this.A.unsafeSet((int) u, j, ruj); // need optimize
            }
        }

        gradSum /= N;
        rateSum /= N;

        this.W.unsafeSet(i, j, getUpdateTerm(gradSum, rateSum));
    }


//    private void train(int i, Map<?, ?> Ri, Map<?, ?> topKRatesOfI, int j, Map<?, ?> Rj) {
//        int N = Rj.size();
//        double gradSum = 0.d;
//        double rateSum = 0.d;
//        double errs = 0.d;
//        for (Map.Entry<?, ?> userRate : Rj.entrySet()) {
//            Object u = userRate.getKey();
//            double ruj = PrimitiveObjectInspectorUtils.getDouble(userRate.getValue(),
//                this.itemJRateValueOI);
//            double rui = 0.d;
//            if (Ri.containsKey(u)) {
//                rui = PrimitiveObjectInspectorUtils.getDouble(Ri.get(u), this.itemIRateValueOI);
//            }
//
//            double eui = rui - predict(u, i, topKRatesOfI, j);
//            gradSum += ruj * eui;
//            rateSum += ruj * ruj;
//            errs += eui * eui;
//        }
//
//        gradSum /= N;
//        rateSum /= N;
//        errs /= N;
//
//        this.loss += errs;
//        this.W.unsafeSet(i, j, getUpdateTerm(gradSum, rateSum));
//    }

    private double getUpdateTerm(double gradSum, double rateSum){
        double update = 0.d;
        if (this.l1 < Math.abs(gradSum)) {
            if (gradSum > 0.) {
                update = (gradSum - this.l1) / (rateSum + this.l2);
            } else {
                update = (gradSum + this.l1) / (rateSum + this.l2);
            }
            if (update < 0.) { // non-negativity constraints
                update = 0.;
            }
        }
        return update;
    }

    private final void runIterativeTraining() throws HiveException {
        for (int iter = 1; iter < this.numIterations; iter++){

        }
    }
}
