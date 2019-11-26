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
package hivemall.smile.regression;

import hivemall.UDTFWithOptions;
import matrix4j.matrix.Matrix;
import matrix4j.matrix.builders.CSRMatrixBuilder;
import matrix4j.matrix.builders.MatrixBuilder;
import matrix4j.matrix.builders.RowMajorDenseMatrixBuilder;
import matrix4j.vector.Vector;
import matrix4j.vector.VectorProcedure;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.smile.utils.SmileTaskExecutor;
import hivemall.utils.codec.Base91;
import hivemall.utils.collections.lists.DoubleArrayList;
import hivemall.utils.datetime.StopWatch;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.SerdeUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.RandomUtils;
import hivemall.utils.random.PRNG;
import hivemall.utils.random.RandomNumberGeneratorFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.exec.MapredContextAccessor;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters.Counter;
import org.apache.hadoop.mapred.Reporter;
import org.roaringbitmap.RoaringBitmap;

@Description(name = "train_randomforest_regressor",
        value = "_FUNC_(array<double|string> features, double target [, string options]) - "
                + "Returns a relation consists of "
                + "<int model_id, int model_type, string model, array<double> var_importance, double oob_errors, int oob_tests>")
public final class RandomForestRegressionUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(RandomForestRegressionUDTF.class);

    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    private PrimitiveObjectInspector targetOI;

    private boolean denseInput;
    private MatrixBuilder matrixBuilder;
    private DoubleArrayList targets;
    /**
     * The number of trees for each task
     */
    private int _numTrees;
    /**
     * The number of random selected features
     */
    private float _numVars;
    /**
     * The maximum number of the tree depth
     */
    private int _maxDepth;
    /**
     * The maximum number of leaf nodes
     */
    private int _maxLeafNodes;
    private int _minSamplesSplit;
    private int _minSamplesLeaf;
    private long _seed;
    private byte[] _nominalAttrs;

    @Nullable
    private transient Reporter _progressReporter;
    @Nullable
    private transient Counter _treeBuildTaskCounter;
    @Nullable
    private transient Counter _treeConstructionTimeCounter;
    @Nullable
    private transient Counter _treeSerializationTimeCounter;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("trees", "num_trees", true,
            "The number of trees for each task [default: 50]");
        opts.addOption("vars", "num_variables", true,
            "The number of random selected features [default: ceil(sqrt(x[0].length))]."
                    + " int(num_variables * x[0].length) is considered if num_variable is (0.0,1.0]");
        opts.addOption("depth", "max_depth", true,
            "The maximum number of the tree depth [default: Integer.MAX_VALUE]");
        opts.addOption("leafs", "max_leaf_nodes", true,
            "The maximum number of leaf nodes [default: Integer.MAX_VALUE]");
        opts.addOption("min_samples_split", true,
            "A node that has greater than or equals to `min_split` examples will split [default: 5]");
        // synonym of min_samples_split
        opts.addOption("split", "min_split", true,
            "A node that has greater than or equals to `min_split` examples will split [default: 5]");
        opts.addOption("min_samples_leaf", true,
            "The minimum number of samples in a leaf node [default: 1]");
        opts.addOption("seed", true, "seed value in long [default: -1 (random)]");
        opts.addOption("attrs", "attribute_types", true, "Comma separated attribute types "
                + "(Q for quantitative variable and C for categorical variable. e.g., [Q,C,Q,C])");
        opts.addOption("nominal_attr_indicies", "categorical_attr_indicies", true,
            "Comma seperated indicies of categorical attributes, e.g., [3,5,6]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        int trees = 50, maxDepth = Integer.MAX_VALUE;
        int maxLeafNodes = Integer.MAX_VALUE, minSamplesSplit = 5, minSamplesLeaf = 1;
        float numVars = -1.f;
        RoaringBitmap attrs = new RoaringBitmap();
        long seed = -1L;

        CommandLine cl = null;
        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs, 2);
            cl = parseOptions(rawArgs);

            trees = Primitives.parseInt(cl.getOptionValue("num_trees"), trees);
            if (trees < 1) {
                throw new IllegalArgumentException("Invalid number of trees: " + trees);
            }
            numVars = Primitives.parseFloat(cl.getOptionValue("num_variables"), numVars);
            maxDepth = Primitives.parseInt(cl.getOptionValue("max_depth"), maxDepth);
            maxLeafNodes = Primitives.parseInt(cl.getOptionValue("max_leaf_nodes"), maxLeafNodes);
            String min_samples_split = cl.getOptionValue("min_samples_split");
            if (min_samples_split == null) {
                minSamplesSplit =
                        Primitives.parseInt(cl.getOptionValue("min_split"), minSamplesSplit);
            } else {
                minSamplesSplit = Integer.parseInt(min_samples_split);
            }
            minSamplesLeaf =
                    Primitives.parseInt(cl.getOptionValue("min_samples_leaf"), minSamplesLeaf);
            seed = Primitives.parseLong(cl.getOptionValue("seed"), seed);
            String nominal_attr_indicies = cl.getOptionValue("nominal_attr_indicies");
            if (nominal_attr_indicies != null) {
                attrs = SmileExtUtils.parseNominalAttributeIndicies(nominal_attr_indicies);
            } else {
                attrs = SmileExtUtils.resolveAttributes(cl.getOptionValue("attribute_types"));
            }
        }

        this._numTrees = trees;
        this._numVars = numVars;
        this._maxDepth = maxDepth;
        this._maxLeafNodes = maxLeafNodes;
        this._minSamplesSplit = minSamplesSplit;
        this._minSamplesLeaf = minSamplesLeaf;
        this._seed = seed;
        this._nominalAttrs = SerdeUtils.serializeRoaring(attrs);

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takes 2 or 3 arguments: array<double|string> features, double target [, const string options]: "
                    + argOIs.length);
        }

        ListObjectInspector listOI = HiveUtils.asListOI(argOIs, 0);
        ObjectInspector elemOI = listOI.getListElementObjectInspector();
        this.featureListOI = listOI;
        if (HiveUtils.isNumberOI(elemOI)) {
            this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);
            this.denseInput = true;
            this.matrixBuilder = new RowMajorDenseMatrixBuilder(8192);
        } else if (HiveUtils.isStringOI(elemOI)) {
            this.featureElemOI = HiveUtils.asStringOI(elemOI);
            this.denseInput = false;
            this.matrixBuilder = new CSRMatrixBuilder(8192);
        } else {
            throw new UDFArgumentException(
                "_FUNC_ takes double[] or string[] for the first argument: "
                        + listOI.getTypeName());
        }
        this.targetOI = HiveUtils.asDoubleCompatibleOI(argOIs, 1);

        processOptions(argOIs);

        this.targets = new DoubleArrayList(1024);

        final ArrayList<String> fieldNames = new ArrayList<String>(6);
        final ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>(6);

        fieldNames.add("model_id");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("model_err");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        fieldNames.add("model");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("var_importance");
        if (denseInput) {
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
        } else {
            fieldOIs.add(ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
        }
        fieldNames.add("oob_errors");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        fieldNames.add("oob_tests");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (args[0] == null) {
            throw new HiveException("array<double> features was null");
        }
        parseFeatures(args[0], matrixBuilder);
        double target = PrimitiveObjectInspectorUtils.getDouble(args[1], targetOI);
        targets.add(target);
    }

    private void parseFeatures(@Nonnull final Object argObj, @Nonnull final MatrixBuilder builder) {
        if (denseInput) {
            final int length = featureListOI.getListLength(argObj);
            for (int i = 0; i < length; i++) {
                Object o = featureListOI.getListElement(argObj, i);
                if (o == null) {
                    continue;
                }
                double v = PrimitiveObjectInspectorUtils.getDouble(o, featureElemOI);
                builder.nextColumn(i, v);
            }
        } else {
            final int length = featureListOI.getListLength(argObj);
            for (int i = 0; i < length; i++) {
                Object o = featureListOI.getListElement(argObj, i);
                if (o == null) {
                    continue;
                }
                String fv = o.toString();
                builder.nextColumn(fv);
            }
        }
        builder.nextRow();
    }

    @Override
    public void close() throws HiveException {
        this._progressReporter = getReporter();
        this._treeBuildTaskCounter = (_progressReporter == null) ? null
                : _progressReporter.getCounter("hivemall.smile.RandomForestRegression$Counter",
                    "Number of finished tree construction tasks");
        this._treeConstructionTimeCounter = (_progressReporter == null) ? null
                : _progressReporter.getCounter("hivemall.smile.RandomForestRegression$Counter",
                    "Elapsed time in seconds for tree construction");
        this._treeSerializationTimeCounter = (_progressReporter == null) ? null
                : _progressReporter.getCounter("hivemall.smile.RandomForestRegression$Counter",
                    "Elapsed time in seconds for tree serialization");

        reportProgress(_progressReporter);

        if (!targets.isEmpty()) {
            Matrix x = matrixBuilder.buildMatrix();
            this.matrixBuilder = null;
            double[] y = targets.toArray();
            this.targets = null;

            // run training
            train(x, y);
        }

        // clean up
        this.featureListOI = null;
        this.featureElemOI = null;
        this.targetOI = null;
        this._nominalAttrs = null;
    }

    private void checkOptions() throws HiveException {
        if (_minSamplesSplit <= 0) {
            throw new HiveException("Invalid minSamplesSplit: " + _minSamplesSplit);
        }
        if (_maxDepth < 1) {
            throw new HiveException("Invalid maxDepth: " + _maxDepth);
        }
    }

    /**
     * @param x features
     * @param y label
     * @param attrs attribute types
     * @param numTrees The number of trees
     * @param _numVars The number of variables to pick up in each node.
     * @param _seed The seed number for Random Forest
     */
    private void train(@Nonnull Matrix x, @Nonnull final double[] y) throws HiveException {
        final int numExamples = x.numRows();
        if (numExamples != y.length) {
            throw new HiveException(
                String.format("The sizes of X and Y don't match: %d != %d", numExamples, y.length));
        }
        checkOptions();

        // Shuffle training samples
        x = SmileExtUtils.shuffle(x, y, _seed);

        int numInputVars = SmileExtUtils.computeNumInputVars(_numVars, x);

        if (logger.isInfoEnabled()) {
            logger.info("numTrees: " + _numTrees + ", numVars: " + numInputVars
                    + ", minSamplesSplit: " + _minSamplesSplit + ", maxDepth: " + _maxDepth
                    + ", maxLeafs: " + _maxLeafNodes + ", nodeCapacity: " + _minSamplesSplit
                    + ", seed: " + _seed);
        }

        final RoaringBitmap nominalAttrs = SerdeUtils.deserializeRoaring(_nominalAttrs);
        this._nominalAttrs = null;
        final double[] prediction = new double[numExamples]; // placeholder for out-of-bag prediction
        final int[] oob = new int[numExamples];
        final AtomicInteger remainingTasks = new AtomicInteger(_numTrees);
        List<TrainingTask> tasks = new ArrayList<TrainingTask>();
        for (int i = 0; i < _numTrees; i++) {
            long s = (_seed == -1L) ? -1L : _seed + i;
            tasks.add(new TrainingTask(this, i, nominalAttrs, x, y, numInputVars, prediction, oob,
                s, remainingTasks));
        }

        MapredContext mapredContext = MapredContextAccessor.get();
        final SmileTaskExecutor executor = new SmileTaskExecutor(mapredContext);
        try {
            executor.run(tasks);
        } catch (Exception ex) {
            throw new HiveException(ex);
        } finally {
            executor.shutdown();
        }
    }

    /**
     * Synchronized because {@link #forward(Object)} should be called from a single thread.
     *
     * @param error
     */
    synchronized void forward(final int taskId, @Nonnull final Text model,
            @Nonnull final Vector importance, @Nonnegative final double error, final double[] y,
            final double[] prediction, final int[] oob, final boolean lastTask)
            throws HiveException {
        double oobErrors = 0.d;
        int oobTests = 0;
        if (lastTask) {
            // out-of-bag error estimate
            for (int i = 0; i < y.length; i++) {
                if (oob[i] > 0) {
                    oobTests++;
                    double pred = prediction[i] / oob[i];
                    oobErrors += smile.math.Math.sqr(pred - y[i]);
                }
            }
        }
        String modelId = RandomUtils.getUUID();
        final Object[] forwardObjs = new Object[6];
        forwardObjs[0] = new Text(modelId);
        forwardObjs[1] = new DoubleWritable(error);
        forwardObjs[2] = model;
        if (denseInput) {
            forwardObjs[3] = WritableUtils.toWritableList(importance.toArray());
        } else {
            final Map<IntWritable, DoubleWritable> map =
                    new HashMap<IntWritable, DoubleWritable>(importance.size());
            importance.each(new VectorProcedure() {
                public void apply(int i, double value) {
                    map.put(new IntWritable(i), new DoubleWritable(value));
                }
            });
            forwardObjs[3] = map;
        }
        forwardObjs[4] = new DoubleWritable(oobErrors);
        forwardObjs[5] = new IntWritable(oobTests);
        forward(forwardObjs);

        reportProgress(_progressReporter);
        incrCounter(_treeBuildTaskCounter, 1);

        logger.info("Forwarded " + taskId + "-th RegressionTree out of " + _numTrees);
    }

    /**
     * Trains a regression tree.
     */
    private static final class TrainingTask implements Callable<Integer> {
        /**
         * Attribute properties.
         */
        private final RoaringBitmap _nominalAttrs;
        /**
         * Training instances.
         */
        private final Matrix _x;
        /**
         * Training sample target values.
         */
        private final double[] _y;
        /**
         * The number of variables to pick up in each node.
         */
        private final int _numVars;
        /**
         * The out-of-bag predictions.
         */
        private final double[] _prediction;
        /**
         * Out-of-bag sample
         */
        private final int[] _oob;

        private final RandomForestRegressionUDTF _udtf;
        private final int _taskId;
        private final long _seed;
        private final AtomicInteger _remainingTasks;

        TrainingTask(RandomForestRegressionUDTF udtf, int taskId, RoaringBitmap nominalAttrs,
                Matrix x, double[] y, int numVars, double[] prediction, int[] oob, long seed,
                AtomicInteger remainingTasks) {
            this._udtf = udtf;
            this._taskId = taskId;
            this._nominalAttrs = nominalAttrs;
            this._x = x;
            this._y = y;
            this._numVars = numVars;
            this._prediction = prediction;
            this._oob = oob;
            this._seed = seed;
            this._remainingTasks = remainingTasks;
        }

        @Override
        public Integer call() throws HiveException {
            long s = (this._seed == -1L) ? SmileExtUtils.generateSeed()
                    : RandomNumberGeneratorFactory.createPRNG(_seed).nextLong();
            final PRNG rnd1 = RandomNumberGeneratorFactory.createPRNG(s);
            final PRNG rnd2 = RandomNumberGeneratorFactory.createPRNG(rnd1.nextLong());
            final int N = _x.numRows();

            // Training samples draw with replacement.
            final int[] samples = new int[N];
            for (int i = 0; i < N; i++) {
                int index = rnd1.nextInt(N);
                samples[index] += 1;
            }

            StopWatch stopwatch = new StopWatch();
            RegressionTree tree = new RegressionTree(_nominalAttrs, _x, _y, _numVars,
                _udtf._maxDepth, _udtf._maxLeafNodes, _udtf._minSamplesSplit, _udtf._minSamplesLeaf,
                samples, rnd2);
            incrCounter(_udtf._treeConstructionTimeCounter, stopwatch.elapsed(TimeUnit.SECONDS));

            // out-of-bag prediction
            int oob = 0;
            double error = 0.d;
            final Vector xProbe = _x.rowVector();
            for (int i = 0; i < samples.length; i++) {
                if (samples[i] != 0) {
                    continue;
                }
                oob++;
                _x.getRow(i, xProbe);
                final double pred = tree.predict(xProbe);
                synchronized (_udtf) {
                    _prediction[i] += pred;
                    _oob[i]++;
                }
                error += Math.abs(pred - _y[i]);
            }
            if (oob != 0) {
                error /= oob;
            }

            stopwatch.reset().start();
            Text model = getModel(tree);
            Vector importance = tree.importance();
            tree = null; // help GC
            int remain = _remainingTasks.decrementAndGet();
            boolean lastTask = (remain == 0);
            _udtf.forward(_taskId + 1, model, importance, error, _y, _prediction, _oob, lastTask);
            incrCounter(_udtf._treeSerializationTimeCounter, stopwatch.elapsed(TimeUnit.SECONDS));

            return Integer.valueOf(remain);
        }

        @Nonnull
        private static Text getModel(@Nonnull final RegressionTree tree) throws HiveException {
            byte[] b = tree.serialize(true);
            b = Base91.encode(b);
            return new Text(b);
        }

    }

}
