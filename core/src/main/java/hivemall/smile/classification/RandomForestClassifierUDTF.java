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
package hivemall.smile.classification;

import hivemall.UDTFWithOptions;
import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.MatrixUtils;
import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.math.matrix.builders.RowMajorDenseMatrixBuilder;
import hivemall.math.matrix.ints.ColumnMajorIntMatrix;
import hivemall.math.matrix.ints.DoKIntMatrix;
import hivemall.math.matrix.ints.IntMatrix;
import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.math.vector.Vector;
import hivemall.math.vector.VectorProcedure;
import hivemall.smile.classification.DecisionTree.SplitRule;
import hivemall.smile.data.Attribute;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.smile.utils.SmileTaskExecutor;
import hivemall.utils.codec.Base91;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Preconditions;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.RandomUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.exec.MapredContextAccessor;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
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

@Description(
        name = "train_randomforest_classifier",
        value = "_FUNC_(array<double|string> features, int label [, const array<double> classWeights, const string options]) - "
                + "Returns a relation consists of "
                + "<int model_id, int model_type, string pred_model, array<double> var_importance, int oob_errors, int oob_tests, double weight>")
public final class RandomForestClassifierUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(RandomForestClassifierUDTF.class);

    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    private PrimitiveObjectInspector labelOI;

    private boolean denseInput;
    private MatrixBuilder matrixBuilder;
    private IntArrayList labels;

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
    private Attribute[] _attributes;
    private SplitRule _splitRule;
    private boolean _stratifiedSampling;
    private double _subsample;

    @Nullable
    private double[] _classWeight;

    @Nullable
    private Reporter _progressReporter;
    @Nullable
    private Counter _treeBuildTaskCounter;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("trees", "num_trees", true,
            "The number of trees for each task [default: 50]");
        opts.addOption("vars", "num_variables", true,
            "The number of random selected features [default: ceil(sqrt(x[0].length))]."
                    + " int(num_variables * x[0].length) is considered if num_variable is (0,1]");
        opts.addOption("depth", "max_depth", true,
            "The maximum number of the tree depth [default: Integer.MAX_VALUE]");
        opts.addOption("leafs", "max_leaf_nodes", true,
            "The maximum number of leaf nodes [default: Integer.MAX_VALUE]");
        opts.addOption("splits", "min_split", true,
            "A node that has greater than or equals to `min_split` examples will split [default: 2]");
        opts.addOption("min_samples_leaf", true,
            "The minimum number of samples in a leaf node [default: 1]");
        opts.addOption("seed", true, "seed value in long [default: -1 (random)]");
        opts.addOption("attrs", "attribute_types", true, "Comma separated attribute types "
                + "(Q for quantitative variable and C for categorical variable. e.g., [Q,C,Q,C])");
        opts.addOption("rule", "split_rule", true, "Split algorithm [default: GINI, ENTROPY]");
        opts.addOption("stratified", "stratified_sampling", false,
            "Enable Stratified sampling for unbalanced data");
        opts.addOption("subsample", true, "Sampling rate in range (0.0,1.0]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        int trees = 50, maxDepth = Integer.MAX_VALUE;
        int numLeafs = Integer.MAX_VALUE, minSplits = 2, minSamplesLeaf = 1;
        float numVars = -1.f;
        Attribute[] attrs = null;
        long seed = -1L;
        SplitRule splitRule = SplitRule.GINI;
        double[] classWeight = null;
        boolean stratifiedSampling = false;
        double subsample = 1.0d;

        CommandLine cl = null;
        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = parseOptions(rawArgs);

            trees = Primitives.parseInt(cl.getOptionValue("num_trees"), trees);
            if (trees < 1) {
                throw new IllegalArgumentException("Invlaid number of trees: " + trees);
            }
            numVars = Primitives.parseFloat(cl.getOptionValue("num_variables"), numVars);
            maxDepth = Primitives.parseInt(cl.getOptionValue("max_depth"), maxDepth);
            numLeafs = Primitives.parseInt(cl.getOptionValue("max_leaf_nodes"), numLeafs);
            minSplits = Primitives.parseInt(cl.getOptionValue("min_split"), minSplits);
            minSamplesLeaf = Primitives.parseInt(cl.getOptionValue("min_samples_leaf"),
                minSamplesLeaf);
            seed = Primitives.parseLong(cl.getOptionValue("seed"), seed);
            attrs = SmileExtUtils.resolveAttributes(cl.getOptionValue("attribute_types"));
            splitRule = SmileExtUtils.resolveSplitRule(cl.getOptionValue("split_rule", "GINI"));
            stratifiedSampling = cl.hasOption("stratified_sampling");
            subsample = Primitives.parseDouble(cl.getOptionValue("subsample"), 1.0d);
            Preconditions.checkArgument(subsample > 0.d && subsample <= 1.0d,
                UDFArgumentException.class, "Invalid -subsample value: " + subsample);

            if (argOIs.length >= 4) {
                classWeight = HiveUtils.getConstDoubleArray(argOIs[3]);
                if (classWeight != null) {
                    for (int i = 0; i < classWeight.length; i++) {
                        double v = classWeight[i];
                        if (Double.isNaN(v)) {
                            classWeight[i] = 1.0d;
                        } else if (v <= 0.d) {
                            throw new UDFArgumentTypeException(3,
                                "each classWeight must be greather than 0: "
                                        + Arrays.toString(classWeight));
                        }
                    }
                }
            }
        }

        this._numTrees = trees;
        this._numVars = numVars;
        this._maxDepth = maxDepth;
        this._maxLeafNodes = numLeafs;
        this._minSamplesSplit = minSplits;
        this._minSamplesLeaf = minSamplesLeaf;
        this._seed = seed;
        this._attributes = attrs;
        this._splitRule = splitRule;
        this._stratifiedSampling = stratifiedSampling;
        this._subsample = subsample;
        this._classWeight = classWeight;

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 2 || argOIs.length > 4) {
            throw new UDFArgumentException(
                "_FUNC_ takes 2 ~ 4 arguments: array<double|string> features, int label [, const string options, const array<double> classWeight]: "
                        + argOIs.length);
        }

        ListObjectInspector listOI = HiveUtils.asListOI(argOIs[0]);
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
                "_FUNC_ takes double[] or string[] for the first argument: " + listOI.getTypeName());
        }
        this.labelOI = HiveUtils.asIntCompatibleOI(argOIs[1]);

        processOptions(argOIs);

        this.labels = new IntArrayList(1024);

        final ArrayList<String> fieldNames = new ArrayList<String>(6);
        final ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>(6);

        fieldNames.add("model_id");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("model_weight");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        fieldNames.add("model");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("var_importance");
        if (denseInput) {
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
        } else {
            fieldOIs.add(ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.writableIntObjectInspector,
                PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
        }
        fieldNames.add("oob_errors");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
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
        int label = PrimitiveObjectInspectorUtils.getInt(args[1], labelOI);
        labels.add(label);
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
                : _progressReporter.getCounter("hivemall.smile.RandomForestClassifier$Counter",
                    "finishedTreeBuildTasks");
        reportProgress(_progressReporter);

        if (!labels.isEmpty()) {
            Matrix x = matrixBuilder.buildMatrix();
            this.matrixBuilder = null;
            int[] y = labels.toArray();
            this.labels = null;

            // run training
            train(x, y);
        }

        // clean up
        this.featureListOI = null;
        this.featureElemOI = null;
        this.labelOI = null;
        this._attributes = null;
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
     * @param numVars The number of variables to pick up in each node.
     * @param seed The seed number for Random Forest
     */
    private void train(@Nonnull Matrix x, @Nonnull final int[] y) throws HiveException {
        final int numExamples = x.numRows();
        if (numExamples != y.length) {
            throw new HiveException(String.format("The sizes of X and Y don't match: %d != %d",
                numExamples, y.length));
        }
        checkOptions();

        // Shuffle training samples
        x = SmileExtUtils.shuffle(x, y, _seed);

        int[] labels = SmileExtUtils.classLables(y);
        Attribute[] attributes = SmileExtUtils.attributeTypes(_attributes, x);
        int numInputVars = SmileExtUtils.computeNumInputVars(_numVars, x);

        if (logger.isInfoEnabled()) {
            logger.info("numTrees: " + _numTrees + ", numVars: " + numInputVars + ", maxDepth: "
                    + _maxDepth + ", minSamplesSplit: " + _minSamplesSplit + ", maxLeafs: "
                    + _maxLeafNodes + ", splitRule: " + _splitRule + ", seed: " + _seed);
        }

        IntMatrix prediction = new DoKIntMatrix(numExamples, labels.length); // placeholder for out-of-bag prediction
        ColumnMajorIntMatrix order = SmileExtUtils.sort(attributes, x);
        AtomicInteger remainingTasks = new AtomicInteger(_numTrees);
        List<TrainingTask> tasks = new ArrayList<TrainingTask>();
        for (int i = 0; i < _numTrees; i++) {
            long s = (_seed == -1L) ? -1L : _seed + i;
            tasks.add(new TrainingTask(this, i, attributes, x, y, numInputVars, order, prediction,
                s, remainingTasks));
        }

        MapredContext mapredContext = MapredContextAccessor.get();
        final SmileTaskExecutor executor = new SmileTaskExecutor(mapredContext);
        try {
            executor.run(tasks);
        } catch (Exception ex) {
            throw new HiveException(ex);
        } finally {
            executor.shotdown();
        }
    }

    /**
     * Synchronized because {@link #forward(Object)} should be called from a single thread.
     * 
     * @param accuracy
     */
    synchronized void forward(final int taskId, @Nonnull final Text model,
            @Nonnull final Vector importance, @Nonnegative final double accuracy, final int[] y,
            @Nonnull final IntMatrix prediction, final boolean lastTask) throws HiveException {
        int oobErrors = 0;
        int oobTests = 0;
        if (lastTask) {
            // out-of-bag error estimate
            for (int i = 0; i < y.length; i++) {
                final int pred = MatrixUtils.whichMax(prediction, i);
                if (pred != -1 && prediction.get(i, pred) > 0) {
                    oobTests++;
                    if (pred != y[i]) {
                        oobErrors++;
                    }
                }
            }
        }

        final Object[] forwardObjs = new Object[6];
        String modelId = RandomUtils.getUUID();
        forwardObjs[0] = new Text(modelId);
        forwardObjs[1] = new DoubleWritable(accuracy);
        forwardObjs[2] = model;
        if (denseInput) {
            forwardObjs[3] = WritableUtils.toWritableList(importance.toArray());
        } else {
            final Map<IntWritable, DoubleWritable> map = new HashMap<IntWritable, DoubleWritable>(
                importance.size());
            importance.each(new VectorProcedure() {
                public void apply(int i, double value) {
                    map.put(new IntWritable(i), new DoubleWritable(value));
                }
            });
            forwardObjs[3] = map;
        }
        forwardObjs[4] = new IntWritable(oobErrors);
        forwardObjs[5] = new IntWritable(oobTests);
        forward(forwardObjs);

        reportProgress(_progressReporter);
        incrCounter(_treeBuildTaskCounter, 1);

        logger.info("Forwarded " + taskId + "-th DecisionTree out of " + _numTrees);
    }

    /**
     * Trains a regression tree.
     */
    private static final class TrainingTask implements Callable<Integer> {
        /**
         * Attribute properties.
         */
        @Nonnull
        private final Attribute[] _attributes;
        /**
         * Training instances.
         */
        @Nonnull
        private final Matrix _x;
        /**
         * Training sample labels.
         */
        @Nonnull
        private final int[] _y;
        /**
         * The index of training values in ascending order. Note that only numeric attributes will be sorted.
         */
        @Nonnull
        private final ColumnMajorIntMatrix _order;
        /**
         * The number of variables to pick up in each node.
         */
        private final int _numVars;
        /**
         * The out-of-bag predictions.
         */
        @Nonnull
        @GuardedBy("_udtf")
        private final IntMatrix _prediction;

        @Nonnull
        private final RandomForestClassifierUDTF _udtf;
        private final int _taskId;
        private final long _seed;
        @Nonnull
        private final AtomicInteger _remainingTasks;

        TrainingTask(@Nonnull RandomForestClassifierUDTF udtf, int taskId,
                @Nonnull Attribute[] attributes, @Nonnull Matrix x, @Nonnull int[] y, int numVars,
                @Nonnull ColumnMajorIntMatrix order, @Nonnull IntMatrix prediction, long seed,
                @Nonnull AtomicInteger remainingTasks) {
            this._udtf = udtf;
            this._taskId = taskId;
            this._attributes = attributes;
            this._x = x;
            this._y = y;
            this._order = order;
            this._numVars = numVars;
            this._prediction = prediction;
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
            final BitSet sampled = new BitSet(N);
            final int[] bags = sampling(sampled, N, rnd1);

            DecisionTree tree = new DecisionTree(_attributes, _x, _y, _numVars, _udtf._maxDepth,
                _udtf._maxLeafNodes, _udtf._minSamplesSplit, _udtf._minSamplesLeaf, bags, _order,
                _udtf._splitRule, rnd2);

            // out-of-bag prediction
            int oob = 0;
            int correct = 0;
            final Vector xProbe = _x.rowVector();
            for (int i = sampled.nextClearBit(0); i < N; i = sampled.nextClearBit(i + 1)) {
                oob++;
                _x.getRow(i, xProbe);
                final int p = tree.predict(xProbe);
                if (p == _y[i]) {
                    correct++;
                }
                synchronized (_udtf) {
                    _prediction.incr(i, p);
                }
            }

            Text model = getModel(tree);
            Vector importance = tree.importance();
            double accuracy = (oob == 0) ? 1.0d : (double) correct / oob;
            int remain = _remainingTasks.decrementAndGet();
            boolean lastTask = (remain == 0);
            _udtf.forward(_taskId + 1, model, importance, accuracy, _y, _prediction, lastTask);

            return Integer.valueOf(remain);
        }

        @Nonnull
        private int[] sampling(@Nonnull final BitSet sampled, final int N, @Nonnull PRNG rnd) {
            return _udtf._stratifiedSampling ? stratifiedSampling(sampled, N, _udtf._subsample, rnd)
                    : uniformSampling(sampled, N, _udtf._subsample, rnd);
        }

        @Nonnull
        private static int[] uniformSampling(@Nonnull final BitSet sampled, final int N,
                final double subsample, final PRNG rnd) {
            final int size = (int) Math.round(N * subsample);
            final int[] bags = new int[N];
            for (int i = 0; i < size; i++) {
                int index = rnd.nextInt(N);
                bags[i] = index;
                sampled.set(index);
            }
            return bags;
        }

        /**
         * Stratified sampling for unbalanced data.
         * 
         * @link https://en.wikipedia.org/wiki/Stratified_sampling
         */
        @Nonnull
        private int[] stratifiedSampling(@Nonnull final BitSet sampled, final int N,
                final double subsample, final PRNG rnd) {
            final IntArrayList bagsList = new IntArrayList(N);
            final int k = smile.math.Math.max(_y) + 1;
            final IntArrayList cj = new IntArrayList(N / k);
            for (int l = 0; l < k; l++) {
                int nj = 0;
                for (int i = 0; i < N; i++) {
                    if (_y[i] == l) {
                        cj.add(i);
                        nj++;
                    }
                }
                if (subsample != 1.0d) {
                    nj = (int) Math.round(nj * subsample);
                }
                final int size = (_udtf._classWeight == null) ? nj : (int) Math.round(nj
                        * _udtf._classWeight[l]);
                for (int j = 0; j < size; j++) {
                    int xi = rnd.nextInt(nj);
                    int index = cj.get(xi);
                    bagsList.add(index);
                    sampled.set(index);
                }
                cj.clear();
            }
            int[] bags = bagsList.toArray(true);
            SmileExtUtils.shuffle(bags, rnd);
            return bags;
        }

        @Nonnull
        private static Text getModel(@Nonnull final DecisionTree tree) throws HiveException {
            byte[] b = tree.predictSerCodegen(true);
            b = Base91.encode(b);
            return new Text(b);
        }

    }

}
