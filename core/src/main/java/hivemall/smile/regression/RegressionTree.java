/*
 * Copyright (c) 2010 Haifeng Li
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// This file includes a modified version of Smile:
// https://github.com/haifengl/smile/blob/master/core/src/main/java/smile/regression/RegressionTree.java
package hivemall.smile.regression;

import static hivemall.smile.classification.PredictionHandler.Operator.EQ;
import static hivemall.smile.classification.PredictionHandler.Operator.GT;
import static hivemall.smile.classification.PredictionHandler.Operator.LE;
import static hivemall.smile.classification.PredictionHandler.Operator.NE;
import static hivemall.smile.utils.SmileExtUtils.NOMINAL;
import static hivemall.smile.utils.SmileExtUtils.NUMERIC;
import static hivemall.smile.utils.SmileExtUtils.resolveFeatureName;

import hivemall.annotations.VisibleForTesting;
import matrix4j.matrix.Matrix;
import matrix4j.vector.DenseVector;
import matrix4j.vector.SparseVector;
import matrix4j.vector.Vector;
import matrix4j.vector.VectorProcedure;
import hivemall.smile.classification.PredictionHandler;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.smile.utils.VariableOrder;
import hivemall.utils.collections.arrays.SparseIntArray;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.function.Consumer;
import hivemall.utils.function.IntPredicate;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.ObjectUtils;
import hivemall.utils.lang.StringUtils;
import hivemall.utils.lang.mutable.MutableInt;
import hivemall.utils.random.PRNG;
import hivemall.utils.random.RandomNumberGeneratorFactory;
import hivemall.utils.sampling.IntReservoirSampler;
import it.unimi.dsi.fastutil.ints.Int2DoubleOpenHashMap;
import it.unimi.dsi.fastutil.ints.Int2IntMap.Entry;
import it.unimi.dsi.fastutil.ints.Int2IntOpenHashMap;
import smile.math.Math;
import smile.regression.GradientTreeBoost;
import smile.regression.RandomForest;
import smile.regression.Regression;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.roaringbitmap.IntConsumer;
import org.roaringbitmap.RoaringBitmap;

/**
 * Decision tree for regression. A decision tree can be learned by splitting the training set into
 * subsets based on an attribute value test. This process is repeated on each derived subset in a
 * recursive manner called recursive partitioning.
 * <p>
 * Classification and Regression Tree techniques have a number of advantages over many of those
 * alternative techniques.
 * <dl>
 * <dt>Simple to understand and interpret.</dt>
 * <dd>In most cases, the interpretation of results summarized in a tree is very simple. This
 * simplicity is useful not only for purposes of rapid classification of new observations, but can
 * also often yield a much simpler "model" for explaining why observations are classified or
 * predicted in a particular manner.</dd>
 * <dt>Able to handle both numerical and categorical data.</dt>
 * <dd>Other techniques are usually specialized in analyzing datasets that have only one type of
 * variable.</dd>
 * <dt>Tree methods are nonparametric and nonlinear.</dt>
 * <dd>The final results of using tree methods for classification or regression can be summarized in
 * a series of (usually few) logical if-then conditions (tree nodes). Therefore, there is no
 * implicit assumption that the underlying relationships between the predictor variables and the
 * dependent variable are linear, follow some specific non-linear link function, or that they are
 * even monotonic in nature. Thus, tree methods are particularly well suited for data mining tasks,
 * where there is often little a priori knowledge nor any coherent set of theories or predictions
 * regarding which variables are related and how. In those types of data analytics, tree methods can
 * often reveal simple relationships between just a few variables that could have easily gone
 * unnoticed using other analytic techniques.</dd>
 * </dl>
 * One major problem with classification and regression trees is their high variance. Often a small
 * change in the data can result in a very different series of splits, making interpretation
 * somewhat precarious. Besides, decision-tree learners can create over-complex trees that cause
 * over-fitting. Mechanisms such as pruning are necessary to avoid this problem. Another limitation
 * of trees is the lack of smoothness of the prediction surface.
 * <p>
 * Some techniques such as bagging, boosting, and random forest use more than one decision tree for
 * their analysis.
 *
 * @see GradientTreeBoost
 * @see RandomForest
 */
public final class RegressionTree implements Regression<Vector> {
    private static final Log logger = LogFactory.getLog(RegressionTree.class);

    /**
     * Training dataset.
     */
    private final Matrix _X;
    /**
     * Training data response value.
     */
    private final double[] _y;
    /**
     * The samples for training this node. Note that samples[i] is the number of sampling of
     * dataset[i]. 0 means that the datum is not included and values of greater than 1 are possible
     * because of sampling with replacement.
     */
    @Nonnull
    private final int[] _samples;
    /**
     * The index of training values in ascending order. Note that only numeric attributes will be
     * sorted.
     */
    @Nonnull
    private final VariableOrder _order;
    /**
     * An index that maps their current position in the {@link #_order} to their original locations
     * in {@link #_samples}.
     */
    @Nonnull
    private final int[] _sampleIndex;
    /**
     * The attributes of independent variable.
     */
    @Nonnull
    private final RoaringBitmap _nominalAttrs;
    /**
     * Variable importance. Every time a split of a node is made on variable the impurity criterion
     * for the two descendant nodes is less than the parent node. Adding up the decreases for each
     * individual variable over the tree gives a simple measure of variable importance.
     */
    private final Vector _importance;
    /**
     * The root of the regression tree
     */
    private final Node _root;
    /**
     * The maximum number of the tree depth
     */
    private final int _maxDepth;
    /**
     * The number of instances in a node below which the tree will not split, setting S = 5
     * generally gives good results.
     */
    private final int _minSamplesSplit;
    /**
     * The minimum number of samples in a leaf node
     */
    private final int _minSamplesLeaf;
    /**
     * The number of input variables to be used to determine the decision at a node of the tree.
     */
    private final int _numVars;
    /**
     * The random number generator.
     */
    private final PRNG _rnd;

    /**
     * An interface to calculate node output. Note that samples[i] is the number of sampling of
     * dataset[i]. 0 means that the datum is not included and values of greater than 1 are possible
     * because of sampling with replacement.
     */
    public interface NodeOutput {
        /**
         * Calculate the node output.
         *
         * @param samples the samples in the node.
         * @return the node output
         */
        double calculate(int[] samples);
    }

    /**
     * Regression tree node.
     */
    public static final class Node implements Externalizable {

        /**
         * Predicted real value for this node.
         */
        double output = 0.0;
        /**
         * The split feature for this node.
         */
        int splitFeature = -1;
        /**
         * The type of split feature
         */
        boolean quantitativeFeature = true;
        /**
         * The split value.
         */
        double splitValue = Double.NaN;
        /**
         * Reduction in squared error compared to parent.
         */
        double splitScore = 0.0;
        /**
         * Children node.
         */
        Node trueChild;
        /**
         * Children node.
         */
        Node falseChild;
        /**
         * Predicted output for children node.
         */
        double trueChildOutput = 0.0;
        /**
         * Predicted output for children node.
         */
        double falseChildOutput = 0.0;

        public Node() {}//for Externalizable

        public Node(double output) {
            this.output = output;
        }

        private boolean isLeaf() {
            return trueChild == null && falseChild == null;
        }

        private void markAsLeaf() {
            this.splitFeature = -1;
            this.splitValue = Double.NaN;
            this.splitScore = 0.0;
            this.trueChild = null;
            this.falseChild = null;
        }

        @VisibleForTesting
        public double predict(@Nonnull final double[] x) {
            return predict(new DenseVector(x));
        }

        /**
         * Evaluate the regression tree over an instance.
         */
        public double predict(@Nonnull final Vector x) {
            if (isLeaf()) {
                return output;
            } else {
                if (quantitativeFeature) {
                    if (x.get(splitFeature, Double.NaN) <= splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                } else {
                    if (x.get(splitFeature, Double.NaN) == splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                }
            }
        }

        public double predict(@Nonnull final Vector x, @Nonnull final PredictionHandler handler) {
            if (isLeaf()) {
                handler.visitLeaf(output);
                return output;
            } else {
                final double feature = x.get(splitFeature, Double.NaN);
                if (quantitativeFeature) {
                    if (feature <= splitValue) {
                        handler.visitBranch(LE, splitFeature, feature, splitValue);
                        return trueChild.predict(x);
                    } else {
                        handler.visitBranch(GT, splitFeature, feature, splitValue);
                        return falseChild.predict(x);
                    }
                } else {
                    if (feature == splitValue) {
                        handler.visitBranch(EQ, splitFeature, feature, splitValue);
                        return trueChild.predict(x);
                    } else {
                        handler.visitBranch(NE, splitFeature, feature, splitValue);
                        return falseChild.predict(x);
                    }
                }
            }
        }

        /**
         * Evaluate the regression tree over an instance.
         */
        public double predict(final int[] x) {
            if (isLeaf()) {
                return output;
            } else if (x[splitFeature] == (int) splitValue) {
                return trueChild.predict(x);
            } else {
                return falseChild.predict(x);
            }
        }

        public void exportJavascript(@Nonnull final StringBuilder builder,
                @Nullable final String[] featureNames, final int depth) {
            if (isLeaf()) {
                indent(builder, depth);
                builder.append(output).append(";\n");
            } else {
                if (quantitativeFeature) {
                    indent(builder, depth);
                    if (featureNames == null) {
                        builder.append("if( x[")
                               .append(splitFeature)
                               .append("] <= ")
                               .append(splitValue)
                               .append(") {\n");
                    } else {
                        builder.append("if( ")
                               .append(resolveFeatureName(splitFeature, featureNames))
                               .append(" <= ")
                               .append(splitValue)
                               .append(") {\n");
                    }
                    trueChild.exportJavascript(builder, featureNames, depth + 1);
                    indent(builder, depth);
                    builder.append("} else {\n");
                    falseChild.exportJavascript(builder, featureNames, depth + 1);
                    indent(builder, depth);
                    builder.append("}\n");
                } else {
                    indent(builder, depth);
                    if (featureNames == null) {
                        builder.append("if( x[")
                               .append(splitFeature)
                               .append("] == ")
                               .append(splitValue)
                               .append(") {\n");
                    } else {
                        builder.append("if( ")
                               .append(resolveFeatureName(splitFeature, featureNames))
                               .append(" == ")
                               .append(splitValue)
                               .append(") {\n");
                    }
                    trueChild.exportJavascript(builder, featureNames, depth + 1);
                    indent(builder, depth);
                    builder.append("} else {\n");
                    falseChild.exportJavascript(builder, featureNames, depth + 1);
                    indent(builder, depth);
                    builder.append("}\n");
                }
            }
        }

        public void exportGraphviz(@Nonnull final StringBuilder builder,
                @Nullable final String[] featureNames, @Nonnull final String outputName,
                final @Nonnull MutableInt nodeIdGenerator, final int parentNodeId) {
            final int myNodeId = nodeIdGenerator.getValue();

            if (isLeaf()) {
                builder.append(String.format(
                    " %d [label=<%s = %s>, fillcolor=\"#00000000\", shape=ellipse];\n", myNodeId,
                    outputName, Double.toString(output)));

                if (myNodeId != parentNodeId) {
                    builder.append(' ').append(parentNodeId).append(" -> ").append(myNodeId);
                    if (parentNodeId == 0) {
                        if (myNodeId == 1) {
                            builder.append(
                                " [labeldistance=2.5, labelangle=45, headlabel=\"True\"]");
                        } else {
                            builder.append(
                                " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"]");
                        }
                    }
                    builder.append(";\n");
                }
            } else {
                if (quantitativeFeature) {
                    builder.append(
                        String.format(" %d [label=<%s &le; %s>, fillcolor=\"#00000000\"];\n",
                            myNodeId, resolveFeatureName(splitFeature, featureNames),
                            Double.toString(splitValue)));
                } else {
                    builder.append(
                        String.format(" %d [label=<%s = %s>, fillcolor=\"#00000000\"];\n", myNodeId,
                            resolveFeatureName(splitFeature, featureNames),
                            Double.toString(splitValue)));
                }

                if (myNodeId != parentNodeId) {
                    builder.append(' ').append(parentNodeId).append(" -> ").append(myNodeId);
                    if (parentNodeId == 0) {//only draw edge label on top
                        if (myNodeId == 1) {
                            builder.append(
                                " [labeldistance=2.5, labelangle=45, headlabel=\"True\"]");
                        } else {
                            builder.append(
                                " [labeldistance=2.5, labelangle=-45, headlabel=\"False\"]");
                        }
                    }
                    builder.append(";\n");
                }

                nodeIdGenerator.addValue(1);
                trueChild.exportGraphviz(builder, featureNames, outputName, nodeIdGenerator,
                    myNodeId);
                nodeIdGenerator.addValue(1);
                falseChild.exportGraphviz(builder, featureNames, outputName, nodeIdGenerator,
                    myNodeId);
            }
        }

        @Deprecated
        public int opCodegen(@Nonnull final List<String> scripts, int depth) {
            int selfDepth = 0;
            final StringBuilder buf = new StringBuilder();
            if (isLeaf()) {
                buf.append("push ").append(output);
                scripts.add(buf.toString());
                buf.setLength(0);
                buf.append("goto last");
                scripts.add(buf.toString());
                selfDepth += 2;
            } else {
                if (quantitativeFeature) {
                    buf.append("push ").append("x[").append(splitFeature).append("]");
                    scripts.add(buf.toString());
                    buf.setLength(0);
                    buf.append("push ").append(splitValue);
                    scripts.add(buf.toString());
                    buf.setLength(0);
                    buf.append("ifle ");
                    scripts.add(buf.toString());
                    depth += 3;
                    selfDepth += 3;
                    int trueDepth = trueChild.opCodegen(scripts, depth);
                    selfDepth += trueDepth;
                    scripts.set(depth - 1, "ifle " + String.valueOf(depth + trueDepth));
                    int falseDepth = falseChild.opCodegen(scripts, depth + trueDepth);
                    selfDepth += falseDepth;
                } else {
                    buf.append("push ").append("x[").append(splitFeature).append("]");
                    scripts.add(buf.toString());
                    buf.setLength(0);
                    buf.append("push ").append(splitValue);
                    scripts.add(buf.toString());
                    buf.setLength(0);
                    buf.append("ifeq ");
                    scripts.add(buf.toString());
                    depth += 3;
                    selfDepth += 3;
                    int trueDepth = trueChild.opCodegen(scripts, depth);
                    selfDepth += trueDepth;
                    scripts.set(depth - 1, "ifeq " + String.valueOf(depth + trueDepth));
                    int falseDepth = falseChild.opCodegen(scripts, depth + trueDepth);
                    selfDepth += falseDepth;
                }
            }
            return selfDepth;
        }

        @Override
        public void writeExternal(ObjectOutput out) throws IOException {
            out.writeInt(splitFeature);
            out.writeByte(quantitativeFeature ? NUMERIC : NOMINAL);
            out.writeDouble(splitValue);

            if (isLeaf()) {
                out.writeBoolean(true);
                out.writeDouble(output);
            } else {
                out.writeBoolean(false);
                if (trueChild == null) {
                    out.writeBoolean(false);
                } else {
                    out.writeBoolean(true);
                    trueChild.writeExternal(out);
                }
                if (falseChild == null) {
                    out.writeBoolean(false);
                } else {
                    out.writeBoolean(true);
                    falseChild.writeExternal(out);
                }
            }
        }

        @Override
        public void readExternal(ObjectInput in) throws IOException, ClassNotFoundException {
            this.splitFeature = in.readInt();
            byte typeId = in.readByte();
            this.quantitativeFeature = (typeId == NUMERIC);
            this.splitValue = in.readDouble();

            if (in.readBoolean()) {// isLeaf()
                this.output = in.readDouble();
            } else {
                if (in.readBoolean()) {
                    this.trueChild = new Node();
                    trueChild.readExternal(in);
                }
                if (in.readBoolean()) {
                    this.falseChild = new Node();
                    falseChild.readExternal(in);
                }
            }
        }
    }

    private static void indent(final StringBuilder builder, final int depth) {
        for (int i = 0; i < depth; i++) {
            builder.append("  ");
        }
    }

    /**
     * Regression tree node for training purpose.
     */
    private final class TrainNode implements Comparable<TrainNode> {

        /**
         * The associated regression tree node.
         */
        @Nonnull
        final Node node;
        /**
         * Depth of the node in the tree
         */
        final int depth;
        /**
         * The lower bound (inclusive) in the order array of the samples belonging to this node.
         */
        final int low;
        /**
         * The upper bound (exclusive) in the order array of the samples belonging to this node.
         */
        final int high;
        /**
         * The number of samples
         */
        final int samples;
        /**
         * Child node that passes the test.
         */
        @Nullable
        TrainNode trueChild;
        /**
         * Child node that fails the test.
         */
        @Nullable
        TrainNode falseChild;

        @Nullable
        int[] constFeatures;

        public TrainNode(@Nonnull Node node, int depth, int low, int high, int samples) {
            this(node, depth, low, high, samples, new int[0]);
        }

        public TrainNode(@Nonnull Node node, int depth, int low, int high, int samples,
                @Nonnull int[] constFeatures) {
            if (low >= high) {
                throw new IllegalArgumentException(
                    "Unexpected condition was met. low=" + low + ", high=" + high);
            }
            this.node = node;
            this.depth = depth;
            this.low = low;
            this.high = high;
            this.samples = samples;
            this.constFeatures = constFeatures;
        }

        @Override
        public int compareTo(final TrainNode a) {
            return (int) Math.signum(a.node.splitScore - node.splitScore);
        }

        /**
         * Calculate the node output for leaves.
         *
         * @param output the output calculate functor.
         */
        public void calculateOutput(final NodeOutput output) {
            if (node.trueChild == null && node.falseChild == null) {
                int[] samples = getSamples();
                node.output = output.calculate(samples);
            } else {
                if (trueChild != null) {
                    trueChild.calculateOutput(output);
                }
                if (falseChild != null) {
                    falseChild.calculateOutput(output);
                }
            }
        }

        @Nonnull
        private int[] getSamples() {
            int size = high - low;
            final IntArrayList result = new IntArrayList(size);

            final int[] sampleIndex = _sampleIndex;
            final int[] samples = _samples;
            for (int i = low, end = high; i < end; i++) {
                int index = sampleIndex[i];
                int sample = samples[index];
                if (sample > 0) {
                    result.add(index);
                }
            }

            return result.toArray(true);
        }

        /**
         * Finds the best attribute to split on at the current node. Returns true if a split exists
         * to reduce squared error, false otherwise.
         */
        public boolean findBestSplit() {
            // avoid split if tree depth is larger than threshold
            if (depth >= _maxDepth) {
                return false;
            }
            // avoid split if the number of samples is less than threshold
            if (samples <= _minSamplesSplit) {
                return false;
            }

            final int[] constFeatures_ = this.constFeatures;

            // Loop through features and compute the reduction of squared error,
            // which is trueCount * trueMean^2 + falseCount * falseMean^2 - count * parentMean^2
            final double sum = node.output * samples;
            for (int varJ : variableIndex()) {
                if (ArrayUtils.contains(constFeatures_, varJ)) {
                    continue;
                }
                final Node split = findBestSplit(samples, sum, varJ);
                if (split.splitScore > node.splitScore) {
                    node.splitFeature = split.splitFeature;
                    node.quantitativeFeature = split.quantitativeFeature;
                    node.splitValue = split.splitValue;
                    node.splitScore = split.splitScore;
                    node.trueChildOutput = split.trueChildOutput;
                    node.falseChildOutput = split.falseChildOutput;
                }
            }

            return node.splitFeature != -1;
        }

        @Nonnull
        private int[] variableIndex() {
            final Matrix X = _X;
            final IntReservoirSampler sampler = new IntReservoirSampler(_numVars, _rnd.nextLong());
            if (X.isSparse()) {
                // sample columns from sampled examples
                final RoaringBitmap cols = new RoaringBitmap();
                final VectorProcedure proc = new VectorProcedure() {
                    public void apply(final int col) {
                        cols.add(col);
                    }
                };
                final int[] sampleIndex = _sampleIndex;
                for (int i = low, end = high; i < end; i++) {
                    int row = sampleIndex[i];
                    X.eachColumnIndexInRow(row, proc);
                }
                cols.forEach(new IntConsumer() {
                    public void accept(final int k) {
                        sampler.add(k);
                    }
                });
            } else {
                final int ncols = X.numColumns();
                for (int i = 0; i < ncols; i++) {
                    sampler.add(i);
                }
            }
            return sampler.getSample();
        }

        /**
         * Finds the best split cutoff for attribute j at the current node.
         *
         * @param n the number instances in this node.
         * @param count the sample count in each class.
         * @param impurity the impurity of this node.
         * @param j the attribute to split on.
         */
        private Node findBestSplit(final int n, final double sum, final int j) {
            final int[] samples = _samples;
            final int[] sampleIndex = _sampleIndex;
            final Matrix X = _X;
            final double[] y = _y;

            final Node split = new Node(0.d);

            if (_nominalAttrs.contains(j)) {// nominal
                final Int2DoubleOpenHashMap trueSum = new Int2DoubleOpenHashMap();
                final Int2IntOpenHashMap trueCount = new Int2IntOpenHashMap();

                int countNaN = 0;
                for (int i = low, end = high; i < end; i++) {
                    final int index = sampleIndex[i];
                    final int numSamples = samples[index];
                    if (numSamples == 0) {
                        continue;
                    }

                    // For each true feature of this datum increment the
                    // sufficient statistics for the "true" branch to evaluate
                    // splitting on this feature.
                    final double v = X.get(i, j, Double.NaN);
                    if (Double.isNaN(v)) {
                        countNaN++;
                        continue;
                    }
                    int x_ij = (int) v;

                    trueSum.addTo(x_ij, y[i]);
                    trueCount.addTo(x_ij, 1);
                }
                final int countDistinctX = trueCount.size() + (countNaN == 0 ? 0 : 1);
                if (countDistinctX <= 1) { // mark as a constant feature
                    this.constFeatures = ArrayUtils.sortedArraySet(constFeatures, j);
                }

                for (Entry e : trueCount.int2IntEntrySet()) {
                    final int k = e.getIntKey();
                    final double tc = e.getIntValue();

                    final double fc = n - tc;

                    // skip splitting
                    if (tc < _minSamplesSplit || fc < _minSamplesSplit) {
                        continue;
                    }

                    // compute penalized means
                    double trueSum_k = trueSum.get(k);
                    final double trueMean = trueSum_k / tc;
                    final double falseMean = (sum - trueSum_k) / fc;

                    final double gain = (tc * trueMean * trueMean + fc * falseMean * falseMean)
                            - n * split.output * split.output;
                    if (gain > split.splitScore) {
                        // new best split
                        split.splitFeature = j;
                        split.quantitativeFeature = false;
                        split.splitValue = k;
                        split.splitScore = gain;
                        split.trueChildOutput = trueMean;
                        split.falseChildOutput = falseMean;
                    }
                }
            } else {
                final MutableInt countNaN = new MutableInt(0);
                final MutableInt replaceCount = new MutableInt(0);

                _order.eachNonNullInColumn(j, low, high, new Consumer() {
                    double trueSum = 0.0;
                    int trueCount = 0;
                    double prevx = Double.NaN, lastx = Double.NaN;

                    public void accept(int pos, final int i) {
                        final int numSamples = samples[i];
                        if (numSamples == 0) {
                            return;
                        }

                        final double x_ij = _X.get(i, j, Double.NaN);
                        if (Double.isNaN(x_ij)) {
                            countNaN.incr();
                            return;
                        }
                        if (lastx != x_ij) {
                            lastx = x_ij;
                            replaceCount.incr();
                        }

                        final double y_i = _y[i];
                        if (Double.isNaN(prevx) || x_ij == prevx) {
                            prevx = x_ij;
                            trueSum += numSamples * y_i;
                            trueCount += numSamples;
                            return;
                        }

                        final double falseCount = n - trueCount;

                        // If either side is empty, skip this feature.
                        if (trueCount < _minSamplesSplit || falseCount < _minSamplesSplit) {
                            prevx = x_ij;
                            trueSum += numSamples * y_i;
                            trueCount += numSamples;
                            return;
                        }

                        // compute penalized means
                        final double trueMean = trueSum / trueCount;
                        final double falseMean = (sum - trueSum) / falseCount;

                        // The gain is actually -(reduction in squared error) for
                        // sorting in priority queue, which treats smaller number with
                        // higher priority.
                        final double gain = (trueCount * trueMean * trueMean
                                + falseCount * falseMean * falseMean)
                                - n * split.output * split.output;
                        if (gain > split.splitScore) {
                            // new best split
                            split.splitFeature = j;
                            split.quantitativeFeature = true;
                            split.splitValue = (x_ij + prevx) / 2;
                            split.splitScore = gain;
                            split.trueChildOutput = trueMean;
                            split.falseChildOutput = falseMean;
                        }

                        prevx = x_ij;
                        trueSum += numSamples * y_i;
                        trueCount += numSamples;
                    }//apply
                });

                final int countDistinctX = replaceCount.get() + (countNaN.get() == 0 ? 0 : 1);
                if (countDistinctX <= 1) { // mark as a constant feature
                    this.constFeatures = ArrayUtils.sortedArraySet(constFeatures, j);
                }
            }

            return split;
        }

        /**
         * Split the node into two children nodes. Returns true if split success.
         */
        public boolean split(@Nullable final PriorityQueue<TrainNode> nextSplits) {
            if (node.splitFeature < 0) {
                throw new IllegalStateException("Split a node with invalid feature.");
            }

            final IntPredicate goesLeft = getPredicate();

            // split samples
            final int tc, fc, pivot;
            {
                MutableInt tc_ = new MutableInt(0);
                MutableInt fc_ = new MutableInt(0);
                pivot = splitSamples(tc_, fc_, goesLeft);
                tc = tc_.get();
                fc = fc_.get();
            }

            if (tc < _minSamplesLeaf || fc < _minSamplesLeaf) {
                node.markAsLeaf();
                return false;
            }

            partitionOrder(low, pivot, high, goesLeft);

            int leaves = 0;

            node.trueChild = new Node(node.trueChildOutput);
            this.trueChild =
                    new TrainNode(node.trueChild, depth + 1, low, pivot, tc, constFeatures.clone());
            node.falseChild = new Node(node.falseChildOutput);
            this.falseChild =
                    new TrainNode(node.falseChild, depth + 1, pivot, high, fc, constFeatures);
            this.constFeatures = null;

            if (tc >= _minSamplesSplit && trueChild.findBestSplit()) {
                if (nextSplits != null) {
                    nextSplits.add(trueChild);
                } else {
                    if (trueChild.split(null) == false) {
                        leaves++;
                    }
                }
            } else {
                leaves++;
            }

            if (fc >= _minSamplesSplit && falseChild.findBestSplit()) {
                if (nextSplits != null) {
                    nextSplits.add(falseChild);
                } else {
                    if (falseChild.split(null) == false) {
                        leaves++;
                    }
                }
            } else {
                leaves++;
            }

            // Prune meaningless branches
            if (leaves == 2) {// both left and right child is leaf node
                if (node.trueChild.output == node.falseChild.output) {// found meaningless branch
                    node.markAsLeaf();
                    return false;
                }
            }

            _importance.incr(node.splitFeature, node.splitScore);

            return true;
        }

        /**
         * @return Pivot to split samples
         */
        private int splitSamples(@Nonnull final MutableInt tc, @Nonnull final MutableInt fc,
                @Nonnull final IntPredicate goesLeft) {
            final int[] sampleIndex = _sampleIndex;
            final int[] samples = _samples;

            int pivot = low;
            for (int k = low, end = high; k < end; k++) {
                final int i = sampleIndex[k];
                final int numSamples = samples[i];
                if (goesLeft.test(i)) {
                    tc.addValue(numSamples);
                    pivot++;
                } else {
                    fc.addValue(numSamples);
                }
            }
            return pivot;
        }

        /**
         * Modifies {@link #_order} and {@link #_sampleIndex} by partitioning the range from low
         * (inclusive) to high (exclusive) so that all elements i for which goesLeft(i) is true come
         * before all elements for which it is false, but element ordering is otherwise preserved.
         * The number of true values returned by goesLeft must equal split-low.
         * 
         * @param low the low bound of the segment of the order arrays which will be partitioned.
         * @param split where the partition's split point will end up.
         * @param high the high bound of the segment of the order arrays which will be partitioned.
         * @param goesLeft whether an element goes to the left side or the right side of the
         *        partition.
         */
        private void partitionOrder(final int low, final int pivot, final int high,
                @Nonnull final IntPredicate goesLeft) {
            final int[] buf = new int[high - pivot];
            _order.eachRow(new Consumer() {
                @Override
                public void accept(int col, @Nonnull final SparseIntArray row) {
                    partitionArray(row, low, pivot, high, goesLeft, buf);
                }
            });
            partitionArray(_sampleIndex, low, pivot, high, goesLeft, buf);
        }

        @Nonnull
        private IntPredicate getPredicate() {
            if (node.quantitativeFeature) {
                return new IntPredicate() {
                    @Override
                    public boolean test(int i) {
                        return _X.get(i, node.splitFeature, Double.NaN) <= node.splitValue;
                    }
                };
            } else {
                return new IntPredicate() {
                    @Override
                    public boolean test(int i) {
                        return _X.get(i, node.splitFeature, Double.NaN) == node.splitValue;
                    }
                };
            }
        }

    }

    private static void partitionArray(@Nonnull final SparseIntArray a, final int low,
            final int pivot, final int high, @Nonnull final IntPredicate goesLeft,
            @Nonnull final int[] buf) {
        final int[] keys = a.keys();
        final int[] values = a.values();
        final int size = a.size();

        final int startPos = ArrayUtils.insertionPoint(keys, size, low);
        final int endPos = ArrayUtils.insertionPoint(keys, size, high);
        int pos = startPos, k = 0;
        for (int i = startPos, j = 0; i < endPos; i++) {
            final int a_i = values[i];
            if (goesLeft.test(a_i)) {
                keys[pos] = low + j;
                values[pos] = a_i;
                pos++;
                j++;
            } else {
                if (k >= buf.length) {
                    throw new IndexOutOfBoundsException(String.format(
                        "low=%d, pivot=%d, high=%d, a.size()=%d, buf.length=%d, i=%d, j=%d, k=%d",
                        low, pivot, high, a.size(), buf.length, i, j, k));
                }
                buf[k++] = a_i;
            }
        }
        for (int i = 0; i < k; i++) {
            keys[pos] = pivot + i;
            values[pos] = buf[i];
            pos++;
        }
        if (pos != endPos) {
            throw new IllegalStateException(
                String.format("pos=%d, startPos=%d, endPos=%d, k=%d", pos, startPos, endPos, k));
        }
    }

    /**
     * Modifies an array in-place by partitioning the range from low (inclusive) to high (exclusive)
     * so that all elements i for which goesLeft(i) is true come before all elements for which it is
     * false, but element ordering is otherwise preserved. The number of true values returned by
     * goesLeft must equal split-low. buf is scratch space large enough (i.e., at least high-split
     * long) to hold all elements for which goesLeft is false.
     */
    private static void partitionArray(@Nonnull final int[] a, final int low, final int pivot,
            final int high, @Nonnull final IntPredicate goesLeft, @Nonnull final int[] buf) {
        int j = low;
        int k = 0;
        for (int i = low; i < high; i++) {
            if (i >= a.length) {
                throw new IndexOutOfBoundsException(String.format(
                    "low=%d, pivot=%d, high=%d, a.length=%d, buf.length=%d, i=%d, j=%d, k=%d", low,
                    pivot, high, a.length, buf.length, i, j, k));
            }
            final int a_i = a[i];
            if (goesLeft.test(a_i)) {
                a[j++] = a_i;
            } else {
                if (k >= buf.length) {
                    throw new IndexOutOfBoundsException(String.format(
                        "low=%d, pivot=%d, high=%d, a.length=%d, buf.length=%d, i=%d, j=%d, k=%d",
                        low, pivot, high, a.length, buf.length, i, j, k));
                }
                buf[k++] = a_i;
            }
        }
        if (k != high - pivot || j != pivot) {
            throw new IndexOutOfBoundsException(
                String.format("low=%d, pivot=%d, high=%d, a.length=%d, buf.length=%d, j=%d, k=%d",
                    low, pivot, high, a.length, buf.length, j, k));
        }
        System.arraycopy(buf, 0, a, pivot, k);
    }

    /**
     * Prunes redundant leaves from the tree. In some cases, a node is split into two leaves that
     * get assigned the same label, so this recursively combines leaves when it notices this
     * situation.
     */
    private static void pruneRedundantLeaves(@Nonnull final Node node, @Nonnull Vector importance) {
        if (node.isLeaf()) {
            return;
        }

        // The children might not be leaves now, but might collapse into leaves given the chance.
        pruneRedundantLeaves(node.trueChild, importance);
        pruneRedundantLeaves(node.falseChild, importance);

        if (node.trueChild.isLeaf() && node.falseChild.isLeaf()
                && node.trueChild.output == node.falseChild.output) {
            node.trueChild = null;
            node.falseChild = null;
            importance.decr(node.splitFeature, node.splitScore);
        }
    }


    public RegressionTree(@Nullable RoaringBitmap nominalAttrs, @Nonnull Matrix x,
            @Nonnull double[] y, int maxLeafs) {
        this(nominalAttrs, x, y, x.numColumns(), Integer.MAX_VALUE, maxLeafs, 5, 1, null, null);
    }

    public RegressionTree(@Nullable RoaringBitmap nominalAttrs, @Nonnull Matrix x,
            @Nonnull double[] y, int maxLeafs, @Nullable PRNG rand) {
        this(nominalAttrs, x, y, x.numColumns(), Integer.MAX_VALUE, maxLeafs, 5, 1, null, rand);
    }

    public RegressionTree(@Nullable RoaringBitmap nominalAttrs, @Nonnull Matrix x,
            @Nonnull double[] y, int numVars, int maxDepth, int maxLeafNodes, int minSamplesSplit,
            int minSamplesLeaf, @Nullable int[] samples, @Nullable PRNG rand) {
        this(nominalAttrs, x, y, numVars, maxDepth, maxLeafNodes, minSamplesSplit, minSamplesLeaf, samples, null, rand);
    }

    /**
     * Constructor. Learns a regression tree for gradient tree boosting.
     *
     * @param nominalAttrs the attribute properties.
     * @param x the training instances.
     * @param y the response variable.
     * @param numVars the number of input variables to pick to split on at each node. It seems that
     *        dim/3 give generally good performance, where dim is the number of variables.
     * @param maxLeafNodes the maximum number of leaf nodes in the tree.
     * @param minSamplesLeaf number of instances in a node below which the tree will not split,
     *        setting 5 generally gives good results.
     * @param samples the sample set of instances for stochastic learning.
     * @param output An interface to calculate node output.
     */
    public RegressionTree(@Nullable RoaringBitmap nominalAttrs, @Nonnull Matrix x,
            @Nonnull double[] y, int numVars, int maxDepth, int maxLeafNodes, int minSamplesSplit,
            int minSamplesLeaf, @Nullable int[] samples, @Nullable NodeOutput output,
            @Nullable PRNG rand) {
        checkArgument(x, y, numVars, maxDepth, maxLeafNodes, minSamplesSplit, minSamplesLeaf);

        this._X = x;
        this._y = y;

        if (nominalAttrs == null) {
            nominalAttrs = new RoaringBitmap();
        }
        this._nominalAttrs = nominalAttrs;

        this._numVars = numVars;
        this._maxDepth = maxDepth;
        // min_sample_leaf >= 2 is satisfied iff min_sample_split >= 4
        // So, split only happens when samples in intermediate nodes has >= 2 * min_sample_leaf nodes.
        if (minSamplesSplit < minSamplesLeaf * 2) {
            if (logger.isInfoEnabled()) {
                logger.info(String.format(
                    "min_sample_leaf = %d replaces min_sample_split = %d with min_sample_split = %d",
                    minSamplesLeaf, minSamplesSplit, minSamplesLeaf * 2));
            }
            minSamplesSplit = minSamplesLeaf * 2;
        }
        this._minSamplesSplit = minSamplesSplit;
        this._minSamplesLeaf = minSamplesLeaf;
        this._importance = x.isSparse() ? new SparseVector() : new DenseVector(x.numColumns());
        this._rnd = (rand == null) ? RandomNumberGeneratorFactory.createPRNG() : rand;

        int n = 0;
        double sum = 0.0;
        final int[] sampleIndex;
        if (samples == null) {
            n = y.length;
            samples = new int[n];
            sampleIndex = new int[n];
            for (int i = 0; i < n; i++) {
                samples[i] = 1;
                sum += y[i];
                sampleIndex[i] = i;
            }
        } else {
            final IntArrayList positions = new IntArrayList(n);
            for (int i = 0, end = y.length; i < end; i++) {
                final int sample = samples[i];
                if (sample != 0) {
                    n += sample;
                    sum += sample * y[i];
                    positions.add(i);
                }
            }
            sampleIndex = positions.toArray(true);
        }
        this._samples = samples;
        this._order = SmileExtUtils.sort(nominalAttrs, x, samples);
        this._sampleIndex = sampleIndex;

        this._root = new Node(sum / n);

        TrainNode trainRoot = new TrainNode(_root, 1, 0, _sampleIndex.length, n);
        if (maxLeafNodes == Integer.MAX_VALUE) {
            if (trainRoot.findBestSplit()) {
                trainRoot.split(null);
            }
        } else {
            // Priority queue for best-first tree growing.
            PriorityQueue<TrainNode> nextSplits = new PriorityQueue<TrainNode>();
            // Now add splits to the tree until max tree size is reached
            if (trainRoot.findBestSplit()) {
                nextSplits.add(trainRoot);
            }
            // Pop best leaf from priority queue, split it, and push
            // children nodes into the queue if possible.
            for (int leaves = 1; leaves < maxLeafNodes; leaves++) {
                // parent is the leaf to split
                TrainNode node = nextSplits.poll();
                if (node == null) {
                    break;
                }
                if (!node.split(nextSplits)) { // Split the parent node into two children nodes
                    leaves--;
                }
            }
            pruneRedundantLeaves(_root, _importance);
        }

        if (output != null) {
            trainRoot.calculateOutput(output);
        }
    }

    private static void checkArgument(@Nonnull Matrix x, @Nonnull double[] y, int numVars,
            int maxDepth, int maxLeafNodes, int minSamplesSplit, int minSamplesLeaf) {
        if (x.numRows() != y.length) {
            throw new IllegalArgumentException(
                String.format("The sizes of X and Y don't match: %d != %d", x.numRows(), y.length));
        }
        if (y.length == 0) {
            throw new IllegalArgumentException("No training example given");
        }
        if (numVars <= 0 || numVars > x.numColumns()) {
            throw new IllegalArgumentException(
                "Invalid number of variables to split on at a node of the tree: " + numVars);
        }
        if (maxDepth < 2) {
            throw new IllegalArgumentException("maxDepth should be greater than 1: " + maxDepth);
        }
        if (maxLeafNodes < 2) {
            throw new IllegalArgumentException("Invalid maximum leaves: " + maxLeafNodes);
        }
        if (minSamplesSplit < 2) {
            throw new IllegalArgumentException(
                "Invalid minimum number of samples required to split an internal node: "
                        + minSamplesSplit);
        }
        if (minSamplesLeaf < 1) {
            throw new IllegalArgumentException(
                "Invalid minimum size of leaf nodes: " + minSamplesLeaf);
        }
    }

    /**
     * Returns the variable importance. Every time a split of a node is made on variable the
     * impurity criterion for the two descendent nodes is less than the parent node. Adding up the
     * decreases for each individual variable over the tree gives a simple measure of variable
     * importance.
     *
     * @return the variable importance
     */
    public Vector importance() {
        return _importance;
    }

    @VisibleForTesting
    public double predict(@Nonnull final double[] x) {
        return predict(new DenseVector(x));
    }

    @Override
    public double predict(@Nonnull final Vector x) {
        return _root.predict(x);
    }

    @Nonnull
    public String predictJsCodegen(@Nonnull final String[] featureNames) {
        StringBuilder buf = new StringBuilder(1024);
        _root.exportJavascript(buf, featureNames, 0);
        return buf.toString();
    }

    @Deprecated
    @Nonnull
    public String predictOpCodegen(@Nonnull String sep) {
        List<String> opslist = new ArrayList<String>();
        _root.opCodegen(opslist, 0);
        opslist.add("call end");
        String scripts = StringUtils.concat(opslist, sep);
        return scripts;
    }

    @Nonnull
    public byte[] serialize(boolean compress) throws HiveException {
        try {
            if (compress) {
                return ObjectUtils.toCompressedBytes(_root);
            } else {
                return ObjectUtils.toBytes(_root);
            }
        } catch (IOException ioe) {
            throw new HiveException("IOException cause while serializing DecisionTree object", ioe);
        } catch (Exception e) {
            throw new HiveException("Exception cause while serializing DecisionTree object", e);
        }
    }

    @Nonnull
    public static Node deserialize(@Nonnull final byte[] serializedObj, final int length,
            final boolean compressed) throws HiveException {
        final Node root = new Node();
        try {
            if (compressed) {
                ObjectUtils.readCompressedObject(serializedObj, 0, length, root);
            } else {
                ObjectUtils.readObject(serializedObj, length, root);
            }
        } catch (IOException ioe) {
            throw new HiveException("IOException cause while deserializing DecisionTree object",
                ioe);
        } catch (Exception e) {
            throw new HiveException("Exception cause while deserializing DecisionTree object", e);
        }
        return root;
    }

    @Override
    public String toString() {
        return _root == null ? "" : predictJsCodegen(null);
    }
}
