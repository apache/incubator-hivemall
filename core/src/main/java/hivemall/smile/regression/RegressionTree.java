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
package hivemall.smile.regression;

import static hivemall.smile.utils.SmileExtUtils.resolveFeatureName;
import hivemall.annotations.VisibleForTesting;
import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.ints.ColumnMajorIntMatrix;
import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.math.vector.DenseVector;
import hivemall.math.vector.Vector;
import hivemall.math.vector.VectorProcedure;
import hivemall.smile.data.Attribute;
import hivemall.smile.data.Attribute.AttributeType;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.collections.sets.IntArraySet;
import hivemall.utils.collections.sets.IntSet;
import hivemall.utils.lang.ObjectUtils;
import hivemall.utils.lang.mutable.MutableInt;
import hivemall.utils.math.MathUtils;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;
import java.util.PriorityQueue;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.metadata.HiveException;

import smile.math.Math;
import smile.regression.GradientTreeBoost;
import smile.regression.RandomForest;
import smile.regression.Regression;

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
    /**
     * The attributes of independent variable.
     */
    private final Attribute[] _attributes;
    private final boolean _hasNumericType;
    /**
     * Variable importance. Every time a split of a node is made on variable the impurity criterion
     * for the two descendant nodes is less than the parent node. Adding up the decreases for each
     * individual variable over the tree gives a simple measure of variable importance.
     */
    private final double[] _importance;
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
    private final int _minSplit;
    /**
     * The minimum number of samples in a leaf node
     */
    private final int _minLeafSize;
    /**
     * The number of input variables to be used to determine the decision at a node of the tree.
     */
    private final int _numVars;
    /**
     * The index of training values in ascending order. Note that only numeric attributes will be
     * sorted.
     */
    private final ColumnMajorIntMatrix _order;

    private final PRNG _rnd;

    private final NodeOutput _nodeOutput;

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
        public double calculate(int[] samples);
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
        AttributeType splitFeatureType = null;
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

        @VisibleForTesting
        public double predict(@Nonnull final double[] x) {
            return predict(new DenseVector(x));
        }

        /**
         * Evaluate the regression tree over an instance.
         */
        public double predict(@Nonnull final Vector x) {
            if (trueChild == null && falseChild == null) {
                return output;
            } else {
                if (splitFeatureType == AttributeType.NOMINAL) {
                    if (x.get(splitFeature, Double.NaN) == splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                } else if (splitFeatureType == AttributeType.NUMERIC) {
                    if (x.get(splitFeature, Double.NaN) <= splitValue) {
                        return trueChild.predict(x);
                    } else {
                        return falseChild.predict(x);
                    }
                } else {
                    throw new IllegalStateException("Unsupported attribute type: "
                            + splitFeatureType);
                }
            }
        }

        /**
         * Evaluate the regression tree over an instance.
         */
        public double predict(final int[] x) {
            if (trueChild == null && falseChild == null) {
                return output;
            } else if (x[splitFeature] == (int) splitValue) {
                return trueChild.predict(x);
            } else {
                return falseChild.predict(x);
            }
        }

        public void exportJavascript(@Nonnull final StringBuilder builder,
                @Nullable final String[] featureNames, final int depth) {
            if (trueChild == null && falseChild == null) {
                indent(builder, depth);
                builder.append(output).append(";\n");
            } else {
                if (splitFeatureType == AttributeType.NOMINAL) {
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
                } else if (splitFeatureType == AttributeType.NUMERIC) {
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
                    throw new IllegalStateException("Unsupported attribute type: "
                            + splitFeatureType);
                }
            }
        }

        public void exportGraphviz(@Nonnull final StringBuilder builder,
                @Nullable final String[] featureNames, @Nonnull final String outputName,
                final @Nonnull MutableInt nodeIdGenerator, final int parentNodeId) {
            final int myNodeId = nodeIdGenerator.getValue();

            if (trueChild == null && falseChild == null) {
                builder.append(String.format(
                    " %d [label=<%s = %s>, fillcolor=\"#00000000\", shape=ellipse];\n", myNodeId,
                    outputName, Double.toString(output)));

                if (myNodeId != parentNodeId) {
                    builder.append(' ').append(parentNodeId).append(" -> ").append(myNodeId);
                    if (parentNodeId == 0) {
                        if (myNodeId == 1) {
                            builder.append(" [labeldistance=2.5, labelangle=45, headlabel=\"True\"]");
                        } else {
                            builder.append(" [labeldistance=2.5, labelangle=-45, headlabel=\"False\"]");
                        }
                    }
                    builder.append(";\n");
                }
            } else {
                if (splitFeatureType == AttributeType.NOMINAL) {
                    builder.append(String.format(
                        " %d [label=<%s = %s>, fillcolor=\"#00000000\"];\n", myNodeId,
                        resolveFeatureName(splitFeature, featureNames), Double.toString(splitValue)));
                } else if (splitFeatureType == AttributeType.NUMERIC) {
                    builder.append(String.format(
                        " %d [label=<%s &le; %s>, fillcolor=\"#00000000\"];\n", myNodeId,
                        resolveFeatureName(splitFeature, featureNames), Double.toString(splitValue)));
                } else {
                    throw new IllegalStateException("Unsupported attribute type: "
                            + splitFeatureType);
                }

                if (myNodeId != parentNodeId) {
                    builder.append(' ').append(parentNodeId).append(" -> ").append(myNodeId);
                    if (parentNodeId == 0) {//only draw edge label on top 
                        if (myNodeId == 1) {
                            builder.append(" [labeldistance=2.5, labelangle=45, headlabel=\"True\"]");
                        } else {
                            builder.append(" [labeldistance=2.5, labelangle=-45, headlabel=\"False\"]");
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


        @Override
        public void writeExternal(ObjectOutput out) throws IOException {
            out.writeInt(splitFeature);
            if (splitFeatureType == null) {
                out.writeByte(-1);
            } else {
                out.writeByte(splitFeatureType.getTypeId());
            }
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
            if (typeId == -1) {
                this.splitFeatureType = null;
            } else {
                this.splitFeatureType = AttributeType.resolve(typeId);
            }
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
        final Node node;
        /**
         * Child node that passes the test.
         */
        TrainNode trueChild;
        /**
         * Child node that fails the test.
         */
        TrainNode falseChild;
        /**
         * Training dataset.
         */
        final Matrix x;
        /**
         * Training data response value.
         */
        final double[] y;

        int[] bags;

        final int depth;

        /**
         * Constructor.
         */
        public TrainNode(Node node, Matrix x, double[] y, int[] bags, int depth) {
            this.node = node;
            this.x = x;
            this.y = y;
            this.bags = bags;
            this.depth = depth;
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
                int[] samples = SmileExtUtils.bagsToSamples(bags);
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
            final int numSamples = bags.length;
            if (numSamples <= _minSplit) {
                return false;
            }

            final double sum = node.output * numSamples;


            // Loop through features and compute the reduction of squared error,
            // which is trueCount * trueMean^2 + falseCount * falseMean^2 - count * parentMean^2      
            final int[] samples = _hasNumericType ? SmileExtUtils.bagsToSamples(bags, x.numRows())
                    : null;
            for (int varJ : variableIndex(x, bags)) {
                final Node split = findBestSplit(numSamples, sum, varJ, samples);
                if (split.splitScore > node.splitScore) {
                    node.splitFeature = split.splitFeature;
                    node.splitFeatureType = split.splitFeatureType;
                    node.splitValue = split.splitValue;
                    node.splitScore = split.splitScore;
                    node.trueChildOutput = split.trueChildOutput;
                    node.falseChildOutput = split.falseChildOutput;
                }
            }

            return node.splitFeature != -1;
        }

        private int[] variableIndex(@Nonnull final Matrix x, @Nonnull final int[] bags) {
            final int[] variableIndex;
            if (x.isSparse()) {
                final IntSet cols = new IntArraySet(_numVars);
                final VectorProcedure proc = new VectorProcedure() {
                    public void apply(int col, double value) {
                        cols.add(col);
                    }
                };
                for (final int row : bags) {
                    x.eachNonNullInRow(row, proc);
                }
                variableIndex = cols.toArray(false);
            } else {
                variableIndex = MathUtils.permutation(_attributes.length);
            }

            if (_numVars < variableIndex.length) {
                SmileExtUtils.shuffle(variableIndex, _rnd);
                return Arrays.copyOf(variableIndex, _numVars);

            }
            return variableIndex;
        }

        /**
         * Finds the best split cutoff for attribute j at the current node.
         * 
         * @param n the number instances in this node.
         * @param count the sample count in each class.
         * @param impurity the impurity of this node.
         * @param j the attribute to split on.
         */
        private Node findBestSplit(final int n, final double sum, final int j,
                @Nullable final int[] samples) {
            final Node split = new Node(0.d);
            if (_attributes[j].type == AttributeType.NOMINAL) {
                final int m = _attributes[j].getSize();
                final double[] trueSum = new double[m];
                final int[] trueCount = new int[m];

                for (int b = 0, size = bags.length; b < size; b++) {
                    int i = bags[b];
                    // For each true feature of this datum increment the
                    // sufficient statistics for the "true" branch to evaluate
                    // splitting on this feature.
                    final double v = x.get(i, j, Double.NaN);
                    if (Double.isNaN(v)) {
                        continue;
                    }
                    int index = (int) v;
                    trueSum[index] += y[i];
                    ++trueCount[index];
                }

                for (int k = 0; k < m; k++) {
                    final double tc = (double) trueCount[k];
                    final double fc = n - tc;

                    // skip splitting
                    if (tc < _minSplit || fc < _minSplit) {
                        continue;
                    }

                    // compute penalized means
                    final double trueMean = trueSum[k] / tc;
                    final double falseMean = (sum - trueSum[k]) / fc;

                    final double gain = (tc * trueMean * trueMean + fc * falseMean * falseMean) - n
                            * split.output * split.output;
                    if (gain > split.splitScore) {
                        // new best split
                        split.splitFeature = j;
                        split.splitFeatureType = AttributeType.NOMINAL;
                        split.splitValue = k;
                        split.splitScore = gain;
                        split.trueChildOutput = trueMean;
                        split.falseChildOutput = falseMean;
                    }
                }
            } else if (_attributes[j].type == AttributeType.NUMERIC) {

                _order.eachNonNullInColumn(j, new VectorProcedure() {
                    double trueSum = 0.0;
                    int trueCount = 0;
                    double prevx = Double.NaN;

                    public void apply(final int row, final int i) {
                        final int sample = samples[i];
                        if (sample == 0) {
                            return;
                        }
                        final double x_ij = x.get(i, j, Double.NaN);
                        if (Double.isNaN(x_ij)) {
                            return;
                        }
                        final double y_i = y[i];

                        if (Double.isNaN(prevx) || x_ij == prevx) {
                            prevx = x_ij;
                            trueSum += sample * y_i;
                            trueCount += sample;
                            return;
                        }

                        final double falseCount = n - trueCount;

                        // If either side is empty, skip this feature.
                        if (trueCount < _minSplit || falseCount < _minSplit) {
                            prevx = x_ij;
                            trueSum += sample * y_i;
                            trueCount += sample;
                            return;
                        }

                        // compute penalized means
                        final double trueMean = trueSum / trueCount;
                        final double falseMean = (sum - trueSum) / falseCount;

                        // The gain is actually -(reduction in squared error) for
                        // sorting in priority queue, which treats smaller number with
                        // higher priority.
                        final double gain = (trueCount * trueMean * trueMean + falseCount
                                * falseMean * falseMean)
                                - n * split.output * split.output;
                        if (gain > split.splitScore) {
                            // new best split
                            split.splitFeature = j;
                            split.splitFeatureType = AttributeType.NUMERIC;
                            split.splitValue = (x_ij + prevx) / 2;
                            split.splitScore = gain;
                            split.trueChildOutput = trueMean;
                            split.falseChildOutput = falseMean;
                        }

                        prevx = x_ij;
                        trueSum += sample * y_i;
                        trueCount += sample;
                    }//apply
                });

            } else {
                throw new IllegalStateException("Unsupported attribute type: "
                        + _attributes[j].type);
            }

            return split;
        }

        /**
         * Split the node into two children nodes. Returns true if split success.
         */
        public boolean split(final PriorityQueue<TrainNode> nextSplits) {
            if (node.splitFeature < 0) {
                throw new IllegalStateException("Split a node with invalid feature.");
            }

            // split sample bags
            int childBagSize = (int) (bags.length * 0.4);
            IntArrayList trueBags = new IntArrayList(childBagSize);
            IntArrayList falseBags = new IntArrayList(childBagSize);
            int tc = splitSamples(trueBags, falseBags);
            int fc = bags.length - tc;

            if (tc < _minLeafSize || fc < _minLeafSize) {
                // set as a leaf node
                node.splitFeature = -1;
                node.splitFeatureType = null;
                node.splitValue = Double.NaN;
                node.splitScore = 0.0;
                if (_nodeOutput == null) {
                    this.bags = null;
                }
                return false;
            }

            this.bags = null; // help GC for recursive call

            node.trueChild = new Node(node.trueChildOutput);
            this.trueChild = new TrainNode(node.trueChild, x, y, trueBags.toArray(), depth + 1);
            trueBags = null; // help GC for recursive call
            if (tc >= _minSplit && trueChild.findBestSplit()) {
                if (nextSplits != null) {
                    nextSplits.add(trueChild);
                } else {
                    trueChild.split(null);
                }
            }

            node.falseChild = new Node(node.falseChildOutput);
            this.falseChild = new TrainNode(node.falseChild, x, y, falseBags.toArray(), depth + 1);
            falseBags = null; // help GC for recursive call
            if (fc >= _minSplit && falseChild.findBestSplit()) {
                if (nextSplits != null) {
                    nextSplits.add(falseChild);
                } else {
                    falseChild.split(null);
                }
            }

            _importance[node.splitFeature] += node.splitScore;

            return true;
        }

        /**
         * @return the number of true samples
         */
        private int splitSamples(@Nonnull final IntArrayList trueBags,
                @Nonnull final IntArrayList falseBags) {
            int tc = 0;
            if (node.splitFeatureType == AttributeType.NOMINAL) {
                final int splitFeature = node.splitFeature;
                final double splitValue = node.splitValue;
                for (int i = 0, size = bags.length; i < size; i++) {
                    final int index = bags[i];
                    if (x.get(index, splitFeature, Double.NaN) == splitValue) {
                        trueBags.add(index);
                        tc++;
                    } else {
                        falseBags.add(index);
                    }
                }
            } else if (node.splitFeatureType == AttributeType.NUMERIC) {
                final int splitFeature = node.splitFeature;
                final double splitValue = node.splitValue;
                for (int i = 0, size = bags.length; i < size; i++) {
                    final int index = bags[i];
                    if (x.get(index, splitFeature, Double.NaN) <= splitValue) {
                        trueBags.add(index);
                        tc++;
                    } else {
                        falseBags.add(index);
                    }
                }
            } else {
                throw new IllegalStateException("Unsupported attribute type: "
                        + node.splitFeatureType);
            }
            return tc;
        }

    }

    public RegressionTree(@Nullable Attribute[] attributes, @Nonnull Matrix x, @Nonnull double[] y,
            int maxLeafs) {
        this(attributes, x, y, x.numColumns(), Integer.MAX_VALUE, maxLeafs, 5, 1, null, null, null);
    }

    public RegressionTree(@Nullable Attribute[] attributes, @Nonnull Matrix x, @Nonnull double[] y,
            int maxLeafs, @Nullable PRNG rand) {
        this(attributes, x, y, x.numColumns(), Integer.MAX_VALUE, maxLeafs, 5, 1, null, null, rand);
    }

    public RegressionTree(@Nullable Attribute[] attributes, @Nonnull Matrix x, @Nonnull double[] y,
            int numVars, int maxDepth, int maxLeafs, int minSplits, int minLeafSize,
            @Nullable ColumnMajorIntMatrix order, @Nullable int[] bags, @Nullable PRNG rand) {
        this(attributes, x, y, numVars, maxDepth, maxLeafs, minSplits, minLeafSize, order, bags, null, rand);
    }

    /**
     * Constructor. Learns a regression tree for gradient tree boosting.
     * 
     * @param attributes the attribute properties.
     * @param x the training instances.
     * @param y the response variable.
     * @param numVars the number of input variables to pick to split on at each node. It seems that
     *        dim/3 give generally good performance, where dim is the number of variables.
     * @param maxLeafs the maximum number of leaf nodes in the tree.
     * @param minSplits number of instances in a node below which the tree will not split, setting S
     *        = 5 generally gives good results.
     * @param order the index of training values in ascending order. Note that only numeric
     *        attributes need be sorted.
     * @param bags the sample set of instances for stochastic learning.
     * @param output An interface to calculate node output.
     */
    public RegressionTree(@Nullable Attribute[] attributes, @Nonnull Matrix x, @Nonnull double[] y,
            int numVars, int maxDepth, int maxLeafs, int minSplits, int minLeafSize,
            @Nullable ColumnMajorIntMatrix order, @Nullable int[] bags,
            @Nullable NodeOutput output, @Nullable PRNG rand) {
        checkArgument(x, y, numVars, maxDepth, maxLeafs, minSplits, minLeafSize);

        this._attributes = SmileExtUtils.attributeTypes(attributes, x);
        if (_attributes.length != x.numColumns()) {
            throw new IllegalArgumentException("-attrs option is invliad: "
                    + Arrays.toString(attributes));
        }
        this._hasNumericType = SmileExtUtils.containsNumericType(_attributes);

        this._numVars = numVars;
        this._maxDepth = maxDepth;
        this._minSplit = minSplits;
        this._minLeafSize = minLeafSize;
        this._order = (order == null) ? SmileExtUtils.sort(_attributes, x) : order;
        this._importance = new double[_attributes.length];
        this._rnd = (rand == null) ? RandomNumberGeneratorFactory.createPRNG() : rand;
        this._nodeOutput = output;

        int n = 0;
        double sum = 0.0;
        if (bags == null) {
            n = y.length;
            bags = new int[n];
            for (int i = 0; i < n; i++) {
                bags[i] = i;
                sum += y[i];
            }
        } else {
            n = bags.length;
            for (int i = 0; i < n; i++) {
                int index = bags[i];
                sum += y[index];
            }
        }

        this._root = new Node(sum / n);

        TrainNode trainRoot = new TrainNode(_root, x, y, bags, 1);
        if (maxLeafs == Integer.MAX_VALUE) {
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
            for (int leaves = 1; leaves < maxLeafs; leaves++) {
                // parent is the leaf to split
                TrainNode node = nextSplits.poll();
                if (node == null) {
                    break;
                }
                node.split(nextSplits); // Split the parent node into two children nodes
            }
        }

        if (output != null) {
            trainRoot.calculateOutput(output);
        }
    }

    private static void checkArgument(@Nonnull Matrix x, @Nonnull double[] y, int numVars,
            int maxDepth, int maxLeafs, int minSplits, int minLeafSize) {
        if (x.numRows() != y.length) {
            throw new IllegalArgumentException(String.format(
                "The sizes of X and Y don't match: %d != %d", x.numRows(), y.length));
        }
        if (numVars <= 0 || numVars > x.numColumns()) {
            throw new IllegalArgumentException(
                "Invalid number of variables to split on at a node of the tree: " + numVars);
        }
        if (maxDepth < 2) {
            throw new IllegalArgumentException("maxDepth should be greater than 1: " + maxDepth);
        }
        if (maxLeafs < 2) {
            throw new IllegalArgumentException("Invalid maximum leaves: " + maxLeafs);
        }
        if (minSplits < 2) {
            throw new IllegalArgumentException(
                "Invalid minimum number of samples required to split an internal node: "
                        + minSplits);
        }
        if (minLeafSize < 1) {
            throw new IllegalArgumentException("Invalid minimum size of leaf nodes: " + minLeafSize);
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
    public double[] importance() {
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

    public String predictJsCodegen(@Nonnull final String[] featureNames) {
        StringBuilder buf = new StringBuilder(1024);
        _root.exportJavascript(buf, featureNames, 0);
        return buf.toString();
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
