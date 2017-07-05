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
package hivemall.smile.classification;

import static hivemall.smile.utils.SmileExtUtils.resolveFeatureName;
import static hivemall.smile.utils.SmileExtUtils.resolveName;
import hivemall.annotations.VisibleForTesting;
import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.ints.ColumnMajorIntMatrix;
import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.math.vector.DenseVector;
import hivemall.math.vector.SparseVector;
import hivemall.math.vector.Vector;
import hivemall.math.vector.VectorProcedure;
import hivemall.smile.data.Attribute;
import hivemall.smile.data.Attribute.AttributeType;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.lang.ObjectUtils;
import hivemall.utils.lang.mutable.MutableInt;
import hivemall.utils.sampling.IntReservoirSampler;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.Arrays;
import java.util.PriorityQueue;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.roaringbitmap.IntConsumer;
import org.roaringbitmap.RoaringBitmap;

import smile.classification.Classifier;
import smile.math.Math;

/**
 * Decision tree for classification. A decision tree can be learned by splitting the training set
 * into subsets based on an attribute value test. This process is repeated on each derived subset in
 * a recursive manner called recursive partitioning. The recursion is completed when the subset at a
 * node all has the same value of the target variable, or when splitting no longer adds value to the
 * predictions.
 * <p>
 * The algorithms that are used for constructing decision trees usually work top-down by choosing a
 * variable at each step that is the next best variable to use in splitting the set of items. "Best"
 * is defined by how well the variable splits the set into homogeneous subsets that have the same
 * value of the target variable. Different algorithms use different formulae for measuring "best".
 * Used by the CART algorithm, Gini impurity is a measure of how often a randomly chosen element
 * from the set would be incorrectly labeled if it were randomly labeled according to the
 * distribution of labels in the subset. Gini impurity can be computed by summing the probability of
 * each item being chosen times the probability of a mistake in categorizing that item. It reaches
 * its minimum (zero) when all cases in the node fall into a single target category. Information
 * gain is another popular measure, used by the ID3, C4.5 and C5.0 algorithms. Information gain is
 * based on the concept of entropy used in information theory. For categorical variables with
 * different number of levels, however, information gain are biased in favor of those attributes
 * with more levels. Instead, one may employ the information gain ratio, which solves the drawback
 * of information gain.
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
 */
public final class DecisionTree implements Classifier<Vector> {
    /**
     * The attributes of independent variable.
     */
    @Nonnull
    private final Attribute[] _attributes;
    private final boolean _hasNumericType;
    /**
     * Variable importance. Every time a split of a node is made on variable the (GINI, information
     * gain, etc.) impurity criterion for the two descendant nodes is less than the parent node.
     * Adding up the decreases for each individual variable over the tree gives a simple measure of
     * variable importance.
     */
    @Nonnull
    private final Vector _importance;
    /**
     * The root of the regression tree
     */
    @Nonnull
    private final Node _root;
    /**
     * The maximum number of the tree depth
     */
    private final int _maxDepth;
    /**
     * The splitting rule.
     */
    @Nonnull
    private final SplitRule _rule;
    /**
     * The number of classes.
     */
    private final int _k;
    /**
     * The number of input variables to be used to determine the decision at a node of the tree.
     */
    private final int _numVars;
    /**
     * The number of instances in a node below which the tree will not split.
     */
    private final int _minSplit;
    /**
     * The minimum number of samples in a leaf node
     */
    private final int _minLeafSize;
    /**
     * The index of training values in ascending order. Note that only numeric attributes will be
     * sorted.
     */
    @Nonnull
    private final ColumnMajorIntMatrix _order;

    @Nonnull
    private final PRNG _rnd;

    /**
     * The criterion to choose variable to split instances.
     */
    public static enum SplitRule {
        /**
         * Used by the CART algorithm, Gini impurity is a measure of how often a randomly chosen
         * element from the set would be incorrectly labeled if it were randomly labeled according
         * to the distribution of labels in the subset. Gini impurity can be computed by summing the
         * probability of each item being chosen times the probability of a mistake in categorizing
         * that item. It reaches its minimum (zero) when all cases in the node fall into a single
         * target category.
         */
        GINI,
        /**
         * Used by the ID3, C4.5 and C5.0 tree generation algorithms.
         */
        ENTROPY,
        /**
         * Classification error.
         */
        CLASSIFICATION_ERROR
    }

    /**
     * Classification tree node.
     */
    public static final class Node implements Externalizable {

        /**
         * Predicted class label for this node.
         */
        int output = -1;
        /**
         * Posteriori probability based on sample ratios in this node.
         */
        @Nullable
        double[] posteriori = null;
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
         * Reduction in splitting criterion.
         */
        double splitScore = 0.0;
        /**
         * Children node.
         */
        Node trueChild = null;
        /**
         * Children node.
         */
        Node falseChild = null;
        /**
         * Predicted output for children node.
         */
        int trueChildOutput = -1;
        /**
         * Predicted output for children node.
         */
        int falseChildOutput = -1;

        public Node() {}// for Externalizable

        public Node(int output, @Nonnull double[] posteriori) {
            this.output = output;
            this.posteriori = posteriori;
        }

        private boolean isLeaf() {
            return posteriori != null;
        }

        @VisibleForTesting
        public int predict(@Nonnull final double[] x) {
            return predict(new DenseVector(x));
        }

        /**
         * Evaluate the regression tree over an instance.
         */
        public int predict(@Nonnull final Vector x) {
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
        public void predict(@Nonnull final Vector x, @Nonnull final PredictionHandler handler) {
            if (trueChild == null && falseChild == null) {
                handler.handle(output, posteriori);
            } else {
                if (splitFeatureType == AttributeType.NOMINAL) {
                    if (x.get(splitFeature, Double.NaN) == splitValue) {
                        trueChild.predict(x, handler);
                    } else {
                        falseChild.predict(x, handler);
                    }
                } else if (splitFeatureType == AttributeType.NUMERIC) {
                    if (x.get(splitFeature, Double.NaN) <= splitValue) {
                        trueChild.predict(x, handler);
                    } else {
                        falseChild.predict(x, handler);
                    }
                } else {
                    throw new IllegalStateException("Unsupported attribute type: "
                            + splitFeatureType);
                }
            }
        }

        public void exportJavascript(@Nonnull final StringBuilder builder,
                @Nullable final String[] featureNames, @Nullable final String[] classNames,
                final int depth) {
            if (trueChild == null && falseChild == null) {
                indent(builder, depth);
                builder.append("").append(resolveName(output, classNames)).append(";\n");
            } else {
                indent(builder, depth);
                if (splitFeatureType == AttributeType.NOMINAL) {
                    if (featureNames == null) {
                        builder.append("if( x[")
                               .append(splitFeature)
                               .append("] == ")
                               .append(splitValue)
                               .append(" ) {\n");
                    } else {
                        builder.append("if( ")
                               .append(resolveFeatureName(splitFeature, featureNames))
                               .append(" == ")
                               .append(splitValue)
                               .append(" ) {\n");
                    }
                } else if (splitFeatureType == AttributeType.NUMERIC) {
                    if (featureNames == null) {
                        builder.append("if( x[")
                               .append(splitFeature)
                               .append("] <= ")
                               .append(splitValue)
                               .append(" ) {\n");
                    } else {
                        builder.append("if( ")
                               .append(resolveFeatureName(splitFeature, featureNames))
                               .append(" <= ")
                               .append(splitValue)
                               .append(" ) {\n");
                    }
                } else {
                    throw new IllegalStateException("Unsupported attribute type: "
                            + splitFeatureType);
                }
                trueChild.exportJavascript(builder, featureNames, classNames, depth + 1);
                indent(builder, depth);
                builder.append("} else  {\n");
                falseChild.exportJavascript(builder, featureNames, classNames, depth + 1);
                indent(builder, depth);
                builder.append("}\n");
            }
        }

        public void exportGraphviz(@Nonnull final StringBuilder builder,
                @Nullable final String[] featureNames, @Nullable final String[] classNames,
                @Nonnull final String outputName, @Nullable double[] colorBrew,
                final @Nonnull MutableInt nodeIdGenerator, final int parentNodeId) {
            final int myNodeId = nodeIdGenerator.getValue();

            if (trueChild == null && falseChild == null) {
                // fillcolor=h,s,v 
                // https://en.wikipedia.org/wiki/HSL_and_HSV
                // http://www.graphviz.org/doc/info/attrs.html#k:colorList
                String hsvColor = (colorBrew == null || output >= colorBrew.length) ? "#00000000"
                        : String.format("%.4f,1.000,1.000", colorBrew[output]);
                builder.append(String.format(
                    " %d [label=<%s = %s>, fillcolor=\"%s\", shape=ellipse];\n", myNodeId,
                    outputName, resolveName(output, classNames), hsvColor));

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
                trueChild.exportGraphviz(builder, featureNames, classNames, outputName, colorBrew,
                    nodeIdGenerator, myNodeId);
                nodeIdGenerator.addValue(1);
                falseChild.exportGraphviz(builder, featureNames, classNames, outputName, colorBrew,
                    nodeIdGenerator, myNodeId);
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

                out.writeInt(output);
                out.writeInt(posteriori.length);
                for (int i = 0; i < posteriori.length; i++) {
                    out.writeDouble(posteriori[i]);
                }
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

            if (in.readBoolean()) {//isLeaf
                this.output = in.readInt();

                final int size = in.readInt();
                final double[] posteriori = new double[size];
                for (int i = 0; i < size; i++) {
                    posteriori[i] = in.readDouble();
                }
                this.posteriori = posteriori;
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
     * Classification tree node for training purpose.
     */
    private final class TrainNode implements Comparable<TrainNode> {
        /**
         * The associated regression tree node.
         */
        final Node node;
        /**
         * Training dataset.
         */
        final Matrix x;
        /**
         * class labels.
         */
        final int[] y;

        int[] bags;

        final int depth;

        /**
         * Constructor.
         */
        public TrainNode(Node node, Matrix x, int[] y, int[] bags, int depth) {
            this.node = node;
            this.x = x;
            this.y = y;
            this.bags = bags;
            this.depth = depth;
        }

        @Override
        public int compareTo(TrainNode a) {
            return (int) Math.signum(a.node.splitScore - node.splitScore);
        }

        /**
         * Finds the best attribute to split on at the current node.
         * 
         * @return true if a split exists to reduce squared error, false otherwise.
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

            // Sample count in each class.
            final int[] count = new int[_k];
            final boolean pure = sampleCount(count);

            // Since all instances have same label, stop splitting.
            if (pure) {
                return false;
            }

            final double impurity = impurity(count, numSamples, _rule);

            final int[] samples = _hasNumericType ? SmileExtUtils.bagsToSamples(bags, x.numRows())
                    : null;
            final int[] falseCount = new int[_k];
            for (int varJ : variableIndex(x, bags)) {
                final Node split = findBestSplit(numSamples, count, falseCount, impurity, varJ,
                    samples);
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

        @Nonnull
        private int[] variableIndex(@Nonnull final Matrix x, @Nonnull final int[] bags) {
            final IntReservoirSampler sampler = new IntReservoirSampler(_numVars, _rnd.nextLong());
            if (x.isSparse()) {
                final RoaringBitmap cols = new RoaringBitmap();
                final VectorProcedure proc = new VectorProcedure() {
                    public void apply(final int col) {
                        cols.add(col);
                    }
                };
                for (final int row : bags) {
                    x.eachColumnIndexInRow(row, proc);
                }
                cols.forEach(new IntConsumer() {
                    public void accept(final int k) {
                        sampler.add(k);
                    }
                });
            } else {
                for (int i = 0, size = _attributes.length; i < size; i++) {
                    sampler.add(i);
                }
            }
            return sampler.getSample();
        }

        private boolean sampleCount(@Nonnull final int[] count) {
            int label = -1;
            boolean pure = true;
            for (int i = 0; i < bags.length; i++) {
                int index = bags[i];
                int y_i = y[index];
                count[y_i]++;

                if (label == -1) {
                    label = y_i;
                } else if (y_i != label) {
                    pure = false;
                }
            }
            return pure;
        }

        /**
         * Finds the best split cutoff for attribute j at the current node.
         * 
         * @param n the number instances in this node.
         * @param count the sample count in each class.
         * @param falseCount an array to store sample count in each class for false child node.
         * @param impurity the impurity of this node.
         * @param j the attribute index to split on.
         */
        private Node findBestSplit(final int n, final int[] count, final int[] falseCount,
                final double impurity, final int j, @Nullable final int[] samples) {
            final Node splitNode = new Node();

            if (_attributes[j].type == AttributeType.NOMINAL) {
                final int m = _attributes[j].getSize();
                final int[][] trueCount = new int[m][_k];

                for (int i = 0, size = bags.length; i < size; i++) {
                    int index = bags[i];
                    final double v = x.get(index, j, Double.NaN);
                    if (Double.isNaN(v)) {
                        continue;
                    }
                    int x_ij = (int) v;
                    trueCount[x_ij][y[index]]++;
                }

                for (int l = 0; l < m; l++) {
                    final int tc = Math.sum(trueCount[l]);
                    final int fc = n - tc;

                    // skip splitting this feature.
                    if (tc < _minSplit || fc < _minSplit) {
                        continue;
                    }

                    for (int q = 0; q < _k; q++) {
                        falseCount[q] = count[q] - trueCount[l][q];
                    }

                    final double gain = impurity - (double) tc / n
                            * impurity(trueCount[l], tc, _rule) - (double) fc / n
                            * impurity(falseCount, fc, _rule);

                    if (gain > splitNode.splitScore) {
                        // new best split
                        splitNode.splitFeature = j;
                        splitNode.splitFeatureType = AttributeType.NOMINAL;
                        splitNode.splitValue = l;
                        splitNode.splitScore = gain;
                        splitNode.trueChildOutput = Math.whichMax(trueCount[l]);
                        splitNode.falseChildOutput = Math.whichMax(falseCount);
                    }
                }
            } else if (_attributes[j].type == AttributeType.NUMERIC) {
                final int[] trueCount = new int[_k];

                _order.eachNonNullInColumn(j, new VectorProcedure() {
                    double prevx = Double.NaN;
                    int prevy = -1;

                    public void apply(final int row, final int i) {
                        final int sample = samples[i];
                        if (sample == 0) {
                            return;
                        }

                        final double x_ij = x.get(i, j, Double.NaN);
                        if (Double.isNaN(x_ij)) {
                            return;
                        }
                        final int y_i = y[i];

                        if (Double.isNaN(prevx) || x_ij == prevx || y_i == prevy) {
                            prevx = x_ij;
                            prevy = y_i;
                            trueCount[y_i] += sample;
                            return;
                        }

                        final int tc = Math.sum(trueCount);
                        final int fc = n - tc;

                        // skip splitting this feature.
                        if (tc < _minSplit || fc < _minSplit) {
                            prevx = x_ij;
                            prevy = y_i;
                            trueCount[y_i] += sample;
                            return;
                        }

                        for (int l = 0; l < _k; l++) {
                            falseCount[l] = count[l] - trueCount[l];
                        }

                        final double gain = impurity - (double) tc / n
                                * impurity(trueCount, tc, _rule) - (double) fc / n
                                * impurity(falseCount, fc, _rule);

                        if (gain > splitNode.splitScore) {
                            // new best split
                            splitNode.splitFeature = j;
                            splitNode.splitFeatureType = AttributeType.NUMERIC;
                            splitNode.splitValue = (x_ij + prevx) / 2.d;
                            splitNode.splitScore = gain;
                            splitNode.trueChildOutput = Math.whichMax(trueCount);
                            splitNode.falseChildOutput = Math.whichMax(falseCount);
                        }

                        prevx = x_ij;
                        prevy = y_i;
                        trueCount[y_i] += sample;
                    }//apply()                    
                });
            } else {
                throw new IllegalStateException("Unsupported attribute type: "
                        + _attributes[j].type);
            }

            return splitNode;
        }

        /**
         * Split the node into two children nodes. Returns true if split success.
         */
        public boolean split(@Nullable final PriorityQueue<TrainNode> nextSplits) {
            if (node.splitFeature < 0) {
                throw new IllegalStateException("Split a node with invalid feature.");
            }

            // split sample bags
            int childBagSize = (int) (bags.length * 0.4);
            IntArrayList trueBags = new IntArrayList(childBagSize);
            IntArrayList falseBags = new IntArrayList(childBagSize);
            double[] trueChildPosteriori = new double[_k];
            double[] falseChildPosteriori = new double[_k];
            int tc = splitSamples(trueBags, falseBags, trueChildPosteriori, falseChildPosteriori);
            int fc = bags.length - tc;
            this.bags = null; // help GC for recursive call

            if (tc < _minLeafSize || fc < _minLeafSize) {
                // set the node as leaf                
                node.splitFeature = -1;
                node.splitFeatureType = null;
                node.splitValue = Double.NaN;
                node.splitScore = 0.0;
                return false;
            }

            for (int i = 0; i < _k; i++) {
                trueChildPosteriori[i] /= tc;
                falseChildPosteriori[i] /= fc;
            }

            node.trueChild = new Node(node.trueChildOutput, trueChildPosteriori);
            TrainNode trueChild = new TrainNode(node.trueChild, x, y, trueBags.toArray(), depth + 1);
            trueBags = null; // help GC for recursive call
            if (tc >= _minSplit && trueChild.findBestSplit()) {
                if (nextSplits != null) {
                    nextSplits.add(trueChild);
                } else {
                    trueChild.split(null);
                }
            }

            node.falseChild = new Node(node.falseChildOutput, falseChildPosteriori);
            TrainNode falseChild = new TrainNode(node.falseChild, x, y, falseBags.toArray(),
                depth + 1);
            falseBags = null; // help GC for recursive call
            if (fc >= _minSplit && falseChild.findBestSplit()) {
                if (nextSplits != null) {
                    nextSplits.add(falseChild);
                } else {
                    falseChild.split(null);
                }
            }

            _importance.incr(node.splitFeature, node.splitScore);
            node.posteriori = null; // posteriori is not needed for non-leaf nodes

            return true;
        }

        /**
         * @param falseChildPosteriori
         * @param trueChildPosteriori
         * @return the number of true samples
         */
        private int splitSamples(@Nonnull final IntArrayList trueBags,
                @Nonnull final IntArrayList falseBags, @Nonnull final double[] trueChildPosteriori,
                @Nonnull final double[] falseChildPosteriori) {
            int tc = 0;
            if (node.splitFeatureType == AttributeType.NOMINAL) {
                final int splitFeature = node.splitFeature;
                final double splitValue = node.splitValue;
                for (int i = 0, size = bags.length; i < size; i++) {
                    final int index = bags[i];
                    if (x.get(index, splitFeature, Double.NaN) == splitValue) {
                        trueBags.add(index);
                        trueChildPosteriori[y[index]]++;
                        tc++;
                    } else {
                        falseBags.add(index);
                        falseChildPosteriori[y[index]]++;
                    }
                }
            } else if (node.splitFeatureType == AttributeType.NUMERIC) {
                final int splitFeature = node.splitFeature;
                final double splitValue = node.splitValue;
                for (int i = 0, size = bags.length; i < size; i++) {
                    final int index = bags[i];
                    if (x.get(index, splitFeature, Double.NaN) <= splitValue) {
                        trueBags.add(index);
                        trueChildPosteriori[y[index]]++;
                        tc++;
                    } else {
                        falseBags.add(index);
                        falseChildPosteriori[y[index]]++;
                    }
                }
            } else {
                throw new IllegalStateException("Unsupported attribute type: "
                        + node.splitFeatureType);
            }
            return tc;
        }

    }

    /**
     * Returns the impurity of a node.
     * 
     * @param count the sample count in each class.
     * @param n the number of samples in the node.
     * @param rule the rule for splitting a node.
     * @return the impurity of a node
     */
    private static double impurity(@Nonnull final int[] count, final int n,
            @Nonnull final SplitRule rule) {
        double impurity = 0.0;

        switch (rule) {
            case GINI: {
                impurity = 1.0;
                for (int i = 0; i < count.length; i++) {
                    final int count_i = count[i];
                    if (count_i > 0) {
                        double p = (double) count_i / n;
                        impurity -= p * p;
                    }
                }
                break;
            }
            case ENTROPY: {
                for (int i = 0; i < count.length; i++) {
                    final int count_i = count[i];
                    if (count_i > 0) {
                        double p = (double) count_i / n;
                        impurity -= p * Math.log2(p);
                    }
                }
                break;
            }
            case CLASSIFICATION_ERROR: {
                impurity = 0.d;
                for (int i = 0; i < count.length; i++) {
                    final int count_i = count[i];
                    if (count_i > 0) {
                        impurity = Math.max(impurity, (double) count_i / n);
                    }
                }
                impurity = Math.abs(1.d - impurity);
                break;
            }
        }

        return impurity;
    }

    public DecisionTree(@Nullable Attribute[] attributes, @Nonnull Matrix x, @Nonnull int[] y,
            int numLeafs) {
        this(attributes, x, y, x.numColumns(), Integer.MAX_VALUE, numLeafs, 2, 1, null, null, SplitRule.GINI, null);
    }

    public DecisionTree(@Nullable Attribute[] attributes, @Nullable Matrix x, @Nullable int[] y,
            int numLeafs, @Nullable PRNG rand) {
        this(attributes, x, y, x.numColumns(), Integer.MAX_VALUE, numLeafs, 2, 1, null, null, SplitRule.GINI, rand);
    }

    /**
     * Constructor. Learns a classification tree for random forest.
     *
     * @param attributes the attribute properties.
     * @param x the training instances.
     * @param y the response variable.
     * @param numVars the number of input variables to pick to split on at each node. It seems that
     *        dim/3 give generally good performance, where dim is the number of variables.
     * @param maxLeafs the maximum number of leaf nodes in the tree.
     * @param minSplits the number of minimum elements in a node to split
     * @param minLeafSize the minimum size of leaf nodes.
     * @param order the index of training values in ascending order. Note that only numeric
     *        attributes need be sorted.
     * @param bags the sample set of instances for stochastic learning.
     * @param rule the splitting rule.
     * @param seed
     */
    public DecisionTree(@Nullable Attribute[] attributes, @Nonnull Matrix x, @Nonnull int[] y,
            int numVars, int maxDepth, int maxLeafs, int minSplits, int minLeafSize,
            @Nullable int[] bags, @Nullable ColumnMajorIntMatrix order, @Nonnull SplitRule rule,
            @Nullable PRNG rand) {
        checkArgument(x, y, numVars, maxDepth, maxLeafs, minSplits, minLeafSize);

        this._k = Math.max(y) + 1;
        if (_k < 2) {
            throw new IllegalArgumentException("Only one class or negative class labels.");
        }

        this._attributes = SmileExtUtils.attributeTypes(attributes, x);
        if (attributes.length != x.numColumns()) {
            throw new IllegalArgumentException("-attrs option is invliad: "
                    + Arrays.toString(attributes));
        }
        this._hasNumericType = SmileExtUtils.containsNumericType(_attributes);

        this._numVars = numVars;
        this._maxDepth = maxDepth;
        this._minSplit = minSplits;
        this._minLeafSize = minLeafSize;
        this._rule = rule;
        this._order = (order == null) ? SmileExtUtils.sort(_attributes, x) : order;
        this._importance = x.isSparse() ? new SparseVector() : new DenseVector(_attributes.length);
        this._rnd = (rand == null) ? RandomNumberGeneratorFactory.createPRNG() : rand;

        final int n = y.length;
        final int[] count = new int[_k];
        if (bags == null) {
            bags = new int[n];
            for (int i = 0; i < n; i++) {
                bags[i] = i;
                count[y[i]]++;
            }
        } else {
            for (int i = 0, size = bags.length; i < size; i++) {
                int index = bags[i];
                count[y[index]]++;
            }
        }

        final double[] posteriori = new double[_k];
        for (int i = 0; i < _k; i++) {
            posteriori[i] = (double) count[i] / n;
        }
        this._root = new Node(Math.whichMax(count), posteriori);

        final TrainNode trainRoot = new TrainNode(_root, x, y, bags, 1);
        if (maxLeafs == Integer.MAX_VALUE) {
            if (trainRoot.findBestSplit()) {
                trainRoot.split(null);
            }
        } else {
            // Priority queue for best-first tree growing.
            final PriorityQueue<TrainNode> nextSplits = new PriorityQueue<TrainNode>();
            // Now add splits to the tree until max tree size is reached
            if (trainRoot.findBestSplit()) {
                nextSplits.add(trainRoot);
            }
            // Pop best leaf from priority queue, split it, and push
            // children nodes into the queue if possible.
            for (int leaves = 1; leaves < maxLeafs; leaves++) {
                // parent is the leaf to split
                TrainNode parent = nextSplits.poll();
                if (parent == null) {
                    break;
                }
                parent.split(nextSplits); // Split the parent node into two children nodes
            }
        }
    }

    @VisibleForTesting
    Node getRootNode() {
        return _root;
    }

    private static void checkArgument(@Nonnull Matrix x, @Nonnull int[] y, int numVars,
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
     * Returns the variable importance. Every time a split of a node is made on variable the (GINI,
     * information gain, etc.) impurity criterion for the two descendent nodes is less than the
     * parent node. Adding up the decreases for each individual variable over the tree gives a
     * simple measure of variable importance.
     *
     * @return the variable importance
     */
    @Nonnull
    public Vector importance() {
        return _importance;
    }

    @VisibleForTesting
    public int predict(@Nonnull final double[] x) {
        return predict(new DenseVector(x));
    }

    @Override
    public int predict(@Nonnull final Vector x) {
        return _root.predict(x);
    }

    /**
     * Predicts the class label of an instance and also calculate a posteriori probabilities. Not
     * supported.
     */
    public int predict(Vector x, double[] posteriori) {
        throw new UnsupportedOperationException("Not supported.");
    }

    public String predictJsCodegen(@Nonnull final String[] featureNames,
            @Nonnull final String[] classNames) {
        StringBuilder buf = new StringBuilder(1024);
        _root.exportJavascript(buf, featureNames, classNames, 0);
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
        return _root == null ? "" : predictJsCodegen(null, null);
    }

}
