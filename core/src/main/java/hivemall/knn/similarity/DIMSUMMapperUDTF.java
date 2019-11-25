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
package hivemall.knn.similarity;

import hivemall.UDTFWithOptions;
import hivemall.factorization.fm.Feature;
import hivemall.factorization.fm.IntFeature;
import hivemall.factorization.fm.StringFeature;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.random.PRNG;
import hivemall.utils.random.RandomNumberGeneratorFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

@Description(name = "dimsum_mapper",
        value = "_FUNC_(array<string> row, map<int col_id, double norm> colNorms [, const string options]) "
                + "- Returns column-wise partial similarities")
public final class DIMSUMMapperUDTF extends UDTFWithOptions {

    private ListObjectInspector rowOI;
    private MapObjectInspector colNormsOI;

    @Nullable
    private Feature[] probes;

    @Nonnull
    private PRNG rnd;

    private double threshold;
    private double sqrtGamma;
    private boolean symmetricOutput;
    private boolean parseFeatureAsInt;

    private Map<Object, Double> colNorms;
    private Map<Object, Double> colProbs;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("th", "threshold", true,
            "Theoretically, similarities above this threshold are estimated [default: 0.5]");
        opts.addOption("g", "gamma", true,
            "Oversampling parameter; if `gamma` is given, `threshold` will be ignored"
                    + " [default: 10 * log(numCols) / threshold]");
        opts.addOption("disable_symmetric", "disable_symmetric_output", false,
            "Output only contains (col j, col k) pair; symmetric (col k, col j) pair is omitted");
        opts.addOption("int_feature", "feature_as_integer", false,
            "Parse a feature (i.e. column ID) as integer");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        double threshold = 0.5d;
        double gamma = Double.POSITIVE_INFINITY;
        boolean symmetricOutput = true;
        boolean parseFeatureAsInt = false;

        CommandLine cl = null;
        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = parseOptions(rawArgs);
            threshold = Primitives.parseDouble(cl.getOptionValue("threshold"), threshold);
            if (threshold < 0.f || threshold >= 1.f) {
                throw new UDFArgumentException("`threshold` MUST be in range [0,1): " + threshold);
            }
            gamma = Primitives.parseDouble(cl.getOptionValue("gamma"), gamma);
            if (gamma <= 1.d) {
                throw new UDFArgumentException("`gamma` MUST be greater than 1: " + gamma);
            }
            symmetricOutput = !cl.hasOption("disable_symmetric_output");
            parseFeatureAsInt = cl.hasOption("feature_as_integer");
        }

        this.threshold = threshold;
        this.sqrtGamma = Math.sqrt(gamma);
        this.symmetricOutput = symmetricOutput;
        this.parseFeatureAsInt = parseFeatureAsInt;

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takes 2 or 3 arguments: array<string> x, map<long, double> colNorms "
                    + "[, CONSTANT STRING options]: " + Arrays.toString(argOIs));
        }

        this.rowOI = HiveUtils.asListOI(argOIs[0]);
        HiveUtils.validateFeatureOI(rowOI.getListElementObjectInspector());

        this.colNormsOI = HiveUtils.asMapOI(argOIs[1]);

        processOptions(argOIs);

        this.rnd = RandomNumberGeneratorFactory.createPRNG(1001);
        this.colNorms = null;
        this.colProbs = null;

        ArrayList<String> fieldNames = new ArrayList<String>();
        fieldNames.add("j");
        fieldNames.add("k");
        fieldNames.add("b_jk");

        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
        if (parseFeatureAsInt) {
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        } else {
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        }
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @SuppressWarnings("unchecked")
    @Override
    public void process(Object[] args) throws HiveException {
        Feature[] row = parseFeatures(args[0]);
        if (row == null) {
            return;
        }
        this.probes = row;

        // since the 2nd argument (column norms) is consistent,
        // column-related values, `colNorms` and `colProbs`, should be cached
        if (colNorms == null || colProbs == null) {
            final int numCols = colNormsOI.getMapSize(args[1]);

            if (sqrtGamma == Double.POSITIVE_INFINITY) { // set default value to `gamma` based on `threshold`
                if (threshold > 0.d) { // if `threshold` = 0, `gamma` is INFINITY i.e. always accept <j, k> pairs
                    this.sqrtGamma = Math.sqrt(10 * Math.log(numCols) / threshold);
                }
            }

            this.colNorms = new HashMap<Object, Double>(numCols);
            this.colProbs = new HashMap<Object, Double>(numCols);
            final Map<Object, Object> m = (Map<Object, Object>) colNormsOI.getMap(args[1]);
            for (Map.Entry<Object, Object> e : m.entrySet()) {
                Object j = e.getKey();
                if (parseFeatureAsInt) {
                    j = HiveUtils.asJavaInt(j);
                } else {
                    j = j.toString();
                }

                double norm = HiveUtils.asJavaDouble(e.getValue());
                if (norm == 0.d) { // avoid zero-division
                    norm = 1.d;
                }

                colNorms.put(j, norm);

                double p = Math.min(1.d, sqrtGamma / norm);
                colProbs.put(j, p);
            }
        }

        if (parseFeatureAsInt) {
            forwardAsIntFeature(row);
        } else {
            forwardAsStringFeature(row);
        }
    }

    private void forwardAsIntFeature(@Nonnull Feature[] row) throws HiveException {
        final int length = row.length;

        Feature[] rowScaled = new Feature[length];
        for (int i = 0; i < length; i++) {
            int j = row[i].getFeatureIndex();

            double norm = Primitives.doubleValue(colNorms.get(j), 0.d);
            if (norm == 0.d) { // avoid zero-division
                norm = 1.d;
            }
            double scaled = row[i].getValue() / Math.min(sqrtGamma, norm);

            rowScaled[i] = new IntFeature(j, scaled);
        }

        final IntWritable jWritable = new IntWritable();
        final IntWritable kWritable = new IntWritable();
        final DoubleWritable bWritable = new DoubleWritable();

        final Object[] forwardObjs = new Object[3];
        forwardObjs[0] = jWritable;
        forwardObjs[1] = kWritable;
        forwardObjs[2] = bWritable;

        for (int ij = 0; ij < length; ij++) {
            int j = rowScaled[ij].getFeatureIndex();
            double jVal = rowScaled[ij].getValue();
            double jProb = Primitives.doubleValue(colProbs.get(j), 0.d);

            if (jVal != 0.d && rnd.nextDouble() < jProb) {
                for (int ik = ij + 1; ik < length; ik++) {
                    int k = rowScaled[ik].getFeatureIndex();
                    double kVal = rowScaled[ik].getValue();
                    double kProb = Primitives.doubleValue(colProbs.get(k), 0.d);

                    if (kVal != 0.d && rnd.nextDouble() < kProb) {
                        // compute b_jk
                        bWritable.set(jVal * kVal);

                        if (symmetricOutput) {
                            // (j, k); similarity matrix is symmetric
                            jWritable.set(j);
                            kWritable.set(k);
                            forward(forwardObjs);

                            // (k, j)
                            jWritable.set(k);
                            kWritable.set(j);
                            forward(forwardObjs);
                        } else {
                            if (j < k) {
                                jWritable.set(j);
                                kWritable.set(k);
                            } else {
                                jWritable.set(k);
                                kWritable.set(j);
                            }
                            forward(forwardObjs);
                        }
                    }
                }
            }
        }
    }

    private void forwardAsStringFeature(@Nonnull Feature[] row) throws HiveException {
        final int length = row.length;

        Feature[] rowScaled = new Feature[length];
        for (int i = 0; i < length; i++) {
            String j = row[i].getFeature();

            double norm = Primitives.doubleValue(colNorms.get(j), 0.d);
            if (norm == 0.d) { // avoid zero-division
                norm = 1.d;
            }
            double scaled = row[i].getValue() / Math.min(sqrtGamma, norm);

            rowScaled[i] = new StringFeature(j, scaled);
        }

        final Text jWritable = new Text();
        final Text kWritable = new Text();
        final DoubleWritable bWritable = new DoubleWritable();

        final Object[] forwardObjs = new Object[3];
        forwardObjs[0] = jWritable;
        forwardObjs[1] = kWritable;
        forwardObjs[2] = bWritable;

        for (int ij = 0; ij < length; ij++) {
            String j = rowScaled[ij].getFeature();
            double jVal = rowScaled[ij].getValue();
            double jProb = Primitives.doubleValue(colProbs.get(j), 0.d);

            if (jVal != 0.d && rnd.nextDouble() < jProb) {
                for (int ik = ij + 1; ik < length; ik++) {
                    String k = rowScaled[ik].getFeature();
                    double kVal = rowScaled[ik].getValue();
                    double kProb = Primitives.doubleValue(colProbs.get(j), 0.d);

                    if (kVal != 0.d && rnd.nextDouble() < kProb) {
                        // compute b_jk
                        bWritable.set(jVal * kVal);

                        if (symmetricOutput) {
                            // (j, k); similarity matrix is symmetric
                            jWritable.set(j);
                            kWritable.set(k);
                            forward(forwardObjs);

                            // (k, j)
                            jWritable.set(k);
                            kWritable.set(j);
                            forward(forwardObjs);
                        } else {
                            if (j.compareTo(k) < 0) {
                                jWritable.set(j);
                                kWritable.set(k);
                            } else {
                                jWritable.set(k);
                                kWritable.set(j);
                            }
                            forward(forwardObjs);
                        }
                    }
                }
            }
        }
    }

    @Nullable
    protected Feature[] parseFeatures(@Nonnull final Object arg) throws HiveException {
        return Feature.parseFeatures(arg, rowOI, probes, parseFeatureAsInt);
    }

    @Override
    public void close() throws HiveException {
        this.probes = null;
        this.colNorms = null;
        this.colProbs = null;
    }
}
