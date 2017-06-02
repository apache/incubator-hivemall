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
import hivemall.fm.Feature;
import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Description(
        name = "dimsum_mapper",
        value = "_FUNC_(array<string> row, map<int col_id, double norm> colNorms [, const string options]) "
                + "- Returns column-wise partial similarities")
public class DIMSUMMapperUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(DIMSUMMapperUDTF.class);

    protected ListObjectInspector rowOI;
    protected MapObjectInspector colNormsOI;

    @Nullable
    protected Feature[] probes;

    @Nonnull
    protected PRNG rnd;

    protected double threshold;
    protected double sqrtGamma;
    protected boolean symmetricOutput;

    protected Map<Integer, Double> colNorms;
    protected Map<Integer, Double> colProbs;

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

        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        double threshold = 0.5d;
        double gamma = Double.MAX_VALUE;
        boolean symmetricOutput = true;

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
        }

        this.threshold = threshold;
        this.sqrtGamma = Math.sqrt(gamma);
        this.symmetricOutput = symmetricOutput;

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentException(
                getClass().getSimpleName()
                        + " takes 2 or 3 arguments: array<string> x, map<long, double> colNorms "
                        + "[, CONSTANT STRING options]: "
                        + Arrays.toString(argOIs));
        }

        this.rowOI = HiveUtils.asListOI(argOIs[0]);
        HiveUtils.validateFeatureOI(rowOI.getListElementObjectInspector());

        this.colNormsOI = HiveUtils.asMapOI(argOIs[1]);

        processOptions(argOIs);

        this.rnd = RandomNumberGeneratorFactory.createPRNG(1001);
        this.colNorms = null;
        this.colProbs = null;

        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("j");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("k");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("b_jk");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

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

            if (sqrtGamma == Double.MAX_VALUE) { // set default value to `gamma` based on `threshold`
                this.sqrtGamma = Math.sqrt(10 * Math.log(numCols) / threshold);
            }

            this.colNorms = new HashMap<Integer, Double>(numCols);
            this.colProbs = new HashMap<Integer, Double>(numCols);
            Map<Object, Object> m = (Map<Object, Object>) colNormsOI.getMap(args[1]);
            for (Map.Entry<Object, Object> e : m.entrySet()) {
                int j = HiveUtils.asJavaInt(e.getKey());
                double norm = HiveUtils.asJavaDouble(e.getValue());

                colNorms.put(j, norm);

                double p = Math.min(1.d, sqrtGamma * norm);
                colProbs.put(j, p);
            }
        }

        final IntWritable jWritable = new IntWritable();
        final IntWritable kWritable = new IntWritable();
        final FloatWritable bWritable = new FloatWritable();

        final Object[] forwardObjs = new Object[3];
        forwardObjs[0] = jWritable;
        forwardObjs[1] = kWritable;
        forwardObjs[2] = bWritable;

        final int length = row.length;

        double[] rowScaledValues = new double[length];
        for (int i = 0; i < length; i++) {
            int j = row[i].getFeatureIndex();
            double norm = colNorms.get(j).doubleValue();
            rowScaledValues[i] = row[i].getValue() / Math.min(sqrtGamma, norm);
        }

        for (int ij = 0; ij < length; ij++) {
            int j = row[ij].getFeatureIndex();
            double jVal = rowScaledValues[ij];

            if (jVal != 0.d && rnd.nextDouble() < colProbs.get(j)) {
                for (int ik = ij + 1; ik < length; ik++) {
                    int k = row[ik].getFeatureIndex();
                    double kVal = rowScaledValues[ik];

                    if (kVal != 0.d && rnd.nextDouble() < colProbs.get(k)) {
                        // compute b_jk
                        bWritable.set((float) (jVal * kVal));

                        // (j, k); similarity matrix is symmetric
                        jWritable.set(j);
                        kWritable.set(k);
                        forward(forwardObjs);

                        if (symmetricOutput) {
                            // (k, j)
                            jWritable.set(k);
                            kWritable.set(j);
                            forward(forwardObjs);
                        }
                    }
                }
            }
        }
    }

    @Nullable
    protected Feature[] parseFeatures(@Nonnull final Object arg) throws HiveException {
        return Feature.parseFeatures(arg, rowOI, probes, true);
    }

    @Override
    public void close() throws HiveException {
        this.probes = null;
        this.colNorms = null;
        this.colProbs = null;
    }
}
