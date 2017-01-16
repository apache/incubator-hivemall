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
package hivemall.ftvec.pairing;

import hivemall.UDTFWithOptions;
import hivemall.model.FeatureValue;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hashing.HashFunction;
import hivemall.utils.lang.Preconditions;

import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

@Description(name = "feature_pairs",
        value = "_FUNC_(feature_vector in array<string>, [, const string options])"
                + " - Returns a relation <string i, string j, double xi, double xj>")
public final class FeaturePairsUDTF extends UDTFWithOptions {

    private Type _type;
    private RowProcessor _proc;

    public FeaturePairsUDTF() {}

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("kpa", false,
            "Generate feature pairs for Kernel-Expansion Passive Aggressive [default:true]");
        opts.addOption("ffm", false,
            "Generate feature pairs for Field-aware Factorization Machines [default:false]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        if (argOIs.length == 2) {
            String args = HiveUtils.getConstString(argOIs[1]);
            cl = parseOptions(args);

            Preconditions.checkArgument(cl.getOptions().length == 1, UDFArgumentException.class,
                "Only one option can be specified: " + cl.getArgList());

            if (cl.hasOption("kpa")) {
                this._type = Type.kpa;
            } else if (cl.hasOption("ffm")) {
                this._type = Type.ffm;
            } else {
                throw new UDFArgumentException("Unsupported option: " + cl.getArgList().get(0));
            }
        } else {
            throw new UDFArgumentException("MUST provide -kpa or -ffm in the option");
        }

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1 && argOIs.length != 2) {
            throw new UDFArgumentException("_FUNC_ takes 1 or 2 arguments");
        }
        processOptions(argOIs);

        ListObjectInspector fvOI = HiveUtils.asListOI(argOIs[0]);
        HiveUtils.validateFeatureOI(fvOI.getListElementObjectInspector());

        final List<String> fieldNames = new ArrayList<String>(4);
        final List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>(4);
        switch (_type) {
            case kpa: {
                this._proc = new KPAProcessor(fvOI);
                fieldNames.add("h");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
                fieldNames.add("hk");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
                fieldNames.add("xh");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
                fieldNames.add("xk");
                fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
                break;
            }
            case ffm: {
                throw new UDFArgumentException("-ffm is not supported yet");
                //break;
            }
            default:
                throw new UDFArgumentException("Illegal condition: " + _type);
        }
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        Object arg0 = args[0];
        if (arg0 == null) {
            return;
        }
        _proc.process(arg0);
    }

    public enum Type {
        kpa, ffm;
    }

    abstract class RowProcessor {

        @Nonnull
        protected final ListObjectInspector fvOI;

        RowProcessor(@Nonnull ListObjectInspector fvOI) {
            this.fvOI = fvOI;
        }

        void process(@Nonnull Object arg) throws HiveException {
            final int size = fvOI.getListLength(arg);
            if (size == 0) {
                return;
            }

            final List<FeatureValue> features = new ArrayList<FeatureValue>(size);
            for (int i = 0; i < size; i++) {
                Object f = fvOI.getListElement(arg, i);
                if (f == null) {
                    continue;
                }
                FeatureValue fv = FeatureValue.parse(f, true);
                features.add(fv);
            }

            process(features);
        }

        abstract void process(@Nonnull List<FeatureValue> features) throws HiveException;

    }

    final class KPAProcessor extends RowProcessor {

        @Nonnull
        private final IntWritable f0, f1;
        @Nonnull
        private final DoubleWritable f2, f3;
        @Nonnull
        private final Writable[] forward;

        KPAProcessor(@Nonnull ListObjectInspector fvOI) {
            super(fvOI);
            this.f0 = new IntWritable();
            this.f1 = new IntWritable();
            this.f2 = new DoubleWritable();
            this.f3 = new DoubleWritable();
            this.forward = new Writable[] {f0, null, null, null};
        }

        @Override
        void process(@Nonnull List<FeatureValue> features) throws HiveException {
            forward[0] = f0;
            f0.set(0);
            forward[1] = null;
            forward[2] = null;
            forward[3] = null;
            forward(forward); // forward h(f0)

            forward[2] = f2;
            for (int i = 0, len = features.size(); i < len; i++) {
                FeatureValue xi = features.get(i);
                int h = xi.getFeatureAsInt();
                double xh = xi.getValue();
                forward[0] = f0;
                f0.set(h);
                forward[1] = null;
                f2.set(xh);
                forward[3] = null;
                forward(forward); // forward h(f0), xh(f2)

                forward[0] = null;
                forward[1] = f1;
                forward[3] = f3;
                for (int j = i + 1; j < len; j++) {
                    FeatureValue xj = features.get(j);
                    int k = xj.getFeatureAsInt();
                    int hk = HashFunction.hash(h, k, true);
                    double xk = xj.getValue();
                    f1.set(hk);
                    f3.set(xk);
                    forward(forward);// forward hk(f1), xh(f2), xk(f3)
                }
            }
        }
    }


    @Override
    public void close() throws HiveException {
        // clean up to help GC
        this._proc = null;
    }

}
