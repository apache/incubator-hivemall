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
package hivemall.ftvec.conv;

import hivemall.UDFWithOptions;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hashing.MurmurHash3;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.StringUtils;
import hivemall.utils.struct.Pair;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

// @formatter:off
@Description(name = "to_libsvm_format",
        value = "_FUNC_(array<string> feautres [, double/integer target, const string options])"
                + " - Returns a string representation of libsvm",
                extended = "Usage:\n" + 
                        " select to_libsvm_format(array('apple:3.4','orange:2.1'))\n" + 
                        " > 6284535:3.4 8104713:2.1\n" + 
                        " select to_libsvm_format(array('apple:3.4','orange:2.1'), '-features 10')\n" + 
                        " > 3:2.1 7:3.4\n" + 
                        " select to_libsvm_format(array('7:3.4','3:2.1'), 5.0)\n" + 
                        " > 5.0 3:2.1 7:3.4")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class ToLibSVMFormatUDF extends UDFWithOptions {

    private ListObjectInspector _featuresOI;
    @Nullable
    private PrimitiveObjectInspector _targetOI = null;
    private int _numFeatures = MurmurHash3.DEFAULT_NUM_FEATURES;
    private StringBuilder _buf;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("features", "num_features", true,
            "The number of features [default: 16777217 (2^24)]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull String optionValue) throws UDFArgumentException {
        CommandLine cl = parseOptions(optionValue);
        this._numFeatures = Primitives.parseInt(cl.getOptionValue("num_features"),
            MurmurHash3.DEFAULT_NUM_FEATURES);
        assumeTrue(_numFeatures > 0, "num_features must be greater than 0: " + _numFeatures);
        return cl;
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        assumeTrue(argOIs.length >= 1 || argOIs.length <= 3,
            "to_libsvm_format UDF takes 1~3 arguments");

        this._featuresOI = HiveUtils.asListOI(argOIs[0]);
        if (argOIs.length == 2) {
            ObjectInspector argOI1 = argOIs[1];
            if (HiveUtils.isNumberOI(argOI1)) {
                this._targetOI = HiveUtils.asNumberOI(argOI1);
            } else if (HiveUtils.isConstString(argOI1)) { // no target
                String opts = HiveUtils.getConstString(argOI1);
                processOptions(opts);
            } else {
                throw new UDFArgumentException(
                    "Unexpected argument type for 2nd argument: " + argOI1.getTypeName());
            }
        } else if (argOIs.length == 3) {
            this._targetOI = HiveUtils.asNumberOI(argOIs[1]);
            String opts = HiveUtils.getConstString(argOIs[2]);
            processOptions(opts);
        }

        this._buf = new StringBuilder();

        return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
    }

    @Nullable
    @Override
    public String evaluate(DeferredObject[] args) throws HiveException {
        final StringBuilder buf = this._buf;
        StringUtils.clear(buf);

        Object arg0 = args[0].get();
        if (arg0 == null) {
            return null;
        }

        final int featureSize = _featuresOI.getListLength(arg0);
        List<Pair<Integer, Double>> features = new ArrayList<>(featureSize);
        for (int i = 0; i < featureSize; i++) {
            Object e = _featuresOI.getListElement(arg0, i);
            if (e == null) {
                continue;
            }
            Pair<Integer, Double> fv = parse(e.toString(), _numFeatures);
            features.add(fv);
        }
        Collections.sort(features, comparator);

        if (_targetOI != null) {
            Object arg1 = args[1].get();
            if (arg1 == null) {
                throw new HiveException("Detected NULL for the 2nd argument");
            }
            if (HiveUtils.isIntegerOI(_targetOI)) {
                int label = HiveUtils.getInt(arg1, _targetOI);
                buf.append(label);
            } else {
                double label = HiveUtils.getDouble(arg1, _targetOI);
                buf.append(label);
            }
            buf.append(' ');
        }
        for (int i = 0, size = features.size(); i < size; i++) {
            if (i != 0) {
                buf.append(' ');
            }
            Pair<Integer, Double> fv = features.get(i);
            buf.append(fv.getKey().intValue());
            buf.append(':');
            buf.append(fv.getValue().doubleValue());
        }

        return buf.toString();
    }

    @Nonnull
    public static Pair<Integer, Double> parse(@Nonnull final String fv,
            @Nonnegative final int numFeatures) throws UDFArgumentException {
        final int headPos = fv.indexOf(':');
        if (headPos == -1) {
            if (NumberUtils.isDigits(fv)) {
                final int f;
                try {
                    f = Integer.parseInt(fv);
                } catch (NumberFormatException e) {
                    throw new UDFArgumentException("Invalid feature value: " + fv);
                }
                return new Pair<>(f, 1.d);
            } else {
                return new Pair<>(mhash(fv, numFeatures), 1.d);
            }
        } else {
            final int tailPos = fv.lastIndexOf(':');
            if (headPos != tailPos) {
                throw new UDFArgumentException("Unsupported feature format: " + fv);
            }
            String f = fv.substring(0, headPos);
            String v = fv.substring(headPos + 1);
            final double d;
            try {
                d = Double.parseDouble(v);
            } catch (NumberFormatException e) {
                throw new UDFArgumentException("Invalid feature value: " + fv);
            }
            if (NumberUtils.isDigits(f)) {
                final int i;
                try {
                    i = Integer.parseInt(f);
                } catch (NumberFormatException e) {
                    throw new UDFArgumentException("Invalid feature value: " + fv);
                }
                return new Pair<>(i, d);
            } else {
                return new Pair<>(mhash(f, numFeatures), d);
            }
        }
    }

    private static int mhash(@Nonnull final String word, final int numFeatures) {
        int r = MurmurHash3.murmurhash3_x86_32(word, 0, word.length(), 0x9747b28c) % numFeatures;
        if (r < 0) {
            r += numFeatures;
        }
        return r + 1;
    }

    private static final Comparator<Pair<Integer, Double>> comparator =
            new Comparator<Pair<Integer, Double>>() {
                @Override
                public int compare(Pair<Integer, Double> l, Pair<Integer, Double> r) {
                    return l.getKey().compareTo(r.getKey());
                }
            };

    @Override
    public String getDisplayString(String[] args) {
        return "to_libsvm_format( " + StringUtils.join(args, ',') + " )";
    }
}
