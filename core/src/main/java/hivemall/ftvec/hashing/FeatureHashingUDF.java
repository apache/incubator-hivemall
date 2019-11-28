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
package hivemall.ftvec.hashing;

import hivemall.HivemallConstants;
import hivemall.UDFWithOptions;
import hivemall.annotations.VisibleForTesting;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hashing.MurmurHash3;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.StringUtils;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

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
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

//@formatter:off
@Description(name = "feature_hashing",
        value = "_FUNC_(array<string> features [, const string options])"
                + " - returns a hashed feature vector in array<string>",
        extended = "select feature_hashing(array('aaa:1.0','aaa','bbb:2.0'), '-libsvm');\n" + 
                " [\"4063537:1.0\",\"4063537:1\",\"8459207:2.0\"]\n" + 
                "\n" + 
                "select feature_hashing(array('aaa:1.0','aaa','bbb:2.0'), '-features 10');\n" + 
                " [\"7:1.0\",\"7\",\"1:2.0\"]\n" + 
                "\n" + 
                "select feature_hashing(array('aaa:1.0','aaa','bbb:2.0'), '-features 10 -libsvm');\n" + 
                " [\"1:2.0\",\"7:1.0\",\"7:1\"]\n" + 
                "")
//@formatter:on
@UDFType(deterministic = true, stateful = false)
public final class FeatureHashingUDF extends UDFWithOptions {

    private static final IndexComparator indexCmp = new IndexComparator();

    @Nullable
    private ListObjectInspector _listOI;
    private boolean _libsvmFormat = false;
    private int _numFeatures = MurmurHash3.DEFAULT_NUM_FEATURES;

    @Nullable
    private transient List<String> _returnObj;

    public FeatureHashingUDF() {}

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("libsvm", false,
            "Returns in libsvm format (<index>:<value>)* sorted by index ascending order");
        opts.addOption("features", "num_features", true,
            "The number of features [default: 16777217 (2^24)]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull String optionValue) throws UDFArgumentException {
        CommandLine cl = parseOptions(optionValue);

        this._libsvmFormat = cl.hasOption("libsvm");
        this._numFeatures = Primitives.parseInt(cl.getOptionValue("num_features"), _numFeatures);
        return cl;
    }

    @Override
    public ObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        if (argOIs.length != 1 && argOIs.length != 2) {
            showHelp("The feature_hashing function takes 1 or 2 arguments: " + argOIs.length);
        }
        ObjectInspector argOI0 = argOIs[0];
        this._listOI = HiveUtils.isListOI(argOI0) ? (ListObjectInspector) argOI0 : null;

        if (argOIs.length == 2) {
            String opts = HiveUtils.getConstString(argOIs[1]);
            processOptions(opts);
        }

        if (_listOI == null) {
            return PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        } else {
            return ObjectInspectorFactory.getStandardListObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector);
        }
    }

    @Override
    public Object evaluate(@Nonnull DeferredObject[] arguments) throws HiveException {
        final Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }

        if (_listOI == null) {
            return evaluateScalar(arg0);
        } else {
            return evaluateList(arg0);
        }
    }

    @Nonnull
    private String evaluateScalar(@Nonnull final Object arg0) {
        String fv = arg0.toString();
        return featureHashing(fv, _numFeatures, _libsvmFormat);
    }

    @Nonnull
    private List<String> evaluateList(@Nonnull final Object arg0) throws HiveException {
        final int len = _listOI.getListLength(arg0);
        List<String> list = _returnObj;
        if (list == null) {
            list = new ArrayList<String>(len);
            this._returnObj = list;
        } else {
            list.clear();
        }

        final int numFeatures = _numFeatures;
        for (int i = 0; i < len; i++) {
            Object obj = _listOI.getListElement(arg0, i);
            if (obj == null) {
                continue;
            }
            String fv = featureHashing(obj.toString(), numFeatures, _libsvmFormat);
            list.add(fv);
        }

        if (_libsvmFormat) {
            try {
                Collections.sort(list, indexCmp);
            } catch (NumberFormatException e) {
                throw new HiveException(e);
            }
        }
        return list;
    }

    @VisibleForTesting
    @Nonnull
    static String featureHashing(@Nonnull final String fv, final int numFeatures) {
        return featureHashing(fv, numFeatures, false);
    }

    @Nonnull
    static String featureHashing(@Nonnull final String fv, final int numFeatures,
            final boolean libsvmFormat) {
        final int headPos = fv.indexOf(':');
        if (headPos == -1) {
            if (fv.equals(HivemallConstants.BIAS_CLAUSE)) {
                return fv;
            }
            final int h = mhash(fv, numFeatures);
            if (libsvmFormat) {
                return h + ":1";
            } else {
                return String.valueOf(h);
            }
        } else {
            final int tailPos = fv.lastIndexOf(':');
            if (headPos == tailPos) {
                String f = fv.substring(0, headPos);
                String tail = fv.substring(headPos);
                if (f.equals(HivemallConstants.BIAS_CLAUSE)) {
                    String v = fv.substring(headPos + 1);
                    double d = Double.parseDouble(v);
                    if (d == 1.d) {
                        return fv;
                    }
                }
                int h = mhash(f, numFeatures);
                return h + tail;
            } else {
                String field = fv.substring(0, headPos + 1);
                String f = fv.substring(headPos + 1, tailPos);
                int h = mhash(f, numFeatures);
                String v = fv.substring(tailPos);
                return field + h + v;
            }
        }
    }

    static int mhash(@Nonnull final String word, final int numFeatures) {
        int r = MurmurHash3.murmurhash3_x86_32(word, 0, word.length(), 0x9747b28c) % numFeatures;
        if (r < 0) {
            r += numFeatures;
        }
        return r + 1;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "feature_hashing(" + StringUtils.join(children, ',') + ')';
    }

    private static final class IndexComparator implements Comparator<String>, Serializable {
        private static final long serialVersionUID = -260142385860586255L;

        @Override
        public int compare(@Nonnull final String lhs, @Nonnull final String rhs) {
            int l = getIndex(lhs);
            int r = getIndex(rhs);
            return Integer.compare(l, r);
        }

        private static int getIndex(@Nonnull final String fv) {
            final int headPos = fv.indexOf(':');
            final int tailPos = fv.lastIndexOf(':');
            final String f;
            if (headPos == tailPos) {
                f = fv.substring(0, headPos);
            } else {
                f = fv.substring(headPos + 1, tailPos);
            }
            return Integer.parseInt(f);
        }

    }

}
