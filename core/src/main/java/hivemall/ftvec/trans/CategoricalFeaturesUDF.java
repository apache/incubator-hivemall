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
package hivemall.ftvec.trans;

import hivemall.UDFWithOptions;
import hivemall.utils.hadoop.HiveUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

@Description(
        name = "categorical_features",
        value = "_FUNC_(array<string> featureNames, feature1, feature2, .. [, const string options])"
                + " - Returns a feature vector array<string>")
@UDFType(deterministic = true, stateful = false)
public final class CategoricalFeaturesUDF extends UDFWithOptions {

    private String[] _featureNames;
    private PrimitiveObjectInspector[] _inputOIs;
    private List<String> _result;

    private boolean _emitNull = false;
    private boolean _forceValue = false;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("no_elim", "no_elimination", false,
            "Wheather to emit NULL and value [default: false]");
        opts.addOption("emit_null", false, "Wheather to emit NULL [default: false]");
        opts.addOption("force_value", false, "Wheather to force emit value [default: false]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull String optionValue) throws UDFArgumentException {
        CommandLine cl = parseOptions(optionValue);
        if (cl.hasOption("no_elim")) {
            this._emitNull = true;
            this._forceValue = true;
        } else {
            this._emitNull = cl.hasOption("emit_null");
            this._forceValue = cl.hasOption("force_value");
        }
        return cl;
    }

    @Override
    public ObjectInspector initialize(@Nonnull final ObjectInspector[] argOIs)
            throws UDFArgumentException {
        final int numArgOIs = argOIs.length;
        if (numArgOIs < 2) {
            throw new UDFArgumentException("argOIs.length must be greater that or equals to 2: "
                    + numArgOIs);
        }

        this._featureNames = HiveUtils.getConstStringArray(argOIs[0]);
        if (_featureNames == null) {
            throw new UDFArgumentException("#featureNames should not be null");
        }
        int numFeatureNames = _featureNames.length;
        if (numFeatureNames < 1) {
            throw new UDFArgumentException("#featureNames must be greater than or equals to 1: "
                    + numFeatureNames);
        }
        for (String featureName : _featureNames) {
            if (featureName == null) {
                throw new UDFArgumentException("featureName should not be null: "
                        + Arrays.toString(_featureNames));
            } else if (featureName.indexOf(':') != -1) {
                throw new UDFArgumentException("featureName should not include colon: "
                        + featureName);
            }
        }

        final int numFeatures;
        final int lastArgIndex = numArgOIs - 1;
        if (lastArgIndex > numFeatureNames) {
            if (lastArgIndex == (numFeatureNames + 1)
                    && HiveUtils.isConstString(argOIs[lastArgIndex])) {
                String optionValue = HiveUtils.getConstString(argOIs[lastArgIndex]);
                processOptions(optionValue);
                numFeatures = numArgOIs - 2;
            } else {
                throw new UDFArgumentException(
                    "Unexpected arguments for _FUNC_"
                            + "(const array<string> featureNames, feature1, feature2, .. [, const string options])");
            }
        } else {
            numFeatures = lastArgIndex;
        }
        if (numFeatureNames != numFeatures) {
            throw new UDFArgumentLengthException("#featureNames '" + numFeatureNames
                    + "' != #features '" + numFeatures + "'");
        }

        this._inputOIs = new PrimitiveObjectInspector[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            ObjectInspector oi = argOIs[i + 1];
            _inputOIs[i] = HiveUtils.asPrimitiveObjectInspector(oi);
        }
        this._result = new ArrayList<String>(numFeatures);

        return ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
    }

    @Override
    public List<String> evaluate(@Nonnull final DeferredObject[] arguments) throws HiveException {
        _result.clear();

        final int size = _featureNames.length;
        for (int i = 0; i < size; i++) {
            Object argument = arguments[i + 1].get();
            if (argument == null) {
                if (_emitNull) {
                    _result.add(null);
                }
                continue;
            }

            PrimitiveObjectInspector oi = _inputOIs[i];
            String s = PrimitiveObjectInspectorUtils.getString(argument, oi);
            if (s.isEmpty()) {
                if (_emitNull) {
                    _result.add(null);
                }
                continue;
            }

            // categorical feature representation   
            final String f;
            if (_forceValue) {
                f = _featureNames[i] + '#' + s + ":1";
            } else {
                f = _featureNames[i] + '#' + s;
            }
            _result.add(f);

        }
        return _result;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "categorical_features(" + Arrays.toString(children) + ")";
    }

}
