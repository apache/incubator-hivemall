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
package hivemall.smile.tools;

import hivemall.UDFWithOptions;
import hivemall.smile.classification.DecisionTree;
import hivemall.smile.regression.RegressionTree;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.codec.Base91;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.mutable.MutableInt;

import java.util.Arrays;

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
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.Text;

@Description(name = "tree_export",
        value = "_FUNC_(string model, const string options, optional array<string> featureNames=null, optional array<string> classNames=null)"
                + " - exports a Decision Tree model as javascript/dot]")
@UDFType(deterministic = true, stateful = false)
public final class TreeExportUDF extends UDFWithOptions {

    private transient Evaluator evaluator;

    private transient StringObjectInspector modelOI;
    @Nullable
    private transient ListObjectInspector featureNamesOI;
    @Nullable
    private transient ListObjectInspector classNamesOI;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("t", "type", true,
            "Type of output [default: js, javascript/js, graphviz/dot");
        opts.addOption("r", "regression", false, "Is regression tree or not");
        opts.addOption("output_name", "outputName", true, "output name [default: predicted]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(@Nonnull String opts) throws UDFArgumentException {
        CommandLine cl = parseOptions(opts);

        OutputType outputType = OutputType.resolve(cl.getOptionValue("type"));
        boolean regression = cl.hasOption("regression");
        String outputName = cl.getOptionValue("output_name", "predicted");
        this.evaluator = new Evaluator(outputType, outputName, regression);

        return cl;
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int argLen = argOIs.length;
        if (argLen < 2 || argLen > 4) {
            showHelp("tree_export UDF takes 2~4 arguments: " + argLen);
        }

        this.modelOI = HiveUtils.asStringOI(argOIs, 0);

        String options = HiveUtils.getConstString(argOIs, 1);
        processOptions(options);

        if (argLen >= 3) {
            this.featureNamesOI = HiveUtils.asListOI(argOIs, 2);
            if (!HiveUtils.isStringOI(featureNamesOI.getListElementObjectInspector())) {
                throw new UDFArgumentException("_FUNC_ expected array<string> for featureNames: "
                        + featureNamesOI.getTypeName());
            }
            if (argLen == 4) {
                this.classNamesOI = HiveUtils.asListOI(argOIs, 3);
                if (!HiveUtils.isStringOI(classNamesOI.getListElementObjectInspector())) {
                    throw new UDFArgumentException("_FUNC_ expected array<string> for classNames: "
                            + classNamesOI.getTypeName());
                }
            }
        }

        return PrimitiveObjectInspectorFactory.writableStringObjectInspector;
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }
        Text model = modelOI.getPrimitiveWritableObject(arg0);

        String[] featureNames = null, classNames = null;
        if (arguments.length >= 3) {
            featureNames = HiveUtils.asStringArray(arguments[2], featureNamesOI);
            if (arguments.length >= 4) {
                classNames = HiveUtils.asStringArray(arguments[3], classNamesOI);
            }
        }

        try {
            return evaluator.export(model, featureNames, classNames);
        } catch (HiveException he) {
            throw he;
        } catch (Throwable e) {
            throw new HiveException(e);
        }
    }

    @Override
    public String getDisplayString(String[] children) {
        return "tree_export(" + Arrays.toString(children) + ")";
    }

    public enum OutputType {
        javascript, graphviz;

        @Nonnull
        public static OutputType resolve(@Nonnull String name) throws UDFArgumentException {
            if ("js".equalsIgnoreCase(name) || "javascript".equalsIgnoreCase(name)) {
                return javascript;
            } else if ("dot".equalsIgnoreCase(name) || "graphviz".equalsIgnoreCase(name)
                    || "graphvis".equalsIgnoreCase(name)) { // "graphvis" for backward compatibility (HIVEMALL-192)
                return graphviz;
            } else {
                throw new UDFArgumentException(
                    "Please provide a valid `-type` option from [javascript, graphviz]: " + name);
            }
        }
    }

    public static class Evaluator {

        @Nonnull
        private final OutputType outputType;
        @Nonnull
        private final String outputName;
        private final boolean regression;

        public Evaluator(@Nonnull OutputType outputType, @Nonnull String outputName,
                boolean regression) {
            this.outputType = outputType;
            this.outputName = outputName;
            this.regression = regression;
        }

        @Nonnull
        public Text export(@Nonnull Text model, @Nullable String[] featureNames,
                @Nullable String[] classNames) throws HiveException {
            int length = model.getLength();
            byte[] b = model.getBytes();
            b = Base91.decode(b, 0, length);

            final String exported;
            if (regression) {
                exported = exportRegressor(b, featureNames);
            } else {
                exported = exportClassifier(b, featureNames, classNames);
            }
            return new Text(exported);
        }

        @Nonnull
        private String exportClassifier(@Nonnull byte[] b, @Nullable String[] featureNames,
                @Nullable String[] classNames) throws HiveException {
            final DecisionTree.Node node = DecisionTree.deserialize(b, b.length, true);

            final StringBuilder buf = new StringBuilder(8192);
            switch (outputType) {
                case javascript: {
                    node.exportJavascript(buf, featureNames, classNames, 0);
                    break;
                }
                case graphviz: {
                    buf.append(
                        "digraph Tree {\n node [shape=box, style=\"filled, rounded\", color=\"black\", fontname=helvetica];\n edge [fontname=helvetica];\n");
                    double[] colorBrew = (classNames == null) ? null
                            : SmileExtUtils.getColorBrew(classNames.length);
                    node.exportGraphviz(buf, featureNames, classNames, outputName, colorBrew,
                        new MutableInt(0), 0);
                    buf.append("}");
                    break;
                }
                default:
                    throw new HiveException("Unsupported outputType: " + outputType);
            }
            return buf.toString();
        }

        @Nonnull
        private String exportRegressor(@Nonnull byte[] b, @Nullable String[] featureNames)
                throws HiveException {
            final RegressionTree.Node node = RegressionTree.deserialize(b, b.length, true);

            final StringBuilder buf = new StringBuilder(8192);
            switch (outputType) {
                case javascript: {
                    node.exportJavascript(buf, featureNames, 0);
                    break;
                }
                case graphviz: {
                    buf.append(
                        "digraph Tree {\n node [shape=box, style=\"filled, rounded\", color=\"black\", fontname=helvetica];\n edge [fontname=helvetica];\n");
                    node.exportGraphviz(buf, featureNames, outputName, new MutableInt(0), 0);
                    buf.append("}");
                    break;
                }
                default:
                    throw new HiveException("Unsupported outputType: " + outputType);
            }
            return buf.toString();
        }

    }

}
