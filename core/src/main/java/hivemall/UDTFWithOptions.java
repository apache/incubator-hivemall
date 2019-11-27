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
package hivemall;

import hivemall.model.FeatureValue;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.CommandLineUtils;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils.ObjectInspectorCopyOption;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapred.Counters.Counter;
import org.apache.hadoop.mapred.Reporter;

public abstract class UDTFWithOptions extends GenericUDTF {

    @Nullable
    protected MapredContext mapredContext;

    @Override
    public final void configure(MapredContext mapredContext) {
        this.mapredContext = mapredContext;
    }

    @Nullable
    protected final Reporter getReporter() {
        if (mapredContext == null) {
            return null;
        }
        return mapredContext.getReporter();
    }

    protected static void reportProgress(@Nullable Reporter reporter) {
        if (reporter != null) {
            synchronized (reporter) {
                reporter.progress();
            }
        }
    }

    protected static void setCounterValue(@Nullable Counter counter, long value) {
        if (counter != null) {
            synchronized (counter) {
                counter.setValue(value);
            }
        }
    }

    protected static void incrCounter(@Nullable Counter counter, long incr) {
        if (counter != null) {
            synchronized (counter) {
                counter.increment(incr);
            }
        }
    }

    protected abstract Options getOptions();

    @Nonnull
    protected final CommandLine parseOptions(String optionValue) throws UDFArgumentException {
        String[] args = optionValue.split("\\s+");
        Options opts = getOptions();
        opts.addOption("help", false, "Show function help");

        final CommandLine cl;
        try {
            cl = CommandLineUtils.parseOptions(args, opts);
        } catch (IllegalArgumentException e) {
            throw new UDFArgumentException(e);
        }
        if (cl.hasOption("help")) {
            showHelp(opts);
        }

        return cl;
    }

    protected void showHelp(@Nullable String errMsg) throws UDFArgumentException {
        showHelp(getOptions(), errMsg);
    }

    private void showHelp(@Nonnull Options opts) throws UDFArgumentException {
        showHelp(opts, null);
    }

    private void showHelp(@Nonnull Options opts, @Nullable String errMsg)
            throws UDFArgumentException {
        Description funcDesc = getClass().getAnnotation(Description.class);
        final String cmdLineSyntax;
        if (funcDesc == null) {
            cmdLineSyntax = getClass().getSimpleName();
        } else {
            String funcName = funcDesc.name();
            cmdLineSyntax = funcName == null ? getClass().getSimpleName()
                    : funcDesc.value().replace("_FUNC_", funcDesc.name());
        }
        StringWriter sw = new StringWriter();
        sw.write('\n');
        if (errMsg != null) {
            if (funcDesc != null && funcDesc.name() != null) {
                errMsg = errMsg.replace("_FUNC_", funcDesc.name());
            }
            sw.write(errMsg);
            sw.write("\n\n");
        }
        PrintWriter pw = new PrintWriter(sw);
        HelpFormatter formatter = new HelpFormatter();
        formatter.printHelp(pw, HelpFormatter.DEFAULT_WIDTH, cmdLineSyntax, null, opts,
            HelpFormatter.DEFAULT_LEFT_PAD, HelpFormatter.DEFAULT_DESC_PAD, null, true);
        pw.flush();
        String helpMsg = sw.toString();
        throw new UDFArgumentException(helpMsg);
    }

    protected abstract CommandLine processOptions(ObjectInspector[] argOIs)
            throws UDFArgumentException;

    @Nonnull
    protected final List<FeatureValue> parseFeatures(@Nonnull final List<?> features,
            @Nonnull final ObjectInspector featureInspector, final boolean parseFeature) {
        final int numFeatures = features.size();
        if (numFeatures == 0) {
            return Collections.emptyList();
        }
        final List<FeatureValue> list = new ArrayList<FeatureValue>(numFeatures);
        for (Object f : features) {
            if (f == null) {
                continue;
            }
            final FeatureValue fv;
            if (parseFeature) {
                fv = FeatureValue.parse(f);
            } else {
                Object o = ObjectInspectorUtils.copyToStandardObject(f, featureInspector,
                    ObjectInspectorCopyOption.WRITABLE);
                Writable k = WritableUtils.toWritable(o);
                fv = new FeatureValue(k, 1.f);
            }
            list.add(fv);
        }
        return list;
    }

    protected static <T> T checkNotNull(@CheckForNull final T arg, @Nonnegative final String errMsg)
            throws UDFArgumentException {
        if (arg == null) {
            throw new UDFArgumentException(errMsg);
        }
        return arg;
    }

    protected static <T> T checkNotNull(@CheckForNull final T arg, @Nonnegative final int index)
            throws UDFArgumentException {
        if (arg == null) {
            throw new UDFArgumentException(String.format("%d-th argument MUST not be null", index));
        }
        return arg;
    }

    protected static Object nonNullArgument(@Nonnull final Object[] args,
            @Nonnegative final int index) throws UDFArgumentException {
        final Object arg = args[index];
        if (arg == null) {
            throw new UDFArgumentException(String.format("%d-th argument MUST not be null", index));
        }
        return arg;
    }

    /**
     * Raise {@link UDFArgumentException} if the given condition is false.
     *
     * @throws UDFArgumentException
     */
    protected static void assumeTrue(final boolean condition, @Nonnull final String errMsg)
            throws UDFArgumentException {
        if (!condition) {
            throw new UDFArgumentException(errMsg);
        }
    }

    /**
     * Raise {@link UDFArgumentException} if the given condition is true.
     *
     * @throws UDFArgumentException
     */
    protected static void assumeFalse(final boolean condition, @Nonnull final String errMsg)
            throws UDFArgumentException {
        if (condition) {
            throw new UDFArgumentException(errMsg);
        }
    }

}
