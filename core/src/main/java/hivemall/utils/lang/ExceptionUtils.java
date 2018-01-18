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
package hivemall.utils.lang;

import javax.annotation.Nonnull;

public final class ExceptionUtils {

    public static final int TRACE_CAUSE_DEPTH = 5;

    private ExceptionUtils() {}

    @Nonnull
    public static String prettyPrintStackTrace(@Nonnull final Throwable throwable) {
        return prettyPrintStackTrace(throwable, TRACE_CAUSE_DEPTH);
    }

    @Nonnull
    public static String prettyPrintStackTrace(@Nonnull final Throwable throwable,
            final int traceDepth) {
        final StringBuilder out = new StringBuilder(512);
        out.append(getMessage(throwable));
        out.append("\n\n---- Debugging information ----");
        final int tracedepth;
        if (throwable instanceof RuntimeException || throwable instanceof Error) {
            tracedepth = -1;
        } else {
            tracedepth = traceDepth;
        }
        String captured = captureThrownWithStrackTrace(throwable, "trace-exception", tracedepth);
        out.append(captured);
        final Throwable cause = throwable.getCause();
        if (cause != null) {
            final Throwable rootCause = getRootCause(cause);
            captured = captureThrownWithStrackTrace(rootCause, "trace-cause", TRACE_CAUSE_DEPTH);
            out.append(captured);
        }
        out.append("\n------------------------------- \n");
        return out.toString();
    }

    @Nonnull
    private static String captureThrownWithStrackTrace(@Nonnull final Throwable throwable,
            final String label, final int traceDepth) {
        assert (traceDepth >= 1 || traceDepth == -1);
        final StringBuilder out = new StringBuilder(255);
        final String clazz = throwable.getClass().getName();
        out.append(String.format("\n%-20s: %s \n", ("* " + label), clazz));
        final StackTraceElement[] st = throwable.getStackTrace();
        int at;
        final int limit = (traceDepth == -1) ? st.length - 1 : traceDepth;
        for (at = 0; at < st.length; at++) {
            if (at < limit) {
                out.append("\tat " + st[at] + '\n');
            } else {
                out.append("\t...\n");
                break;
            }
        }
        if (st.length == 0) {
            out.append("\t no stack traces...");
        } else if (at != (st.length - 1)) {
            out.append("\tat " + st[st.length - 1]);
        }
        String errmsg = throwable.getMessage();
        if (errmsg != null) {
            out.append(String.format("\n%-20s: \n", ("* " + label + "-error-msg")));
            String[] line = errmsg.split("\n");
            final int maxlines = Math.min(line.length, Math.max(1, TRACE_CAUSE_DEPTH - 2));
            for (int i = 0; i < maxlines; i++) {
                out.append('\t');
                out.append(line[i]);
                if (i != (maxlines - 1)) {
                    out.append('\n');
                }
            }
        }
        return out.toString();
    }

    @Nonnull
    public static String getMessage(@Nonnull final Throwable throwable) {
        String errMsg = throwable.getMessage();
        String clazz = throwable.getClass().getName();
        return (errMsg != null) ? clazz + ": " + errMsg : clazz;
    }

    @Nonnull
    private static Throwable getRootCause(@Nonnull final Throwable throwable) {
        Throwable top = throwable;
        while (top != null) {
            Throwable parent = top.getCause();
            if (parent != null) {
                top = parent;
            } else {
                break;
            }
        }
        return top;
    }

}
