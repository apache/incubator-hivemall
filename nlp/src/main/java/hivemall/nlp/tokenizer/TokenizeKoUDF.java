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
package hivemall.nlp.tokenizer;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.IOUtils;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.Set;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ko.KoreanAnalyzer;
import org.apache.lucene.analysis.ko.KoreanPartOfSpeechStopFilter;
import org.apache.lucene.analysis.ko.KoreanTokenizer;
import org.apache.lucene.analysis.ko.KoreanTokenizer.DecompoundMode;
import org.apache.lucene.analysis.ko.POS;
import org.apache.lucene.analysis.ko.dict.UserDictionary;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

@Description(name = "tokenize_ko",
        value = "_FUNC_(String line [, const array<string> userDict, const string mode = \"discard\", const array<string> stopTags, boolean outputUnknownUnigrams])"
                + " - returns tokenized strings in array<string>",
        extended = "select tokenize_ko(\"소설 무궁화꽃이 피었습니다.\");\n" + "\n"
                + "> [\"소설\",\"무궁\",\"화\",\"꽃\",\"피\"]\n")
@UDFType(deterministic = true, stateful = false)
public final class TokenizeKoUDF extends GenericUDF {

    @Nullable
    private UserDictionary userDict;

    private DecompoundMode mode;
    private Set<POS.Tag> stopTags;
    private boolean outputUnknownUnigrams;

    private transient KoreanAnalyzer analyzer;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {
        final int arglen = arguments.length;
        if (arglen > 5) {
            throw new UDFArgumentException(
                "Invalid number of arguments for `tokenize_ko`: " + arglen);
        }

        this.userDict = (arglen >= 3) ? parseUserDict(arguments[1]) : null;
        this.mode = (arglen >= 3) ? parseDecompoundMode(arguments[2])
                : KoreanTokenizer.DEFAULT_DECOMPOUND;
        this.stopTags = (arglen >= 4) ? parseStopTags(arguments[3])
                : KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS;
        this.outputUnknownUnigrams = (arglen >= 5) && HiveUtils.getConstBoolean(arguments[4]);

        this.analyzer = null;

        return ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector);
    }

    @Override
    public List<Text> evaluate(DeferredObject[] arguments) throws HiveException {
        if (arguments.length == 0) {
            final Properties properties = new Properties();
            try {
                properties.load(this.getClass().getResourceAsStream("tokenizer.properties"));
            } catch (IOException e) {
                throw new HiveException("Failed to read tokenizer.properties");
            }
            return Collections.singletonList(
                new Text(properties.getProperty("tokenize_ko.version")));
        }

        if (analyzer == null) {
            this.analyzer = new KoreanAnalyzer(userDict, mode, stopTags, outputUnknownUnigrams);
        }

        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }

        String line = arg0.toString();

        final List<Text> tokens = new ArrayList<Text>(32);
        TokenStream stream = null;
        try {
            stream = analyzer.tokenStream("", line);
            if (stream != null) {
                analyzeTokens(stream, tokens);
            }
        } catch (IOException e) {
            IOUtils.closeQuietly(analyzer);
            throw new HiveException(e);
        } finally {
            IOUtils.closeQuietly(stream);
        }
        return tokens;
    }

    @Override
    public void close() throws IOException {
        IOUtils.closeQuietly(analyzer);
    }

    @Nullable
    private static UserDictionary parseUserDict(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (HiveUtils.isVoidOI(oi)) {
            return null;
        }
        final String[] array = HiveUtils.getConstStringArray(oi);
        if (array == null) {
            return null;
        }
        final int length = array.length;
        if (length == 0) {
            return null;
        }
        final StringBuilder builder = new StringBuilder();
        for (int i = 0; i < length; i++) {
            String row = array[i];
            if (row != null) {
                builder.append(row).append('\n');
            }
        }

        final Reader reader = new StringReader(builder.toString());
        try {
            return UserDictionary.open(reader); // return null if empty
        } catch (Throwable e) {
            throw new UDFArgumentException(
                "Failed to create user dictionary based on the given array<string>: "
                        + builder.toString());
        }
    }

    @Nonnull
    private static DecompoundMode parseDecompoundMode(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        String arg = HiveUtils.getConstString(oi);
        if (arg == null) {
            return KoreanTokenizer.DEFAULT_DECOMPOUND;
        }
        final DecompoundMode mode;
        try {
            mode = DecompoundMode.valueOf(arg.toUpperCase(Locale.ENGLISH));
        } catch (IllegalArgumentException e) {
            final StringBuilder sb = new StringBuilder();
            for (DecompoundMode v : DecompoundMode.values()) {
                sb.append(v.toString()).append(", ");
            }
            throw new UDFArgumentException(
                "Expected either " + sb.toString() + "but got an unexpected mode: " + arg);
        }
        return mode;
    }

    @Nonnull
    private static Set<POS.Tag> parseStopTags(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (HiveUtils.isVoidOI(oi)) {
            return KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS;
        }
        final String[] array = HiveUtils.getConstStringArray(oi);
        if (array == null) {
            return KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS;
        }
        final int length = array.length;
        if (length == 0) {
            return Collections.emptySet();
        }
        final Set<POS.Tag> stopTags = new HashSet<POS.Tag>(length);
        for (int i = 0; i < length; i++) {
            String s = array[i];
            if (s != null) {
                try {
                    stopTags.add(POS.resolveTag(s));
                } catch (IllegalArgumentException e) {
                    throw new UDFArgumentException(
                        "Unrecognized POS tag has been specified as a stop tag: " + e.getMessage());
                }
            }
        }
        return stopTags;
    }

    private static void analyzeTokens(@Nonnull TokenStream stream, @Nonnull List<Text> results)
            throws IOException {
        // instantiate an attribute placeholder once
        CharTermAttribute termAttr = stream.getAttribute(CharTermAttribute.class);
        stream.reset();

        while (stream.incrementToken()) {
            String term = termAttr.toString();
            results.add(new Text(term));
        }
    }

    @Override
    public String getDisplayString(String[] children) {
        return "tokenize_ko(" + Arrays.toString(children) + ')';
    }

}
