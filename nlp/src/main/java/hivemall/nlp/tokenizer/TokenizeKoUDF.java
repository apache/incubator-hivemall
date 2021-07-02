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

import hivemall.UDFWithOptions;
import hivemall.nlp.tokenizer.ext.KoreanAnalyzer;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.HttpUtils;
import hivemall.utils.io.IOUtils;
import hivemall.utils.lang.ExceptionUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.io.StringReader;
import java.net.HttpURLConnection;
import java.nio.charset.CharsetDecoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
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

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ko.KoreanTokenizer;
import org.apache.lucene.analysis.ko.KoreanTokenizer.DecompoundMode;
import org.apache.lucene.analysis.ko.POS;
import org.apache.lucene.analysis.ko.dict.UserDictionary;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

@Description(name = "tokenize_ko",
        value = "_FUNC_(String line [, const string mode = \"discard\" (or const string opts), const array<string> stopWords, const array<string> stopTags, const array<string> userDict (or const string userDictURL)])"
                + " - returns tokenized strings in array<string>",
        extended = "select tokenize_ko(\"소설 무궁화꽃이 피었습니다.\");\n" + "\n"
                + "> [\"소설\",\"무궁\",\"화\",\"꽃\",\"피\"]\n")
@UDFType(deterministic = true, stateful = false)
public final class TokenizeKoUDF extends UDFWithOptions {
    private static final int CONNECT_TIMEOUT_MS = 10000; // 10 sec
    private static final int READ_TIMEOUT_MS = 60000; // 60 sec
    private static final long MAX_INPUT_STREAM_SIZE = 32L * 1024L * 1024L; // ~32MB

    private DecompoundMode mode;
    @Nullable
    private String[] stopWordsArray;
    private Set<POS.Tag> stopTags;
    private boolean outputUnknownUnigrams = false;

    @Nullable
    private Object userDictObj; // String[] or String

    private transient KoreanAnalyzer analyzer;

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("mode", true,
            "The tokenization mode. One of ['node', 'discard' (default), 'mixed']");
        opts.addOption("outputUnknownUnigrams", false, "outputs unigrams for unknown words.");
        return opts;
    }

    @Override
    protected CommandLine processOptions(String optionValue) throws UDFArgumentException {
        CommandLine cl = parseOptions(optionValue);
        if (cl.hasOption("mode")) {
            String modeStr = cl.getOptionValue("mode");
            this.mode = decompoundMode(modeStr);
        }
        this.outputUnknownUnigrams = cl.hasOption("outputUnknownUnigrams");
        return cl;
    }

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {
        final int arglen = arguments.length;
        if (arglen > 6) {
            showHelp("Invalid number of arguments for `tokenize_ko`: " + arglen);
        }

        this.mode = KoreanTokenizer.DEFAULT_DECOMPOUND;
        if (arglen >= 2) {
            String arg1 = HiveUtils.getConstString(arguments[1]);
            if (arg1 != null) {
                if (arg1.startsWith("-")) {
                    processOptions(arg1);
                } else {
                    this.mode = decompoundMode(arg1);
                }
            }
        }

        if (arglen >= 3 && !HiveUtils.isVoidOI(arguments[2])) {
            this.stopWordsArray = HiveUtils.getConstStringArray(arguments[2]);
        }

        this.stopTags =
                (arglen >= 4) ? stopTags(arguments[3]) : KoreanAnalyzer.getDefaultStopTags();

        if (arglen >= 5) {
            if (HiveUtils.isConstListOI(arguments[4])) {
                this.userDictObj = HiveUtils.getConstStringArray(arguments[4]);
            } else if (HiveUtils.isConstString(arguments[4])) {
                this.userDictObj = HiveUtils.getConstString(arguments[4]);
            } else {
                throw new UDFArgumentException(
                    "User dictionary MUST be given as an array of constant string or constant string (URL)");
            }
        }

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
            CharArraySet stopWords = stopWords(stopWordsArray);

            UserDictionary userDict = null;
            if (userDictObj instanceof String[]) {
                userDict = userDictionary((String[]) userDictObj);
            } else if (userDictObj instanceof String) {
                userDict = userDictionary((String) userDictObj);
            }

            this.analyzer =
                    new KoreanAnalyzer(userDict, mode, stopWords, stopTags, outputUnknownUnigrams);
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

    @Nonnull
    private static DecompoundMode decompoundMode(@Nullable final String arg)
            throws UDFArgumentException {
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
    private static CharArraySet stopWords(@Nullable final String[] array)
            throws UDFArgumentException {
        final CharArraySet stopWords = KoreanAnalyzer.getDefaultStopSet();
        if (array == null) {
            return stopWords;
        }
        if (array.length == 0) {
            return CharArraySet.EMPTY_SET;
        }
        stopWords.addAll(Arrays.asList(array));
        return stopWords;
    }

    @Nonnull
    private static Set<POS.Tag> stopTags(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (HiveUtils.isVoidOI(oi)) {
            return KoreanAnalyzer.getDefaultStopTags();
        }
        final String[] array = HiveUtils.getConstStringArray(oi);
        if (array == null) {
            return KoreanAnalyzer.getDefaultStopTags();
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

    @Nullable
    private static UserDictionary userDictionary(@Nullable final String[] userDictArray)
            throws UDFArgumentException {
        if (userDictArray == null) {
            return null;
        }
        if (userDictArray.length == 0) {
            return null;
        }

        final StringBuilder builder = new StringBuilder();
        for (String row : userDictArray) {
            builder.append(row).append('\n');
        }
        final Reader reader = new StringReader(builder.toString());
        try {
            return UserDictionary.open(reader); // return null if empty
        } catch (Throwable e) {
            throw new UDFArgumentException(
                "Failed to create user dictionary based on the given array<string>: "
                        + builder.toString() + '\n' + ExceptionUtils.prettyPrintStackTrace(e));
        }
    }

    @Nullable
    private static UserDictionary userDictionary(@Nonnull final ObjectInspector oi)
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


    @Nullable
    private static UserDictionary userDictionary(@Nullable final String userDictURL)
            throws UDFArgumentException {
        if (userDictURL == null) {
            return null;
        }

        final HttpURLConnection conn;
        try {
            conn = HttpUtils.getHttpURLConnection(userDictURL);
        } catch (IllegalArgumentException | IOException e) {
            throw new UDFArgumentException("Failed to create HTTP connection to the URL: "
                    + userDictURL + '\n' + ExceptionUtils.prettyPrintStackTrace(e));
        }

        // allow to read as a compressed GZIP file for efficiency
        conn.setRequestProperty("Accept-Encoding", "gzip");

        conn.setConnectTimeout(CONNECT_TIMEOUT_MS); // throw exception from connect()
        conn.setReadTimeout(READ_TIMEOUT_MS); // throw exception from getXXX() methods

        final int responseCode;
        try {
            responseCode = conn.getResponseCode();
        } catch (IOException e) {
            throw new UDFArgumentException("Failed to get response code: " + userDictURL + '\n'
                    + ExceptionUtils.prettyPrintStackTrace(e));
        }
        if (responseCode != 200) {
            throw new UDFArgumentException("Got invalid response code: " + responseCode);
        }

        final InputStream is;
        try {
            is = IOUtils.decodeInputStream(
                HttpUtils.getLimitedInputStream(conn, MAX_INPUT_STREAM_SIZE));
        } catch (NullPointerException | IOException e) {
            throw new UDFArgumentException("Failed to get input stream from the connection: "
                    + userDictURL + '\n' + ExceptionUtils.prettyPrintStackTrace(e));
        }

        CharsetDecoder decoder =
                StandardCharsets.UTF_8.newDecoder()
                                      .onMalformedInput(CodingErrorAction.REPORT)
                                      .onUnmappableCharacter(CodingErrorAction.REPORT);
        final Reader reader = new InputStreamReader(is, decoder);
        try {
            return UserDictionary.open(reader); // return null if empty
        } catch (Throwable e) {
            throw new UDFArgumentException(
                "Failed to parse the dictionary CSV file: " + userDictURL + '\n'
                + "Please ensure that \n"
                + "  1) file encoding is UTF-8, \n"
                + "  2) no duplicate entry.\"\n"
                + "  3) the maximum dictionary size is limited to 32MB (SHOULD be compressed using gzip with .gz suffix)\n"
                + "  4) read timeout is set to 60 sec and connection must be established in 10 sec.\n"
                        +  ExceptionUtils.prettyPrintStackTrace(e));
        }
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
