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
import org.apache.lucene.analysis.ja.JapaneseAnalyzer;
import org.apache.lucene.analysis.ja.JapaneseTokenizer;
import org.apache.lucene.analysis.ja.JapaneseTokenizer.Mode;
import org.apache.lucene.analysis.ja.dict.UserDictionary;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.CharArraySet;

@Description(
        name = "tokenize_ja",
        value = "_FUNC_(String line [, const string mode = \"normal\", const array<string> stopWords, const array<string> stopTags, const array<string> userDict (or string userDictURL)])"
                + " - returns tokenized strings in array<string>",
        extended = "select tokenize_ja(\"kuromojiを使った分かち書きのテストです。第二引数にはnormal/search/extendedを指定できます。デフォルトではnormalモードです。\");\n"
                + "\n"
                + "> [\"kuromoji\",\"使う\",\"分かち書き\",\"テスト\",\"第\",\"二\",\"引数\",\"normal\",\"search\",\"extended\",\"指定\",\"デフォルト\",\"normal\",\" モード\"]\n")
@UDFType(deterministic = true, stateful = false)
public final class KuromojiUDF extends GenericUDF {
    private static final int CONNECT_TIMEOUT_MS = 10000; // 10 sec
    private static final int READ_TIMEOUT_MS = 60000; // 60 sec
    private static final long MAX_INPUT_STREAM_SIZE = 32L * 1024L * 1024L; // ~32MB

    private String _modeString;
    private String[] _stopWordsArray;
    private String[] _stopTagsArray;
    private Object _userDictObj;

    // workaround to avoid org.apache.hive.com.esotericsoftware.kryo.KryoException: java.util.ConcurrentModificationException
    private transient JapaneseAnalyzer _analyzer;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {
        final int arglen = arguments.length;
        if (arglen < 1 || arglen > 5) {
            throw new UDFArgumentException("Invalid number of arguments for `tokenize_ja`: "
                    + arglen);
        }

        this._modeString = (arglen >= 2) ? HiveUtils.getConstString(arguments[1]) : "NORMAL";

        this._stopWordsArray = null;
        if (arglen >= 3 && !HiveUtils.isVoidOI(arguments[2])) {
            this._stopWordsArray = HiveUtils.getConstStringArray(arguments[2]);
        }

        this._stopTagsArray = null;
        if (arglen >= 4 && !HiveUtils.isVoidOI(arguments[3])) {
            this._stopTagsArray = HiveUtils.getConstStringArray(arguments[3]);
        }

        this._userDictObj = null;
        if (arglen >= 5) {
            if (HiveUtils.isConstListOI(arguments[4])) {
                this._userDictObj = HiveUtils.getConstStringArray(arguments[4]);
            } else if (HiveUtils.isConstString(arguments[4])) {
                this._userDictObj = HiveUtils.getConstString(arguments[4]);
            } else {
                throw new UDFArgumentException(
                    "User dictionary MUST be given as an array of constant string or constant string (URL)");
            }
        }

        this._analyzer = null;

        return ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
    }

    @Override
    public List<Text> evaluate(DeferredObject[] arguments) throws HiveException {
        if (_analyzer == null) {
            Mode mode = tokenizationMode(_modeString);
            CharArraySet stopWords = stopWords(_stopWordsArray);
            Set<String> stopTags = stopTags(_stopTagsArray);

            UserDictionary userDict = null;
            if (_userDictObj instanceof String[]) {
                userDict = userDictionary((String[]) _userDictObj);
            } else if (_userDictObj instanceof String) {
                userDict = userDictionary((String) _userDictObj);
            }

            this._analyzer = new JapaneseAnalyzer(userDict, mode, stopWords, stopTags);
        }

        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }
        String line = arg0.toString();

        final List<Text> results = new ArrayList<Text>(32);
        TokenStream stream = null;
        try {
            stream = _analyzer.tokenStream("", line);
            if (stream != null) {
                analyzeTokens(stream, results);
            }
        } catch (IOException e) {
            IOUtils.closeQuietly(_analyzer);
            throw new HiveException(e);
        } finally {
            IOUtils.closeQuietly(stream);
        }
        return results;
    }

    @Override
    public void close() throws IOException {
        IOUtils.closeQuietly(_analyzer);
    }

    @Nonnull
    private static Mode tokenizationMode(@Nullable final String arg) throws UDFArgumentException {
        if (arg == null) {
            return Mode.NORMAL;
        }
        final Mode mode;
        if ("NORMAL".equalsIgnoreCase(arg)) {
            mode = Mode.NORMAL;
        } else if ("SEARCH".equalsIgnoreCase(arg)) {
            mode = Mode.SEARCH;
        } else if ("EXTENDED".equalsIgnoreCase(arg)) {
            mode = Mode.EXTENDED;
        } else if ("DEFAULT".equalsIgnoreCase(arg)) {
            mode = JapaneseTokenizer.DEFAULT_MODE;
        } else {
            throw new UDFArgumentException(
                "Expected NORMAL|SEARCH|EXTENDED|DEFAULT but got an unexpected mode: " + arg);
        }
        return mode;
    }

    @Nonnull
    private static CharArraySet stopWords(@Nullable final String[] array)
            throws UDFArgumentException {
        if (array == null) {
            return JapaneseAnalyzer.getDefaultStopSet();
        }
        if (array.length == 0) {
            return CharArraySet.EMPTY_SET;
        }
        CharArraySet results = new CharArraySet(Arrays.asList(array), /* ignoreCase */true);
        return results;
    }

    @Nonnull
    private static Set<String> stopTags(@Nullable final String[] array) throws UDFArgumentException {
        if (array == null) {
            return JapaneseAnalyzer.getDefaultStopTags();
        }
        final int length = array.length;
        if (length == 0) {
            return Collections.emptySet();
        }
        final Set<String> results = new HashSet<String>(length);
        for (int i = 0; i < length; i++) {
            String s = array[i];
            if (s != null) {
                results.add(s);
            }
        }
        return results;
    }

    @Nullable
    private static UserDictionary userDictionary(@Nullable final String[] userDictArray)
            throws UDFArgumentException {
        if (userDictArray == null) {
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
            is = IOUtils.decodeInputStream(HttpUtils.getLimitedInputStream(conn,
                MAX_INPUT_STREAM_SIZE));
        } catch (NullPointerException | IOException e) {
            throw new UDFArgumentException("Failed to get input stream from the connection: "
                    + userDictURL + '\n' + ExceptionUtils.prettyPrintStackTrace(e));
        }

        CharsetDecoder decoder = StandardCharsets.UTF_8.newDecoder()
                                                       .onMalformedInput(CodingErrorAction.REPORT)
                                                       .onUnmappableCharacter(
                                                           CodingErrorAction.REPORT);
        final Reader reader = new InputStreamReader(is, decoder);
        try {
            return UserDictionary.open(reader); // return null if empty
        } catch (Throwable e) {
            throw new UDFArgumentException(
                "Failed to parse the file in CSV format (UTF-8 encoding is expected): "
                        + userDictURL + '\n' + ExceptionUtils.prettyPrintStackTrace(e));
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
        return "tokenize_ja(" + Arrays.toString(children) + ')';
    }

}
