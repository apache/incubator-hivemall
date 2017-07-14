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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

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
import org.apache.lucene.analysis.cn.smart.SmartChineseAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.util.CharArraySet;

@Description(name = "tokenize_cn", value = "_FUNC_(String line [, const list<string> stopWords])"
        + " - returns tokenized strings in array<string>")
@UDFType(deterministic = true, stateful = false)
public final class SmartcnUDF extends GenericUDF {

    private String[] _stopWordsArray;

    private transient SmartChineseAnalyzer _analyzer;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {
        final int arglen = arguments.length;
        if (arglen < 1 || arglen > 2) {
            throw new UDFArgumentException("Invalid number of arguments for `tokenize_cn`: "
                    + arglen);
        }

        this._stopWordsArray = (arglen >= 2) ? HiveUtils.getConstStringArray(arguments[1]) : null;
        this._analyzer = null;

        return ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
    }

    @Override
    public List<Text> evaluate(DeferredObject[] arguments) throws HiveException {
        SmartChineseAnalyzer analyzer = _analyzer;
        if (analyzer == null) {
            CharArraySet stopwords = stopWords(_stopWordsArray);
            analyzer = new SmartChineseAnalyzer(stopwords);
            this._analyzer = analyzer;
        }

        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            return null;
        }
        String line = arg0.toString();

        final List<Text> results = new ArrayList<Text>(32);
        TokenStream stream = null;
        try {
            stream = analyzer.tokenStream("", line);
            if (stream != null) {
                analyzeTokens(stream, results);
            }
        } catch (IOException e) {
            IOUtils.closeQuietly(analyzer);
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
    private static CharArraySet stopWords(@Nonnull final String[] array)
            throws UDFArgumentException {
        if (array == null) {
            return SmartChineseAnalyzer.getDefaultStopSet();
        }
        if (array.length == 0) {
            return CharArraySet.EMPTY_SET;
        }
        CharArraySet results = new CharArraySet(Arrays.asList(array), true /* ignoreCase */);
        return results;
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
        return "tokenize_cn(" + Arrays.toString(children) + ')';
    }
}
