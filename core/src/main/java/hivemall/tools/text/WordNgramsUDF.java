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
package hivemall.tools.text;

import hivemall.utils.lang.StringUtils;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.Text;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import java.util.ArrayList;
import java.util.List;

@Description(name = "word_ngrams",
        value = "_FUNC_(array<string> words, int minSize, int maxSize])"
                + " - Returns list of n-grams for given words, where `minSize <= n <= maxSize`",
        extended = "SELECT word_ngrams(tokenize('Machine learning is fun!', true), 1, 2);\n" + "\n"
                + " [\"machine\",\"machine learning\",\"learning\",\"learning is\",\"is\",\"is fun\",\"fun\"]")
@UDFType(deterministic = true, stateful = false)
public final class WordNgramsUDF extends UDF {

    @Nullable
    public List<Text> evaluate(@Nullable final List<Text> words, final int minSize,
            final int maxSize) throws HiveException {
        if (words == null) {
            return null;
        }
        if (minSize <= 0) {
            throw new UDFArgumentException("`minSize` must be greater than zero: " + minSize);
        }
        if (minSize > maxSize) {
            throw new UDFArgumentException(
                "`maxSize` must be greater than or equal to `minSize`: " + maxSize);
        }
        return getNgrams(words, minSize, maxSize);
    }

    @Nonnull
    private static List<Text> getNgrams(@Nonnull final List<Text> words,
            @Nonnegative final int minSize, @Nonnegative final int maxSize) throws HiveException {
        final List<Text> ngrams = new ArrayList<Text>();
        final StringBuilder ngram = new StringBuilder();

        for (int i = 0, numWords = words.size(); i < numWords; i++) {
            for (int ngramSize = minSize; ngramSize <= maxSize; ngramSize++) {
                final int end = i + ngramSize;
                if (end > numWords) { // exceeds the final element
                    continue;
                }

                StringUtils.clear(ngram);
                for (int j = i; j < end; j++) {
                    final Text word = words.get(j);
                    if (word == null) {
                        throw new UDFArgumentException(
                            "`array<string> words` must not contain NULL element");
                    }
                    if (j > i) { // insert single whitespace between elements
                        ngram.append(" ");
                    }
                    ngram.append(word.toString());
                }
                ngrams.add(new Text(ngram.toString()));
            }
        }

        return ngrams;
    }

}
