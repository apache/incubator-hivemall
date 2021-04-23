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
package hivemall.nlp.tokenizer.ext;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.ko.dict.UserDictionary;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.junit.Assert;
import org.junit.Test;

public class KoreanAnalyzerTest {

    @Test
    public void testStopwords() throws IOException {
        KoreanAnalyzer analyzer = new KoreanAnalyzer();
        Assert.assertTrue(analyzer.getStopwordSet().size() > 10);

        List<String> results = analyzeTokens(analyzer.tokenStream("", "소설 무궁화꽃이 피었습니다."));
        Assert.assertEquals(5, results.size());
        analyzer.close();
    }

    @Test
    public void testUserDict() {
        UserDictionary dict = readDict();
        Assert.assertNotNull(dict);;
    }

    @Nonnull
    public static UserDictionary readDict() {
        InputStream is = KoreanAnalyzer.class.getResourceAsStream("userdict-ko.txt");
        if (is == null) {
            throw new RuntimeException("Cannot find userdict-ko.txt in test classpath!");
        }
        try {
            try {
                Reader reader = new InputStreamReader(is, StandardCharsets.UTF_8);
                return UserDictionary.open(reader);
            } finally {
                is.close();
            }
        } catch (IOException ioe) {
            throw new RuntimeException(ioe);
        }
    }

    @Nonnull
    private static List<String> analyzeTokens(@Nonnull TokenStream stream) throws IOException {
        final List<String> results = new ArrayList<String>();

        // instantiate an attribute placeholder once
        CharTermAttribute termAttr = stream.getAttribute(CharTermAttribute.class);
        stream.reset();

        while (stream.incrementToken()) {
            String term = termAttr.toString();
            results.add(term);
        }
        return results;
    }

}
