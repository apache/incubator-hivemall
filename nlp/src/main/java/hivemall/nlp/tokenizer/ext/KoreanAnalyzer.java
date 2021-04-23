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

import static org.apache.lucene.analysis.TokenStream.DEFAULT_TOKEN_ATTRIBUTE_FACTORY;

import java.io.IOException;
import java.util.Set;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.LowerCaseFilter;
import org.apache.lucene.analysis.StopwordAnalyzerBase;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.ko.KoreanPartOfSpeechStopFilter;
import org.apache.lucene.analysis.ko.KoreanReadingFormFilter;
import org.apache.lucene.analysis.ko.KoreanTokenizer;
import org.apache.lucene.analysis.ko.KoreanTokenizer.DecompoundMode;
import org.apache.lucene.analysis.ko.POS;
import org.apache.lucene.analysis.ko.dict.UserDictionary;

/**
 * Korean analyzer supporting stopwords.
 */
public final class KoreanAnalyzer extends StopwordAnalyzerBase {

    private final UserDictionary userDict;
    private final KoreanTokenizer.DecompoundMode mode;
    private final Set<POS.Tag> stopTags;
    private final boolean outputUnknownUnigrams;

    /**
     * Creates a new KoreanAnalyzer.
     */
    public KoreanAnalyzer() {
        this(null, KoreanTokenizer.DEFAULT_DECOMPOUND, DefaultSetHolder.DEFAULT_STOP_SET, KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS, false);
    }

    /**
     * Creates a new KoreanAnalyzer.
     *
     * @param userDict Optional: if non-null, user dictionary.
     * @param mode Decompound mode.
     * @param stopTags The set of part of speech that should be filtered.
     * @param outputUnknownUnigrams If true outputs unigrams for unknown words.
     */
    public KoreanAnalyzer(@Nullable UserDictionary userDict, @Nonnull DecompoundMode mode,
            @Nullable CharArraySet stopwords, @Nonnull Set<POS.Tag> stopTags,
            boolean outputUnknownUnigrams) {
        super(stopwords);
        this.userDict = userDict;
        this.mode = mode;
        this.stopTags = stopTags;
        this.outputUnknownUnigrams = outputUnknownUnigrams;
    }

    @Nonnull
    public static CharArraySet getDefaultStopSet() {
        return DefaultSetHolder.DEFAULT_STOP_SET;
    }

    @Nonnull
    public static Set<POS.Tag> getDefaultStopTags() {
        return KoreanPartOfSpeechStopFilter.DEFAULT_STOP_TAGS;
    }

    private static class DefaultSetHolder {
        static final CharArraySet DEFAULT_STOP_SET;

        static {
            try {
                DEFAULT_STOP_SET =
                        loadStopwordSet(true, KoreanAnalyzer.class, "stopwords-ko.txt", "#");
            } catch (IOException ex) {
                throw new RuntimeException("Unable to load default stopword set");
            }
        }
    }

    @Override
    protected TokenStreamComponents createComponents(String fieldName) {
        Tokenizer tokenizer = new KoreanTokenizer(DEFAULT_TOKEN_ATTRIBUTE_FACTORY, userDict, mode,
            outputUnknownUnigrams);
        TokenStream stream = new KoreanPartOfSpeechStopFilter(tokenizer, stopTags);
        stream = new KoreanReadingFormFilter(stream);
        stream = new LowerCaseFilter(stream);
        return new TokenStreamComponents(tokenizer, stream);
    }

    @Override
    protected TokenStream normalize(String fieldName, TokenStream in) {
        return new LowerCaseFilter(in);
    }

    @Nonnull
    public static TokenStream normalize(@Nonnull TokenStream in) {
        return new LowerCaseFilter(in);
    }
}
