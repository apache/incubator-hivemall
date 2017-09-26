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
package hivemall.embedding;

import hivemall.UDTFWithOptions;
import hivemall.math.random.PRNG;
import hivemall.math.random.RandomNumberGeneratorFactory;
import hivemall.utils.collections.maps.Int2FloatOpenHashTable;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.Text;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

public class Word2VecFeatureUDTF extends UDTFWithOptions {
    private PRNG rnd;

    // parameters for skip-gram with negative sampling
    @Nonnegative
    private int win;
    private int neg;
    @Nonnegative
    private int iter;
    private boolean skipgram;

    // alias sampler for negative sampling
    private Int2FloatOpenHashTable S;
    private String[] aliasIndex2Word;
    private String[] aliasIndex2OtherWord;

    private ListObjectInspector negativeTableOI;
    private ListObjectInspector negativeTableElementListOI;
    private PrimitiveObjectInspector negativeTableElementOI;

    private ListObjectInspector docOI;
    private PrimitiveObjectInspector wordOI;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;

        if (numArgs != 2 && numArgs != 3) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takges 2 or 3 arguments:  [, constant string options]: "
                    + Arrays.toString(argOIs));
        }

        this.negativeTableOI = HiveUtils.asListOI(argOIs[0]);
        this.negativeTableElementListOI = HiveUtils.asListOI(negativeTableOI.getListElementObjectInspector());
        this.negativeTableElementOI = HiveUtils.asStringOI(negativeTableElementListOI.getListElementObjectInspector());
        this.docOI = HiveUtils.asListOI(argOIs[1]);
        this.wordOI = HiveUtils.asStringOI(docOI.getListElementObjectInspector());

        processOptions(argOIs);

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        if (skipgram) {
            fieldNames.add("inWord");
            fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        } else {
            fieldNames.add("inWords");
            fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector));
        }

        fieldNames.add("posWord");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);

        fieldNames.add("negWords");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector));

        this.S = null;
        this.rnd = RandomNumberGeneratorFactory.createPRNG(1001);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("win", "window", true, "Context window size [default: 5]");
        opts.addOption("neg", "negative", true,
            "The number of negative sampled words per word [default: 5]");
        opts.addOption("iter", "iteration", true,
            "The number of skip-gram per word. It is equivalent to the epoch of word2vec [default: 5]");
        opts.addOption("model", "modelName", true,
            "The model name of word2vec: skipgram or cbow [default: skipgram]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;

        int win = 5;
        int neg = 5;
        int iter = 5;
        String modelName = "skipgram";

        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = parseOptions(rawArgs);

            win = Primitives.parseInt(cl.getOptionValue("win"), win);
            if (win <= 0) {
                throw new UDFArgumentException("Argument `int win` must be positive: " + win);
            }

            neg = Primitives.parseInt(cl.getOptionValue("neg"), neg);
            if (neg < 0) {
                throw new UDFArgumentException("Argument `int neg` must be non-negative: " + neg);
            }

            iter = Primitives.parseInt(cl.getOptionValue("iter"), iter);
            if (iter <= 0) {
                throw new UDFArgumentException("Argument `int iter` must be non-negative: " + iter);
            }

            modelName = cl.getOptionValue("model", modelName);
            if (!(modelName.equals("skipgram") || modelName.equals("cbow"))) {
                throw new UDFArgumentException("Argument `string model` must be skipgram or cbow: "
                        + modelName);
            }
        }

        this.win = win;
        this.neg = neg;
        this.iter = iter;
        this.skipgram = modelName.equals("skipgram");
        return cl;
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (S == null){
            parseNegativeTable(args[0]);
        }

        List<?> rawDoc = docOI.getList(args[1]);

        // parse rawDoc
        List<String> doc = new ArrayList<>(rawDoc.size());
        for (int i = 0; i < rawDoc.size(); i++) {
            doc.add(PrimitiveObjectInspectorUtils.getString(rawDoc.get(i), wordOI));
        }

        if (skipgram) {
            forwardSkipGramSample(doc);
        } else {
            forwardCBoWSample(doc);
        }
    }

    private void forwardSkipGramSample(@Nonnull final List<String> doc) throws HiveException {
        final int numNegative = neg;
        final PRNG _rnd = rnd;
        final Int2FloatOpenHashTable _S = S;
        final String[] _aliasIndex2Word = aliasIndex2Word;
        final String[] _aliasIndex2OtherWord = aliasIndex2OtherWord;

        final Text inWord = new Text();
        final Text posWord = new Text();
        final String[] negWords = new String[numNegative];

        final Object[] forwardObjs = new Object[3];
        forwardObjs[0] = inWord;
        forwardObjs[1] = posWord;
        forwardObjs[2] = Arrays.asList(negWords);

        // reuse instance
        int windowSize, k;
        String negSample;

        int docLength = doc.size();
        for (int inputWordPosition = 0; inputWordPosition < docLength; inputWordPosition++) {
            String inputWord = doc.get(inputWordPosition);
            inWord.set(inputWord);

            for (int i = 0; i < iter; i++) {
                windowSize = _rnd.nextInt(win) + 1;

                for (int contextPosition = inputWordPosition - windowSize; contextPosition < inputWordPosition
                        + windowSize + 1; contextPosition++) {
                    if (contextPosition == inputWordPosition || contextPosition < 0
                            || contextPosition >= docLength) {
                        continue;
                    }

                    String contextWord = doc.get(contextPosition);
                    posWord.set(contextWord);

                    // negative sampling
                    for (int d = 0; d < numNegative; d++) {
                        do {
                            k = _rnd.nextInt(_S.size());

                            if (_S.get(k) > _rnd.nextDouble()) {
                                negSample = _aliasIndex2Word[k];
                            } else {
                                negSample = _aliasIndex2OtherWord[k];
                            }
                        } while (negSample.equals(contextWord));
                        negWords[d] = negSample;
                    }

                    forward(forwardObjs);
                }
            }
        }
    }

    private void forwardCBoWSample(@Nonnull final List<String> doc) throws HiveException {
        final int numNegative = neg;
        final PRNG _rnd = rnd;
        final Int2FloatOpenHashTable _S = S;
        final String[] _aliasIndex2Word = aliasIndex2Word;
        final String[] _aliasIndex2OtherWord = aliasIndex2OtherWord;

        final List<String> inWords = new ArrayList<>();
        final Text posWord = new Text();
        final String[] negWords = new String[numNegative];

        final Object[] forwardObjs = new Object[3];
        forwardObjs[0] = inWords;
        forwardObjs[1] = posWord;
        forwardObjs[2] = Arrays.asList(negWords);

        // reuse instance
        int windowSize, k;
        String negSample;

        int docLength = doc.size();
        for (int positiveWordPosition = 0; positiveWordPosition < docLength; positiveWordPosition++) {
            String positiveWord = doc.get(positiveWordPosition);
            posWord.set(positiveWord);

            for (int i = 0; i < iter; i++) {
                windowSize = _rnd.nextInt(win) + 1;

                // collect context words
                for (int contextPosition = positiveWordPosition - windowSize; contextPosition < positiveWordPosition
                        + windowSize + 1; contextPosition++) {
                    if (contextPosition == positiveWordPosition || contextPosition < 0
                            || contextPosition >= docLength) {
                        continue;
                    }
                    inWords.add(doc.get(contextPosition));
                }

                // negative sampling
                for (int d = 0; d < numNegative; d++) {
                    do {
                        k = _rnd.nextInt(_S.size());

                        if (_S.get(k) > _rnd.nextDouble()) {
                            negSample = _aliasIndex2Word[k];
                        } else {
                            negSample = _aliasIndex2OtherWord[k];
                        }
                    } while (negSample.equals(positiveWord));
                    negWords[d] = negSample;
                }

                forward(forwardObjs);
                inWords.clear();
            }
        }
    }

    private void parseNegativeTable(Object listObj) {
        int aliasSize = negativeTableOI.getListLength(listObj);
        Int2FloatOpenHashTable S = new Int2FloatOpenHashTable(aliasSize);
        String[] aliasIndex2Word = new String[aliasSize];
        String[] aliasIndex2OtherWord = new String[aliasSize];

        for (int i = 0; i < aliasSize; i++) {
            List<?> aliasBin = negativeTableElementListOI.getList(negativeTableOI.getListElement(
                listObj, i));
            aliasIndex2Word[i] = PrimitiveObjectInspectorUtils.getString(aliasBin.get(0),
                negativeTableElementOI);
            S.put(i, Float.parseFloat(PrimitiveObjectInspectorUtils.getString(aliasBin.get(1),
                negativeTableElementOI)));
            aliasIndex2OtherWord[i] = PrimitiveObjectInspectorUtils.getString(aliasBin.get(2),
                negativeTableElementOI);
        }

        this.S = S;
        this.aliasIndex2Word = aliasIndex2Word;
        this.aliasIndex2OtherWord = aliasIndex2OtherWord;
    }

    @Override
    public void close() throws HiveException {}
}
