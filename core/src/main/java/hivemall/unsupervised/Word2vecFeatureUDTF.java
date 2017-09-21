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
package hivemall.unsupervised;

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
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import javax.annotation.Nonnull;

public class Word2vecFeatureUDTF extends UDTFWithOptions {
    private PRNG rnd;

    // skip-gram with negative sampling parameters
    private int win;
    private int neg;
    private int iter;

    // alias sampler for negative sampling
    private Int2FloatOpenHashTable S;
    private String[] aliasIndex2Word;
    private String[] aliasIndex2OtherWord;
    private int previousNegativeSamplerId;

    private PrimitiveObjectInspector splitIdOI;

    private ListObjectInspector negativeTableOI;
    private ListObjectInspector negativeTableElementListOI;
    private PrimitiveObjectInspector negativeTableElementOI;

    private ListObjectInspector docOI;
    private PrimitiveObjectInspector wordOI;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;

        if (numArgs != 3 && numArgs != 4) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takges 3 or 4 arguments:  [, constant string options]: "
                    + Arrays.toString(argOIs));
        }

        this.splitIdOI = HiveUtils.asIntCompatibleOI(argOIs[0]);
        this.negativeTableOI = HiveUtils.asListOI(argOIs[1]);
        this.negativeTableElementListOI = HiveUtils.asListOI(negativeTableOI.getListElementObjectInspector());
        this.negativeTableElementOI = HiveUtils.asStringOI(negativeTableElementListOI.getListElementObjectInspector());
        this.docOI = HiveUtils.asListOI(argOIs[2]);
        this.wordOI = HiveUtils.asStringOI(docOI.getListElementObjectInspector());

        processOptions(argOIs);

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("k");
        fieldNames.add("inWord");
        fieldNames.add("posWord");
        fieldNames.add("negWords");

        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableStringObjectInspector));

        this.previousNegativeSamplerId = -1;
        this.rnd = RandomNumberGeneratorFactory.createPRNG(1001);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("win", "window", true, "Range for context word [default: 5]");
        opts.addOption("neg", "negative", true,
            "The number of negative sampled words per word [default: 5]");
        opts.addOption("iter", "iteration", true,
            "The number of skip-gram per word. It is equivalent to the epoch of word2vec [default: 5]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;

        int win = 5;
        int neg = 5;
        int iter = 5;

        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[3]);
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
            if (iter < 0) {
                throw new UDFArgumentException("Argument `int iter` must be non-negative: " + iter);
            }
        }

        this.win = win;
        this.neg = neg;
        this.iter = iter;
        return cl;
    }

    @Override
    public void process(Object[] args) throws HiveException {
        int negativeSamplerId = PrimitiveObjectInspectorUtils.getInt(args[0], splitIdOI);
        if (previousNegativeSamplerId != negativeSamplerId) {
            parseNegativeTable(args[1]);
            previousNegativeSamplerId = negativeSamplerId;
        }

        List<?> doc = docOI.getList(args[2]);
        forwardSample(doc);
    }

    private void forwardSample(@Nonnull final List<?> doc) throws HiveException {
        final int numNegative = neg;
        final PRNG _rnd = rnd;
        final PrimitiveObjectInspector _wordOI = wordOI;
        final Int2FloatOpenHashTable S_ = S;
        final String[] aliasIndex2Word_ = aliasIndex2Word;
        final String[] aliasIndex2OtherWord_ = aliasIndex2OtherWord;

        final Text inWord = new Text();
        final Text posWord = new Text();
        final Text[] negWords = new Text[numNegative];
        for (int i = 0; i < numNegative; i++) {
            negWords[i] = new Text();
        }

        final Object[] forwardObjs = new Object[4];
        forwardObjs[0] = new IntWritable(previousNegativeSamplerId);
        forwardObjs[1] = inWord;
        forwardObjs[2] = posWord;
        forwardObjs[3] = negWords;

        int docLength = doc.size();
        for (int inputWordPosition = 0; inputWordPosition < docLength; inputWordPosition++) {
            String inputWord = PrimitiveObjectInspectorUtils.getString(doc.get(inputWordPosition),
                _wordOI);
            inWord.set(inputWord);

            for (int i = 0; i < iter; i++) {
                int windowSize = _rnd.nextInt(win) + 1;

                for (int contextPosition = inputWordPosition - windowSize; contextPosition < inputWordPosition
                        + windowSize + 1; contextPosition++) {
                    if (contextPosition == inputWordPosition)
                        continue;
                    if (contextPosition < 0)
                        continue;
                    if (contextPosition >= docLength)
                        continue;

                    String contextWord = PrimitiveObjectInspectorUtils.getString(
                        doc.get(contextPosition), _wordOI);
                    posWord.set(contextWord);

                    // negative sampling
                    for (int d = 0; d < numNegative; d++) {
                        String sample;
                        do {
                            int k = _rnd.nextInt(S_.size());

                            if (S_.get(k) > _rnd.nextDouble()) {
                                sample = aliasIndex2Word_[k];
                            } else {
                                sample = aliasIndex2OtherWord_[k];
                            }
                        } while (sample.equals(contextWord));

                        negWords[d].set(sample);
                    }

                    forward(forwardObjs);
                }
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
