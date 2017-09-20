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
import hivemall.utils.collections.maps.Int2FloatOpenHashTable;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;
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

import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class Word2vecFeatureUDTF extends UDTFWithOptions {
    private Random rnd;

    // skip-gram with negative sampling parameters
    private int win;
    private int neg;

    // alias sampler for negative sampling
    private Int2FloatOpenHashTable S;
    private String[] aliasIndex2Word;
    private String[] aliasIndex2OtherWord;
    private int previousNegativeSamplerId;

    private ListObjectInspector docOI;
    private PrimitiveObjectInspector wordOI;

    private ListObjectInspector negativeTableOI;
    private ListObjectInspector negativeTableElementListOI;
    private PrimitiveObjectInspector negativeTableElementOI;

    private PrimitiveObjectInspector splitIdOI;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;

        if (numArgs != 3 && numArgs != 4) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takges 3 or 4 arguments:  [, constant string options]: "
                    + Arrays.toString(argOIs));
        }

        this.docOI = HiveUtils.asListOI(argOIs[0]);
        wordOI = HiveUtils.asStringOI(docOI.getListElementObjectInspector());

        this.splitIdOI = HiveUtils.asIntCompatibleOI(argOIs[1]);
        this.negativeTableOI = HiveUtils.asListOI(argOIs[2]);
        this.negativeTableElementListOI = HiveUtils.asListOI(negativeTableOI.getListElementObjectInspector());
        this.negativeTableElementOI = HiveUtils.asStringOI(negativeTableElementListOI.getListElementObjectInspector());

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
        this.rnd = new Random();

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("win", "window", true, "Range for context word [default: 5]");
        opts.addOption("neg", "negative", true,
            "The number of negative sampled words per word [default: 5]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;

        int win = 5;
        int neg = 5;

        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[3]);
            cl = parseOptions(rawArgs);

            win = Primitives.parseInt(cl.getOptionValue("win"), win);
            if (win <= 0.d) {
                throw new UDFArgumentException("Argument `int win` must be positive: " + win);
            }

            neg = Primitives.parseInt(cl.getOptionValue("neg"), neg);
            if (neg < 0.d) {
                throw new UDFArgumentException("Argument `int neg` must be non-negative: " + neg);
            }
        }

        this.win = win;
        this.neg = neg;
        return cl;
    }

    @Override
    public void process(Object[] args) throws HiveException {
        int negativeSamplerId = PrimitiveObjectInspectorUtils.getInt(args[1], this.splitIdOI);
        if (previousNegativeSamplerId != negativeSamplerId) {
            parseNegativeTable(args[2]);
            previousNegativeSamplerId = negativeSamplerId;
        }

        // parse document
        List<?> doc = docOI.getList(args[0]);
        int docLength = doc.size();
        for (int inputWordPosition = 0; inputWordPosition < docLength; inputWordPosition++) {
            String word = PrimitiveObjectInspectorUtils.getString(
                docOI.getListElement(args[0], inputWordPosition), wordOI);
            forwardSample(word, inputWordPosition, doc);
        }
    }

    private void forwardSample(String inputWord, int inputWordPosition, List<?> doc)
            throws HiveException {
        final Text inWord = new Text(inputWord);
        final Text posWord = new Text();
        final List<Text> negWords = new ArrayList<>(neg);
        for (int d = 0; d < neg; d++) {
            negWords.add(new Text());
        }

        final Object[] forwardObjs = new Object[4];
        forwardObjs[0] = new IntWritable(previousNegativeSamplerId);
        forwardObjs[1] = inWord;
        forwardObjs[2] = posWord;
        forwardObjs[3] = negWords;

        int docLength = doc.size();
        int windowSize = rnd.nextInt(win) + 1;

        for (int contextPosition = inputWordPosition - windowSize; contextPosition < inputWordPosition
                + windowSize + 1; contextPosition++) {
            if (contextPosition == inputWordPosition)
                continue;
            if (contextPosition < 0)
                continue;
            if (contextPosition >= docLength)
                continue;

            String contextWord = PrimitiveObjectInspectorUtils.getString(doc.get(contextPosition),
                wordOI);
            posWord.set(contextWord);

            for (int d = 0; d < neg; d++) {
                negWords.set(d, new Text(negativeSample(contextWord)));
            }
            forward(forwardObjs);
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
            S.put(i, Float.parseFloat(PrimitiveObjectInspectorUtils.getString(aliasBin.get(1),
                negativeTableElementOI)));
            aliasIndex2Word[i] = PrimitiveObjectInspectorUtils.getString(aliasBin.get(0),
                negativeTableElementOI);
            aliasIndex2OtherWord[i] = PrimitiveObjectInspectorUtils.getString(aliasBin.get(2),
                negativeTableElementOI);
        }

        this.S = S;
        this.aliasIndex2Word = aliasIndex2Word;
        this.aliasIndex2OtherWord = aliasIndex2OtherWord;
    }


    private String negativeSample(final String excludeWord) {
        String sample;
        do {
            int k = rnd.nextInt(S.size());

            if (S.get(k) > rnd.nextFloat()) {
                sample = aliasIndex2Word[k];
            } else {
                sample = aliasIndex2OtherWord[k];
            }
        } while (sample.equals(excludeWord));
        return sample;
    }

    public void close() throws HiveException {}
}
