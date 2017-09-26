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
import hivemall.utils.collections.IMapIterator;
import hivemall.utils.collections.maps.Int2FloatOpenHashTable;
import hivemall.utils.collections.maps.OpenHashTable;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import javax.annotation.Nonnegative;
import java.util.List;
import java.util.Arrays;
import java.util.ArrayList;

public class Word2VecUDTF extends UDTFWithOptions {
    protected transient AbstractWord2VecModel model;
    @Nonnegative
    private float startingLR;
    @Nonnegative
    private long numTrainWords;
    private OpenHashTable<String, Integer> word2index;

    @Nonnegative
    private int dim;
    @Nonnegative
    private int win;
    @Nonnegative
    private int neg;
    @Nonnegative
    private int iter;
    private boolean skipgram;

    private Int2FloatOpenHashTable S;
    private int[] aliasWordId;

    private ListObjectInspector negativeTableOI;
    private ListObjectInspector negativeTableElementListOI;
    private PrimitiveObjectInspector negativeTableElementOI;

    private ListObjectInspector docOI;
    private PrimitiveObjectInspector wordOI;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;

        if (numArgs != 3) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takes 3 arguments:  [, constant string options]: "
                    + Arrays.toString(argOIs));
        }

        processOptions(argOIs);

        this.negativeTableOI = HiveUtils.asListOI(argOIs[0]);
        this.negativeTableElementListOI = HiveUtils.asListOI(negativeTableOI.getListElementObjectInspector());
        this.negativeTableElementOI = HiveUtils.asStringOI(negativeTableElementListOI.getListElementObjectInspector());
        this.docOI = HiveUtils.asListOI(argOIs[1]);
        this.wordOI = HiveUtils.asStringOI(docOI.getListElementObjectInspector());

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("word");
        fieldNames.add("i");
        fieldNames.add("wi");

        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        this.model = null;
        this.word2index = null;
        this.S = null;
        this.aliasWordId = null;

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (model == null) {
            parseNegativeTable(args[0]);
            this.model = createModel();
        }

        List<?> rawDoc = docOI.getList(args[1]);

        // parse rawDoc
        List<Integer> doc = new ArrayList<>(rawDoc.size());
        for (int i = 0; i < rawDoc.size(); i++) {
            doc.add(getWordId(PrimitiveObjectInspectorUtils.getString(rawDoc.get(i), wordOI)));
        }

        model.trainOnDoc(doc);
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("n", "numTrainWords", true, "the number of words in document");
        opts.addOption("dim", "dimension", true, "the number of vector dimension [default: 100]");
        opts.addOption("win", "window", true, "Context window size [default: 5]");
        opts.addOption("neg", "negative", true,
            "The number of negative sampled words per word [default: 5]");
        opts.addOption("iter", "iteration", true,
            "The number of skip-gram per word. It is equivalent to the epoch of word2vec [default: 5]");
        opts.addOption("model", "modelName", true,
            "The model name of word2vec: skipgram or cbow [default: skipgram]");
        opts.addOption("lr", "learningRate", true,
            "initial learning rate of SGD [default: 0.025 (skipgram) or 0.05 (cbow)]");


        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        int win = 5;
        int neg = 5;
        int iter = 5;
        int dim = 100;
        long numTrainWords = 0L;
        String modelName = "skipgram";
        float lr = 0.025f;

        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = parseOptions(rawArgs);

            numTrainWords = Primitives.parseLong(cl.getOptionValue("n"), numTrainWords);
            if (numTrainWords <= 0) {
                throw new UDFArgumentException("Argument `int numTrainWords` must be positive: "
                        + numTrainWords);
            }

            dim = Primitives.parseInt(cl.getOptionValue("dim"), dim);
            if (dim <= 0.d) {
                throw new UDFArgumentException("Argument `int dim` must be positive: " + dim);
            }

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

            if (modelName.equals("cbow")) {
                lr = 0.05f;
            }

            lr = Primitives.parseFloat(cl.getOptionValue("lr"), lr);
            if (lr <= 0.f) {
                throw new UDFArgumentException("Argument `float lr` must be positive: " + lr);
            }
        }

        this.numTrainWords = numTrainWords;
        this.win = win;
        this.neg = neg;
        this.iter = iter;
        this.dim = dim;
        this.skipgram = modelName.equals("skipgram");
        this.startingLR = lr;
        return cl;
    }

    public void close() throws HiveException {
        if (model != null) {
            forwardModel();
            this.model = null;
            this.word2index = null;
            this.S = null;
        }
    }

    private void forwardModel() throws HiveException {
        final Text word = new Text();
        final IntWritable dimIndex = new IntWritable();
        final FloatWritable value = new FloatWritable();

        final Object[] result = new Object[3];
        result[0] = word;
        result[1] = dimIndex;
        result[2] = value;

        IMapIterator<String, Integer> iter = word2index.entries();
        while (iter.next() != -1) {
            int wordId = iter.getValue();
            word.set(iter.getKey());

            for (int i = 0; i < dim; i++) {
                dimIndex.set(i);
                value.set(model.inputWeights.get(wordId * dim + i));
                forward(result);
            }
        }
    }

    private int getWordId(String word) {
        if (word2index.containsKey(word)) {
            return word2index.get(word);
        } else {
            int w = word2index.size();
            word2index.put(word, w);
            return w;
        }
    }

    private void parseNegativeTable(Object listObj) {
        int aliasSize = negativeTableOI.getListLength(listObj);
        Int2FloatOpenHashTable S = new Int2FloatOpenHashTable(aliasSize);
        int[] aliasWordId = new int[aliasSize];

        this.word2index = new OpenHashTable<>(aliasSize);

        for (int i = 0; i < aliasSize; i++) {
            List<?> aliasBin = negativeTableElementListOI.getList(negativeTableOI.getListElement(
                listObj, i));
            getWordId(PrimitiveObjectInspectorUtils.getString(aliasBin.get(0),
                negativeTableElementOI));
            S.put(i, Float.parseFloat(PrimitiveObjectInspectorUtils.getString(aliasBin.get(1),
                negativeTableElementOI)));
        }

        for (int i = 0; i < aliasSize; i++) {
            List<?> aliasBin = negativeTableElementListOI.getList(negativeTableOI.getListElement(
                listObj, i));
            aliasWordId[i] = getWordId(PrimitiveObjectInspectorUtils.getString(aliasBin.get(2),
                negativeTableElementOI));
        }

        this.S = S;
        this.aliasWordId = aliasWordId;
    }

    private AbstractWord2VecModel createModel() {
        if (skipgram) {
            return new SkipGramModel(dim, win, neg, iter, startingLR, iter * numTrainWords, S,
                aliasWordId);
        } else {
            return new CBoWModel(dim, win, neg, iter, startingLR, iter * numTrainWords, S,
                aliasWordId);
        }
    }
}
