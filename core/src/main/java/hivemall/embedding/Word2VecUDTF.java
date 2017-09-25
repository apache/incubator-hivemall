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
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import javax.annotation.Nonnegative;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.HashMap;

public class Word2VecUDTF extends UDTFWithOptions {
    protected transient AbstractWord2VecModel model;
    @Nonnegative
    private float startingLR;
    @Nonnegative
    private long numTrainWords;
    private int dim;
    private Map<String, Integer> word2index;
    private boolean skipgram;

    private PrimitiveObjectInspector inWordOI;
    private ListObjectInspector inWordsOI;
    private PrimitiveObjectInspector posWordOI;
    private ListObjectInspector negWordsOI;
    private PrimitiveObjectInspector negWordOI;
    private PrimitiveObjectInspector numTrainWordsOI;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;

        if (numArgs != 4 && numArgs != 5) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takes 4 or 5 arguments:  [, constant string options]: "
                    + Arrays.toString(argOIs));
        }

        processOptions(argOIs);

        if (skipgram) {
            this.inWordOI = HiveUtils.asStringOI(argOIs[0]);
        } else {
            this.inWordsOI = HiveUtils.asListOI(argOIs[0]);
            this.inWordOI = HiveUtils.asStringOI(inWordsOI.getListElementObjectInspector());
        }

        this.posWordOI = HiveUtils.asStringOI(argOIs[1]);
        this.negWordsOI = HiveUtils.asListOI(argOIs[2]);
        this.negWordOI = HiveUtils.asStringOI(negWordsOI.getListElementObjectInspector());
        this.numTrainWordsOI = HiveUtils.asLongCompatibleOI(argOIs[3]);

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

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (model == null) {
            this.numTrainWords = PrimitiveObjectInspectorUtils.getLong(args[3], numTrainWordsOI);
            this.model = createModel();
            this.word2index = new HashMap<>();
        }

        int posWord = getWordId(PrimitiveObjectInspectorUtils.getString(args[1], posWordOI));

        List<?> negWordsList = negWordsOI.getList(args[2]);
        final int[] negWords = new int[negWordsList.size()];
        for (int i = 0; i < negWords.length; i++) {
            negWords[i] = getWordId(PrimitiveObjectInspectorUtils.getString(negWordsList.get(i),
                negWordOI));
        }


        if (skipgram) {
            int inWord = getWordId(PrimitiveObjectInspectorUtils.getString(args[0], inWordOI));
            model.onlineTrain(inWord, posWord, negWords);
        } else {
            int[] inWords = new int[inWordsOI.getListLength(args[0])];
            for (int i = 0; i < inWords.length; i++) {
                getWordId(PrimitiveObjectInspectorUtils.getString(
                    inWordsOI.getListElement(args[0], i), inWordOI));
            }
            model.onlineTrain(inWords, posWord, negWords);
        }
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("dim", "dimension", true, "the number of vector dimension [default: 100]");
        opts.addOption("model", "modelName", true,
            "The model name of word2vec: skipgram or cbow [default: skipgram]");
        opts.addOption("lr", "learningRate", true,
            "initial learning rate of SGD [default: 0.025 (skipgram) or 0.05 (cbow)]");

        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;
        int dim = 100;
        String modelName = "skipgram";
        float lr = 0.025f;

        if (argOIs.length >= 5) {
            String rawArgs = HiveUtils.getConstString(argOIs[4]);
            cl = parseOptions(rawArgs);

            dim = Primitives.parseInt(cl.getOptionValue("dim"), dim);
            if (dim <= 0.d) {
                throw new UDFArgumentException("Argument `int dim` must be positive: " + dim);
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

        this.dim = dim;
        this.skipgram = modelName.equals("skipgram");
        this.startingLR = lr;
        return cl;
    }

    public void close() throws HiveException {
        if (model != null) {
            forwardModel();
            model = null;
            word2index = null;
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

        for (Map.Entry<String, Integer> entry : word2index.entrySet()) {
            int wordId = entry.getValue();
            word.set(entry.getKey());

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

    protected AbstractWord2VecModel createModel() {
        if (skipgram) {
            return new SkipGramModel(dim, startingLR, numTrainWords);
        } else {
            return new CBoWModel(dim, startingLR, numTrainWords);
        }
    }
}
