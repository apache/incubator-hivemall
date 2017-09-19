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

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;
import org.apache.commons.cli.CommandLine;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.commons.cli.Options;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class SkipGramUDTF extends Word2vecBaseUDTF {
    private PrimitiveObjectInspector inWordOI;
    private PrimitiveObjectInspector posWordOI;
    private ListObjectInspector negWordsOI;
    private PrimitiveObjectInspector wordOI;
    private PrimitiveObjectInspector numTrainWordsOI;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        final int numArgs = argOIs.length;

        if (numArgs != 4 && numArgs != 5) {
            throw new UDFArgumentException(getClass().getSimpleName()
                    + " takes 4 or 5 arguments:  [, constant string options]: "
                    + Arrays.toString(argOIs));
        }

        this.inWordOI = HiveUtils.asStringOI(argOIs[0]);
        this.posWordOI = HiveUtils.asStringOI(argOIs[1]);
        this.negWordsOI = HiveUtils.asListOI(argOIs[2]);
        this.wordOI = HiveUtils.asStringOI(negWordsOI.getListElementObjectInspector());
        this.numTrainWordsOI = HiveUtils.asLongCompatibleOI(argOIs[3]);

        processOptions(argOIs);

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
        wordCount = 0L;
        lastWordCount = 0L;
        wordCountActual = 0L;

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    protected Options getOptions() {
        Options opts = super.getOptions();
        opts.addOption("lr", "learningRate", true, "initial learning rate of SGD [default: 0.025]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = super.processOptions(argOIs);

        float lr = 0.025f;

        if (argOIs.length >= 5) {
            String rawArgs = HiveUtils.getConstString(argOIs[4]);
            cl = parseOptions(rawArgs);

            lr = Primitives.parseFloat(cl.getOptionValue("lr"), lr);
            if (lr < 0.d) {
                throw new UDFArgumentException("Argument `float lr` must be positive: " + lr);
            }
        }

        this.currentLR = this.startingLR = lr;
        return cl;
    }

    @Override
    public void process(Object[] args) throws HiveException {
        if (model == null) {
            this.model = createModel();
            this.word2index = new HashMap<>();
        }

        int inWord = getWordId(PrimitiveObjectInspectorUtils.getString(args[0], inWordOI));
        int posWord = getWordId(PrimitiveObjectInspectorUtils.getString(args[1], posWordOI));

        List<?> negWordsList = negWordsOI.getList(args[2]);
        int[] negWords = new int[negWordsList.size()];
        for (int i = 0; i < negWords.length; i++) {
            negWords[i] = getWordId(PrimitiveObjectInspectorUtils.getString(negWordsList.get(i),
                wordOI));
        }

        numTrainWords = PrimitiveObjectInspectorUtils.getLong(args[3], numTrainWordsOI);

        if (wordCount - lastWordCount > 10000) {
            wordCountActual += wordCount - lastWordCount;
            lastWordCount = wordCount;
            currentLR = model.getLearningRate(wordCountActual, numTrainWords, startingLR);
        }

        model.onlineTrain(inWord, posWord, negWords, currentLR);
        wordCount++;
    }

    public void close() throws HiveException {
        if (model != null) {
            forwardModel();
            model = null;
            word2index = null;
        }
    }

    protected AbstractWord2vecModel createModel() {
        return new SkipGramModel(dim);
    }
}
