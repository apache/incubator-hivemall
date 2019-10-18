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
package hivemall.topicmodel;

import hivemall.UDTFWithOptions;
import hivemall.annotations.VisibleForTesting;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.FileUtils;
import hivemall.utils.io.NIOUtils;
import hivemall.utils.io.NioStatefulSegment;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Counters;
import org.apache.hadoop.mapred.Reporter;

import com.google.common.base.Preconditions;

public abstract class ProbabilisticTopicModelBaseUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(ProbabilisticTopicModelBaseUDTF.class);

    public static final int DEFAULT_TOPICS = 10;

    // Options
    protected int topics;
    protected int iterations;
    protected double eps;
    protected int miniBatchSize;

    protected String[][] miniBatch;
    protected int miniBatchCount;

    protected transient AbstractProbabilisticTopicModel model;

    protected ListObjectInspector wordCountsOI;

    // for iterations
    protected transient NioStatefulSegment fileIO;
    protected transient ByteBuffer inputBuf;

    private float cumPerplexity;

    public ProbabilisticTopicModelBaseUDTF() {
        this.topics = DEFAULT_TOPICS;
        this.iterations = 10;
        this.eps = 1E-1d;
        this.miniBatchSize = 128; // if 1, truly online setting
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("k", "topics", true, "The number of topics [default: 10]");
        opts.addOption("iter", "iterations", true,
            "The maximum number of iterations [default: 10]");
        opts.addOption("eps", "epsilon", true,
            "Check convergence based on the difference of perplexity [default: 1E-1]");
        opts.addOption("s", "mini_batch_size", true,
            "Repeat model updating per mini-batch [default: 128]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;

        if (argOIs.length >= 2) {
            String rawArgs = HiveUtils.getConstString(argOIs[1]);
            cl = parseOptions(rawArgs);
            this.topics = Primitives.parseInt(cl.getOptionValue("topics"), DEFAULT_TOPICS);
            this.iterations = Primitives.parseInt(cl.getOptionValue("iterations"), 10);
            if (iterations < 1) {
                throw new UDFArgumentException(
                    "'-iterations' must be greater than or equals to 1: " + iterations);
            }
            this.eps = Primitives.parseDouble(cl.getOptionValue("epsilon"), 1E-1d);
            this.miniBatchSize = Primitives.parseInt(cl.getOptionValue("mini_batch_size"), 128);
        }

        return cl;
    }

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 1) {
            throw new UDFArgumentException(
                "_FUNC_ takes 1 arguments: array<string> words [, const string options]");
        }

        this.wordCountsOI = HiveUtils.asListOI(argOIs[0]);
        HiveUtils.validateFeatureOI(wordCountsOI.getListElementObjectInspector());

        processOptions(argOIs);

        this.model = null;
        this.miniBatch = new String[miniBatchSize][];
        this.miniBatchCount = 0;
        this.cumPerplexity = 0.f;

        ArrayList<String> fieldNames = new ArrayList<String>();
        ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>();
        fieldNames.add("topic");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("word");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("score");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Nonnull
    protected abstract AbstractProbabilisticTopicModel createModel();

    @Override
    public void process(Object[] args) throws HiveException {
        if (model == null) {
            this.model = createModel();
        }

        Preconditions.checkArgument(args.length >= 1);
        Object arg0 = args[0];
        if (arg0 == null) {
            return;
        }

        final int length = wordCountsOI.getListLength(arg0);
        final String[] wordCounts = new String[length];
        int j = 0;
        for (int i = 0; i < length; i++) {
            Object o = wordCountsOI.getListElement(arg0, i);
            if (o == null) {
                throw new HiveException("Given feature vector contains invalid null elements");
            }
            String s = o.toString();
            wordCounts[j] = s;
            j++;
        }
        if (j == 0) {// avoid empty documents
            return;
        }

        model.accumulateDocCount();

        update(wordCounts);

        recordTrainSampleToTempFile(wordCounts);
    }

    protected void recordTrainSampleToTempFile(@Nonnull final String[] wordCounts)
            throws HiveException {
        if (iterations == 1) {
            return;
        }

        ByteBuffer buf = inputBuf;
        NioStatefulSegment dst = fileIO;

        if (buf == null) {
            final File file;
            try {
                file = File.createTempFile("hivemall_topicmodel", ".sgmt");
                file.deleteOnExit();
                if (!file.canWrite()) {
                    throw new UDFArgumentException(
                        "Cannot write a temporary file: " + file.getAbsolutePath());
                }
                logger.info("Record training samples to a file: " + file.getAbsolutePath());
            } catch (IOException ioe) {
                throw new UDFArgumentException(ioe);
            } catch (Throwable e) {
                throw new UDFArgumentException(e);
            }
            this.inputBuf = buf = ByteBuffer.allocateDirect(1024 * 1024); // 1 MB
            this.fileIO = dst = new NioStatefulSegment(file, false);
        }

        // wordCounts length, wc1 length, wc1 string, wc2 length, wc2 string, ...
        int wcLengthTotal = 0;
        for (String wc : wordCounts) {
            if (wc == null) {
                continue;
            }
            wcLengthTotal += wc.length();
        }
        int recordBytes = SizeOf.INT + SizeOf.INT * wordCounts.length + wcLengthTotal * SizeOf.CHAR;
        int requiredBytes = SizeOf.INT + recordBytes; // need to allocate space for "recordBytes" itself

        int remain = buf.remaining();
        if (remain < requiredBytes) {
            writeBuffer(buf, dst);
        }

        buf.putInt(recordBytes);
        buf.putInt(wordCounts.length);
        for (String wc : wordCounts) {
            NIOUtils.putString(wc, buf);
        }
    }

    private void update(@Nonnull final String[] wordCounts) {
        miniBatch[miniBatchCount] = wordCounts;
        miniBatchCount++;

        if (miniBatchCount == miniBatchSize) {
            train();
        }
    }

    protected void train() {
        if (miniBatchCount == 0) {
            return;
        }

        model.train(miniBatch);

        this.cumPerplexity += model.computePerplexity();

        Arrays.fill(miniBatch, null); // clear
        miniBatchCount = 0;
    }

    private static void writeBuffer(@Nonnull ByteBuffer srcBuf, @Nonnull NioStatefulSegment dst)
            throws HiveException {
        srcBuf.flip();
        try {
            dst.write(srcBuf);
        } catch (IOException e) {
            throw new HiveException("Exception causes while writing a buffer to file", e);
        }
        srcBuf.clear();
    }

    @Override
    public void close() throws HiveException {
        if (model == null) {
            logger.warn(
                "Model is not initialized bacause no training exmples to learn. Better to revise input data.");
            return;
        } else if (model.getDocCount() == 0L) {
            logger.warn(
                "model.getDocCount() is zero because no training exmples to learn. Better to revise input data.");
            this.model = null;
            return;
        }

        finalizeTraining();
        forwardModel();
        this.model = null;
    }

    @VisibleForTesting
    void finalizeTraining() throws HiveException {
        if (miniBatchCount > 0) { // update for remaining samples
            model.train(Arrays.copyOfRange(miniBatch, 0, miniBatchCount));
        }
        if (iterations > 1) {
            runIterativeTraining(iterations);
        }
    }

    protected final void runIterativeTraining(@Nonnegative final int iterations)
            throws HiveException {
        final ByteBuffer buf = this.inputBuf;
        final NioStatefulSegment dst = this.fileIO;
        assert (buf != null);
        assert (dst != null);
        final long numTrainingExamples = model.getDocCount();

        long numTrain = numTrainingExamples / miniBatchSize;
        if (numTrainingExamples % miniBatchSize != 0L) {
            numTrain++;
        }

        final Reporter reporter = getReporter();
        final Counters.Counter iterCounter = (reporter == null) ? null
                : reporter.getCounter("hivemall.topicmodel.ProbabilisticTopicModel$Counter",
                    "iteration");

        try {
            if (dst.getPosition() == 0L) {// run iterations w/o temporary file
                if (buf.position() == 0) {
                    return; // no training example
                }
                buf.flip();

                int iter = 2;
                float perplexity = cumPerplexity / numTrain;
                float perplexityPrev;
                for (; iter <= iterations; iter++) {
                    perplexityPrev = perplexity;
                    cumPerplexity = 0.f;

                    reportProgress(reporter);
                    setCounterValue(iterCounter, iter);

                    while (buf.remaining() > 0) {
                        int recordBytes = buf.getInt();
                        assert (recordBytes > 0) : recordBytes;
                        int wcLength = buf.getInt();
                        final String[] wordCounts = new String[wcLength];
                        for (int j = 0; j < wcLength; j++) {
                            wordCounts[j] = NIOUtils.getString(buf);
                        }
                        update(wordCounts);
                    }
                    buf.rewind();

                    // mean perplexity over `numTrain` mini-batches
                    perplexity = cumPerplexity / numTrain;
                    logger.info("Mean perplexity over mini-batches: " + perplexity);
                    if (Math.abs(perplexityPrev - perplexity) < eps) {
                        break;
                    }
                }
                logger.info("Performed " + Math.min(iter, iterations) + " iterations of "
                        + NumberUtils.formatNumber(numTrainingExamples)
                        + " training examples on memory (thus "
                        + NumberUtils.formatNumber(numTrainingExamples * Math.min(iter, iterations))
                        + " training updates in total) ");
            } else {// read training examples in the temporary file and invoke train for each example
                // write training examples in buffer to a temporary file
                if (buf.remaining() > 0) {
                    writeBuffer(buf, dst);
                }
                try {
                    dst.flush();
                } catch (IOException e) {
                    throw new HiveException(
                        "Failed to flush a file: " + dst.getFile().getAbsolutePath(), e);
                }
                if (logger.isInfoEnabled()) {
                    File tmpFile = dst.getFile();
                    logger.info("Wrote " + numTrainingExamples
                            + " records to a temporary file for iterative training: "
                            + tmpFile.getAbsolutePath() + " (" + FileUtils.prettyFileSize(tmpFile)
                            + ")");
                }

                // run iterations
                int iter = 2;
                float perplexity = cumPerplexity / numTrain;
                float perplexityPrev;
                for (; iter <= iterations; iter++) {
                    perplexityPrev = perplexity;
                    cumPerplexity = 0.f;

                    setCounterValue(iterCounter, iter);

                    buf.clear();
                    dst.resetPosition();
                    while (true) {
                        reportProgress(reporter);
                        // TODO prefetch
                        // writes training examples to a buffer in the temporary file
                        final int bytesRead;
                        try {
                            bytesRead = dst.read(buf);
                        } catch (IOException e) {
                            throw new HiveException(
                                "Failed to read a file: " + dst.getFile().getAbsolutePath(), e);
                        }
                        if (bytesRead == 0) { // reached file EOF
                            break;
                        }
                        assert (bytesRead > 0) : bytesRead;

                        // reads training examples from a buffer
                        buf.flip();
                        int remain = buf.remaining();
                        if (remain < SizeOf.INT) {
                            throw new HiveException("Illegal file format was detected");
                        }
                        while (remain >= SizeOf.INT) {
                            int pos = buf.position();
                            int recordBytes = buf.getInt();
                            remain -= SizeOf.INT;
                            if (remain < recordBytes) {
                                buf.position(pos);
                                break;
                            }

                            int wcLength = buf.getInt();
                            final String[] wordCounts = new String[wcLength];
                            for (int j = 0; j < wcLength; j++) {
                                wordCounts[j] = NIOUtils.getString(buf);
                            }
                            update(wordCounts);

                            remain -= recordBytes;
                        }
                        buf.compact();
                    }

                    // mean perplexity over `numTrain` mini-batches
                    perplexity = cumPerplexity / numTrain;
                    logger.info("Mean perplexity over mini-batches: " + perplexity);
                    if (Math.abs(perplexityPrev - perplexity) < eps) {
                        break;
                    }
                }
                logger.info("Performed " + Math.min(iter, iterations) + " iterations of "
                        + NumberUtils.formatNumber(numTrainingExamples)
                        + " training examples on a secondary storage (thus "
                        + NumberUtils.formatNumber(numTrainingExamples * Math.min(iter, iterations))
                        + " training updates in total)");
            }
        } catch (Throwable e) {
            throw new HiveException("Exception caused in the iterative training", e);
        } finally {
            // delete the temporary file and release resources
            try {
                dst.close(true);
            } catch (IOException e) {
                throw new HiveException(
                    "Failed to close a file: " + dst.getFile().getAbsolutePath(), e);
            }
            this.inputBuf = null;
            this.fileIO = null;
        }
    }

    protected void forwardModel() throws HiveException {
        final IntWritable topicIdx = new IntWritable();
        final Text word = new Text();
        final FloatWritable score = new FloatWritable();

        final Object[] forwardObjs = new Object[3];
        forwardObjs[0] = topicIdx;
        forwardObjs[1] = word;
        forwardObjs[2] = score;

        for (int k = 0; k < topics; k++) {
            topicIdx.set(k);

            final SortedMap<Float, List<String>> topicWords = model.getTopicWords(k);
            if (topicWords == null) {
                continue;
            }
            for (Map.Entry<Float, List<String>> e : topicWords.entrySet()) {
                score.set(e.getKey().floatValue());
                for (String v : e.getValue()) {
                    word.set(v);
                    forward(forwardObjs);
                }
            }
        }

        logger.info("Forwarded topic words each of " + topics + " topics");
    }

    @VisibleForTesting
    float getWordScore(String label, int k) {
        return model.getWordScore(label, k);
    }

    @VisibleForTesting
    SortedMap<Float, List<String>> getTopicWords(int k) {
        return model.getTopicWords(k);
    }

    @VisibleForTesting
    float[] getTopicDistribution(@Nonnull String[] doc) {
        return model.getTopicDistribution(doc);
    }
}
