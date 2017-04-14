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
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.io.FileUtils;
import hivemall.utils.io.NioStatefullSegment;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.SizeOf;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.Description;
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

import com.google.common.annotations.VisibleForTesting;

@Description(name = "train_lda", value = "_FUNC_(array<string> words[, const string options])"
        + " - Returns a relation consists of <int topic, string word, float score>")
public class LDAUDTF extends UDTFWithOptions {
    private static final Log logger = LogFactory.getLog(LDAUDTF.class);

    // Options
    protected int topic;
    protected float alpha;
    protected float eta;
    protected int numDoc;
    protected double tau0;
    protected double kappa;
    protected int iterations;
    protected double delta;
    protected double eps;

    // if `num_doc` option is not given, this flag will be true
    // in that case, UDTF automatically sets `count` value to the _D parameter in an online LDA model
    protected boolean isAutoD;

    // number of proceeded training samples
    protected long count;

    protected OnlineLDAModel model;

    protected ListObjectInspector wordCountsOI;

    // for iterations
    protected NioStatefullSegment fileIO;
    protected ByteBuffer inputBuf;

    public LDAUDTF() {
        this.topic = 10;
        this.alpha = 1.f / topic;
        this.eta = 1.f / topic;
        this.numDoc = -1;
        this.tau0 = 64.d;
        this.kappa = 0.7;
        this.iterations = 1;
        this.delta = 1E-5d;
        this.eps = 1E-1d;
    }

    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("k", "topic", true, "The number of topics [default: 10]");
        opts.addOption("alpha", true, "The hyperparameter for theta [default: 1/k]");
        opts.addOption("eta", true, "The hyperparameter for beta [default: 1/k]");
        opts.addOption("d", "num_doc", true, "The total number of documents [default: auto]");
        opts.addOption("tau", "tau0", true,
            "The parameter which downweights early iterations [default: 64.0]");
        opts.addOption("kappa", true, "Exponential decay rate (i.e., learning rate) [default: 0.7]");
        opts.addOption("iter", "iterations", true, "The maximum number of iterations [default: 1]");
        opts.addOption("delta", true, "Check convergence in the expectation step [default: 1E-5]");
        opts.addOption("eps", "epsilon", true,
            "Check convergence based on the difference of perplexity [default: 1E-1]");
        return opts;
    }

    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
        CommandLine cl = null;

        if (argOIs.length >= 2) {
            String rawArgs = HiveUtils.getConstString(argOIs[1]);
            cl = parseOptions(rawArgs);
            this.topic = Primitives.parseInt(cl.getOptionValue("topic"), 10);
            this.alpha = Primitives.parseFloat(cl.getOptionValue("alpha"), 1.f / topic);
            this.eta = Primitives.parseFloat(cl.getOptionValue("eta"), 1.f / topic);
            this.numDoc = Primitives.parseInt(cl.getOptionValue("num_doc"), -1);
            this.tau0 = Primitives.parseDouble(cl.getOptionValue("tau0"), 64.d);
            if (tau0 <= 0.d) {
                throw new UDFArgumentException("'-tau0' must be positive: " + tau0);
            }
            this.kappa = Primitives.parseDouble(cl.getOptionValue("kappa"), 0.7d);
            if (kappa <= 0.5 || kappa > 1.d) {
                throw new UDFArgumentException("'-kappa' must be in (0.5, 1.0]: " + kappa);
            }
            this.iterations = Primitives.parseInt(cl.getOptionValue("iterations"), 1);
            if (iterations < 1) {
                throw new UDFArgumentException(
                    "'-iterations' must be greater than or equals to 1: " + iterations);
            }
            this.delta = Primitives.parseDouble(cl.getOptionValue("delta"), 1E-5d);
            this.eps = Primitives.parseDouble(cl.getOptionValue("epsilon"), 1E-1d);
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

        this.model = new OnlineLDAModel(topic, alpha, eta, numDoc, tau0, kappa, delta);
        this.count = 0L;
        this.isAutoD = (numDoc < 0);

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

    @Override
    public void process(Object[] args) throws HiveException {
        int length = wordCountsOI.getListLength(args[0]);
        String[] wordCounts = new String[length];
        int j = 0;
        for (int i = 0; i < length; i++) {
            Object o = wordCountsOI.getListElement(args[0], i);
            if (o == null) {
                continue;
            }
            String s = o.toString();
            wordCounts[j] = s;
            j++;
        }

        count++;
        if (isAutoD) {
            model.setNumTotalDocs((int) count);
        }

        recordTrainSampleToTempFile(wordCounts);
        model.train(new String[][] {wordCounts}); // mini-batch w/ single sample
    }

    protected void recordTrainSampleToTempFile(@Nonnull final String[] wordCounts)
            throws HiveException {
        if (iterations == 1) {
            return;
        }

        ByteBuffer buf = inputBuf;
        NioStatefullSegment dst = fileIO;

        if (buf == null) {
            final File file;
            try {
                file = File.createTempFile("hivemall_lda", ".sgmt");
                file.deleteOnExit();
                if (!file.canWrite()) {
                    throw new UDFArgumentException("Cannot write a temporary file: "
                            + file.getAbsolutePath());
                }
                logger.info("Record training samples to a file: " + file.getAbsolutePath());
            } catch (IOException ioe) {
                throw new UDFArgumentException(ioe);
            } catch (Throwable e) {
                throw new UDFArgumentException(e);
            }
            this.inputBuf = buf = ByteBuffer.allocateDirect(1024 * 1024); // 1 MB
            this.fileIO = dst = new NioStatefullSegment(file, false);
        }

        int wcLength = 0;
        for (String wc : wordCounts) {
            wcLength += wc.getBytes().length;
        }
        // recordBytes, wordCounts length, wc1 length, wc1 string, wc2 length, wc2 string, ...
        int recordBytes = (Integer.SIZE * 2 + Integer.SIZE * wcLength) / 8 + wcLength;
        int remain = buf.remaining();
        if (remain < recordBytes) {
            writeBuffer(buf, dst);
        }

        buf.putInt(recordBytes);
        buf.putInt(wordCounts.length);
        for (String wc : wordCounts) {
            buf.putInt(wc.length());
            buf.put(wc.getBytes());
        }
    }

    private static void writeBuffer(@Nonnull ByteBuffer srcBuf, @Nonnull NioStatefullSegment dst)
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
        if (count == 0) {
            this.model = null;
            return;
        }
        if (iterations > 1) {
            runIterativeTraining(iterations);
        }
        forwardModel();
        this.model = null;
    }

    protected final void runIterativeTraining(@Nonnegative final int iterations)
            throws HiveException {
        final ByteBuffer buf = this.inputBuf;
        final NioStatefullSegment dst = this.fileIO;
        assert (buf != null);
        assert (dst != null);
        final long numTrainingExamples = count;

        final Reporter reporter = getReporter();
        final Counters.Counter iterCounter = (reporter == null) ? null : reporter.getCounter(
            "hivemall.lda.OnlineLDA$Counter", "iteration");

        try {
            if (dst.getPosition() == 0L) {// run iterations w/o temporary file
                if (buf.position() == 0) {
                    return; // no training example
                }
                buf.flip();

                int iter = 2;
                float perplexityPrev;
                float perplexity = Float.MAX_VALUE;
                for (; iter <= iterations; iter++) {
                    reportProgress(reporter);
                    setCounterValue(iterCounter, iter);

                    List<String[]> miniBatchList = new ArrayList<String[]>();

                    while (buf.remaining() > 0) {
                        int recordBytes = buf.getInt();
                        assert (recordBytes > 0) : recordBytes;
                        int wcLength = buf.getInt();
                        final String[] wordCounts = new String[wcLength];
                        for (int j = 0; j < wcLength; j++) {
                            int len = buf.getInt();
                            byte[] bytes = new byte[len];
                            buf.get(bytes);
                            wordCounts[j] = new String(bytes);
                        }
                        miniBatchList.add(wordCounts);
                    }
                    buf.rewind();

                    String[][] miniBatch = new String[miniBatchList.size()][];
                    miniBatchList.toArray(miniBatch);
                    model.train(miniBatch);

                    perplexityPrev = perplexity;
                    perplexity = model.computePerplexity();
                    if (Math.abs(perplexityPrev - perplexity) < eps) {
                        break;
                    }
                }
                logger.info("Performed "
                        + Math.min(iter, iterations)
                        + " iterations of "
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
                    throw new HiveException("Failed to flush a file: "
                            + dst.getFile().getAbsolutePath(), e);
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
                float perplexityPrev;
                float perplexity = Float.MAX_VALUE;
                for (; iter <= iterations; iter++) {
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
                            throw new HiveException("Failed to read a file: "
                                    + dst.getFile().getAbsolutePath(), e);
                        }
                        if (bytesRead == 0) { // reached file EOF
                            break;
                        }
                        assert (bytesRead > 0) : bytesRead;

                        List<String[]> miniBatchList = new ArrayList<String[]>();

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
                                int len = buf.getInt();
                                byte[] bytes = new byte[len];
                                buf.get(bytes);
                                wordCounts[j] = new String(bytes);
                            }
                            miniBatchList.add(wordCounts);

                            remain -= recordBytes;
                        }
                        buf.compact();

                        String[][] miniBatch = new String[miniBatchList.size()][];
                        miniBatchList.toArray(miniBatch);
                        model.train(miniBatch);
                    }

                    perplexityPrev = perplexity;
                    perplexity = model.computePerplexity();
                    if (Math.abs(perplexityPrev - perplexity) < eps) {
                        break;
                    }
                }
                logger.info("Performed "
                        + Math.min(iter, iterations)
                        + " iterations of "
                        + NumberUtils.formatNumber(numTrainingExamples)
                        + " training examples on a secondary storage (thus "
                        + NumberUtils.formatNumber(numTrainingExamples * Math.min(iter, iterations))
                        + " training updates in total)");
            }
        } finally {
            // delete the temporary file and release resources
            try {
                dst.close(true);
            } catch (IOException e) {
                throw new HiveException("Failed to close a file: "
                        + dst.getFile().getAbsolutePath(), e);
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

        for (int k = 0; k < topic; k++) {
            topicIdx.set(k);

            final SortedMap<Float, List<String>> topicWords = model.getTopicWords(k);
            for (Map.Entry<Float, List<String>> e : topicWords.entrySet()) {
                score.set(e.getKey());
                List<String> words = e.getValue();
                for (int i = 0; i < words.size(); i++) {
                    word.set(words.get(i));
                    forward(forwardObjs);
                }
            }
        }

        logger.info("Forwarded topic words each of " + topic + " topics");
    }

    /*
     * For testing:
     */

    @VisibleForTesting
    double getLambda(String label, int k) {
        return model.getLambda(label, k);
    }

    @VisibleForTesting
    SortedMap<Float, List<String>> getTopicWords(int k) {
        return model.getTopicWords(k);
    }

    @VisibleForTesting
    SortedMap<Float, List<String>> getTopicWords(int k, int topN) {
        return model.getTopicWords(k, topN);
    }

    @VisibleForTesting
    float[] getTopicDistribution(@Nonnull String[] doc) {
        return model.getTopicDistribution(doc);
    }
}
