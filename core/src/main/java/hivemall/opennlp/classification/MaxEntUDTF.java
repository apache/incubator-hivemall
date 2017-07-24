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

package hivemall.opennlp.classification;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicInteger;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.Options;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.hive.ql.exec.MapredContext;
import org.apache.hadoop.hive.ql.exec.MapredContextAccessor;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.Reporter;
import org.apache.hadoop.mapred.Counters.Counter;

import hivemall.UDTFWithOptions;
import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.MatrixUtils;
import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.math.matrix.ints.DoKIntMatrix;
import hivemall.math.matrix.ints.IntMatrix;
import hivemall.smile.data.Attribute;
import hivemall.smile.data.Attribute.AttributeType;
import hivemall.opennlp.tools.BigGIS;
import hivemall.opennlp.tools.MatrixEventStream;
import hivemall.opennlp.tools.MatrixForTraining;
import hivemall.opennlp.tools.OnePassBigDataIndexer;
import hivemall.opennlp.tools.SepDelimitedTextGISModelWriter;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.smile.utils.SmileTaskExecutor;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Primitives;
import hivemall.utils.lang.RandomUtils;

import opennlp.maxent.io.GISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.ComparableEvent;
import opennlp.model.Event;
import opennlp.model.EventStream;

public class MaxEntUDTF extends UDTFWithOptions{
	private static final Log logger = LogFactory.getLog(MaxEntUDTF.class);
	
	private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    private PrimitiveObjectInspector labelOI;

    private MatrixBuilder matrixBuilder;
    private IntArrayList labels;
    
	private boolean _real;
	private Attribute[] _attributes;
	private static boolean _USE_SMOOTHING;
	private double _SMOOTHING_OBSERVATION;
	
	private int _numTrees = 1;
    
    @Nullable
    private Reporter _progressReporter;
    @Nullable
    private Counter _treeBuildTaskCounter;
    
    @Override
    protected Options getOptions() {
        Options opts = new Options();
        opts.addOption("real", "quantative_feature_presence_indication", true,
            "true or false [default: true]");
        opts.addOption("smoothing", "smoothimg", true, "Shall smoothing be performed [default: false]");
        opts.addOption("constant", "smoothing_constant", true, "real number [default: 1.0]");
        opts.addOption("attrs", "attribute_types", true, "Comma separated attribute types "
                + "(Q for quantitative variable and C for categorical variable. e.g., [Q,C,Q,C])");
        return opts;
    }
    
    @Override
    protected CommandLine processOptions(ObjectInspector[] argOIs) throws UDFArgumentException {
    	boolean real = true;
 	    boolean USE_SMOOTHING = false;
 	    double SMOOTHING_OBSERVATION = 0.1;
 	    
        Attribute[] attrs = null;

        CommandLine cl = null;
        if (argOIs.length >= 3) {
            String rawArgs = HiveUtils.getConstString(argOIs[2]);
            cl = parseOptions(rawArgs);

            real = Primitives.parseBoolean(cl.getOptionValue("quantative_feature_presence_indication"), real);
            attrs = SmileExtUtils.resolveAttributes(cl.getOptionValue("attribute_types"));
            USE_SMOOTHING = Primitives.parseBoolean(cl.getOptionValue("smoothing"), USE_SMOOTHING);
            SMOOTHING_OBSERVATION = Primitives.parseDouble(cl.getOptionValue("smoothing_constant"), SMOOTHING_OBSERVATION);
        }

        this._real = real;
        this._attributes = attrs;
        this._USE_SMOOTHING = USE_SMOOTHING;
        this._SMOOTHING_OBSERVATION = SMOOTHING_OBSERVATION;

        return cl;
    }
    
    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length < 2 || argOIs.length > 3) {
            throw new UDFArgumentException(
                "_FUNC_ takes 2 ~ 3 arguments: array<double> features, int label [, const string options]: "
                        + argOIs.length);
        }

        ListObjectInspector listOI = HiveUtils.asListOI(argOIs[0]);
        ObjectInspector elemOI = listOI.getListElementObjectInspector();
        this.featureListOI = listOI;
        if (HiveUtils.isNumberOI(elemOI)) {
            this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);
            this.matrixBuilder = new CSRMatrixBuilder(8192);
        } else {
            throw new UDFArgumentException(
                "_FUNC_ takes double[] for the first argument: " + listOI.getTypeName());
        }
        this.labelOI = HiveUtils.asIntCompatibleOI(argOIs[1]);

        processOptions(argOIs);

        this.labels = new IntArrayList(1024);

        final ArrayList<String> fieldNames = new ArrayList<String>(6);
        final ArrayList<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>(6);

        fieldNames.add("model_id");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("model_weight");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);
        fieldNames.add("model");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("attributes");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("oob_errors");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);
        fieldNames.add("oob_tests");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }
    
    @Override
    public void process(Object[] args) throws HiveException {
        if (args[0] == null) {
            throw new HiveException("array<double> features was null");
        }
        parseFeatures(args[0], matrixBuilder);
        int label = PrimitiveObjectInspectorUtils.getInt(args[1], labelOI);
        labels.add(label);
    }
    
    private void parseFeatures(@Nonnull final Object argObj, @Nonnull final MatrixBuilder builder) {
    	final int length = featureListOI.getListLength(argObj);
        for (int i = 0; i < length; i++) {
            Object o = featureListOI.getListElement(argObj, i);
            if (o == null) {
                continue;
            }
            double v = PrimitiveObjectInspectorUtils.getDouble(o, featureElemOI);
            builder.nextColumn(i, v);
        } 
        builder.nextRow();
    }
    
    @Override
    public void close() throws HiveException {
        this._progressReporter = getReporter();
        this._treeBuildTaskCounter = (_progressReporter == null) ? null
                : _progressReporter.getCounter("hivemall.smile.MaxEntClassifier$Counter",
                    "finishedGISTask");
        reportProgress(_progressReporter);

        if (!labels.isEmpty()) {
            Matrix x = matrixBuilder.buildMatrix();
            this.matrixBuilder = null;
            int[] y = labels.toArray();
            this.labels = null;

            // run training
            train(x, y);
        }

        // clean up
        this.featureListOI = null;
        this.featureElemOI = null;
        this.labelOI = null;
    }
    
    private void checkOptions() throws HiveException {
    	if (_USE_SMOOTHING == false && _SMOOTHING_OBSERVATION != 0.1) {
            throw new HiveException("Instructions received to avoid smoothing, but smoothing constant is set [" + _SMOOTHING_OBSERVATION + "]");
        }
    }
    
    /**
     * @param x features
     * @param y label
     * @param attrs attribute types
     * @param numTrees The number of trees
     * @param numVars The number of variables to pick up in each node.
     * @param seed The seed number for Random Forest
     */
    private void train(@Nonnull Matrix x, @Nonnull final int[] y) throws HiveException {
        final int numExamples = x.numRows();
        if (numExamples != y.length) {
            throw new HiveException(String.format("The sizes of X and Y don't match: %d != %d",
                numExamples, y.length));
        }
        checkOptions();

        int[] labels = SmileExtUtils.classLables(y);
        Attribute[] attributes = SmileExtUtils.attributeTypes(_attributes, x);

        if (logger.isInfoEnabled()) {
            logger.info("real: " + _real + ", smoothing: " + this._USE_SMOOTHING + ", smoothing constant: "
                    + _SMOOTHING_OBSERVATION);
        }

        IntMatrix prediction = new DoKIntMatrix(numExamples, labels.length); // placeholder for out-of-bag prediction
        AtomicInteger remainingTasks = new AtomicInteger(_numTrees);
        List<TrainingTask> tasks = new ArrayList<TrainingTask>();
        for (int i = 0; i < _numTrees; i++) {
            tasks.add(new TrainingTask(this, i, attributes, x, y, prediction, remainingTasks));
        }
        
        MapredContext mapredContext = MapredContextAccessor.get();
        final SmileTaskExecutor executor = new SmileTaskExecutor(mapredContext);
        try {
            executor.run(tasks);
        } catch (Exception ex) {
            throw new HiveException(ex);
        } finally {
            executor.shotdown();
        }
        
    }
    

    
    /**
     * Synchronized because {@link #forward(Object)} should be called from a single thread.
     * 
     * @param accuracy
     */
    synchronized void forward(final int taskId, @Nonnull final Text model,
    		@Nonnull Attribute[] attributes,
            @Nonnegative final double accuracy, final int[] y,
            @Nonnull final IntMatrix prediction, final boolean lastTask) throws HiveException {
        int oobErrors = 0;
        int oobTests = 0;
        if (lastTask) {
            // out-of-bag error estimate
            for (int i = 0; i < y.length; i++) {
                final int pred = MatrixUtils.whichMax(prediction, i);
                if (pred != -1 && prediction.get(i, pred) > 0) {
                    oobTests++;
                    if (pred != y[i]) {
                        oobErrors++;
                    }
                }
            }
        }
        
        final Object[] forwardObjs = new Object[6];
        String modelId = RandomUtils.getUUID();
        forwardObjs[0] = new Text(modelId);
        forwardObjs[1] = new DoubleWritable(accuracy);
        forwardObjs[2] = model;
        forwardObjs[3] = new Text(SmileExtUtils.resolveAttributes(attributes));
        forwardObjs[4] = new IntWritable(oobErrors);
        forwardObjs[5] = new IntWritable(oobTests);
        forward(forwardObjs);

        reportProgress(_progressReporter);
        incrCounter(_treeBuildTaskCounter, 1);

        logger.info("Forwarded " + taskId + "-th DecisionTree out of " + _numTrees);
    }
    
    /**
     * Trains a regression tree.
     */
    private static final class TrainingTask implements Callable<Integer> {

        /**
         * Training instances.
         */
        @Nonnull
        private final Matrix _x;
        /**
         * Training sample labels.
         */
        @Nonnull
        private final int[] _y;
        
        /**
         * Attribute properties.
         */
        @Nonnull
        private final Attribute[] _attributes;

        /**
         * The out-of-bag predictions.
         */
        @Nonnull
        @GuardedBy("_udtf")
        private final IntMatrix _prediction;

        @Nonnull
        private final MaxEntUDTF _udtf;
        private final int _taskId;
 
        @Nonnull
        private final AtomicInteger _remainingTasks;

        TrainingTask(@Nonnull MaxEntUDTF udtf, int taskId,
        		@Nonnull Attribute[] attributes, @Nonnull Matrix x, @Nonnull int[] y, 
                @Nonnull IntMatrix prediction, @Nonnull AtomicInteger remainingTasks) {
            this._udtf = udtf;
            this._taskId = taskId;
            this._attributes = attributes;
            this._x = x;
            this._y = y;
            this._prediction = prediction;
            this._remainingTasks = remainingTasks;
        }

        @Override
        public Integer call() throws HiveException {
            final int N = _x.numRows();

            EventStream es = new MatrixEventStream(_x, _y, _attributes);
            AbstractModel model;
			try {
				MatrixForTraining mx = new MatrixForTraining(_x, _y, _attributes);
				model = BigGIS.trainModel(100, new OnePassBigDataIndexer(es,0), mx);
			} catch (IOException e) {
				throw new HiveException(e.getMessage());
			}
           
            // out-of-bag prediction
            int oob = 0;
            int correct = 0;
            EventStream test = new MatrixEventStream(_x, _y, _attributes);
            for (int i = 0; i < N; i++) {
                oob++;
                Event event;
				try {
					event = test.next();
				
	         	    float[] vals = event.getValues();
	         	    String[] contexts = event.getContext();
	         	    double[] ocs = model.eval(contexts,vals);
	         	    int p = Integer.valueOf(model.getBestOutcome(ocs));
	
	                if (p == _y[i]) {
	                    correct++;
	                }
	                synchronized (_udtf) {
	                    _prediction.incr(i, p);
	                }
				} catch (IOException e) {
					throw new HiveException(e.getMessage());
				}
            }

            double accuracy = (oob == 0) ? 1.0d : (double) correct / oob;
            int remain = _remainingTasks.decrementAndGet();
            boolean lastTask = (remain == 0);
            GISModelWriter writer;
			try {
				writer = new SepDelimitedTextGISModelWriter(model, "@");
				writer.persist();
			} catch (FileNotFoundException e) {
				throw new HiveException(e.getMessage());
			} catch (IOException e) {
				throw new HiveException(e.getMessage());
			}
            
            _udtf.forward(_taskId + 1, new Text(writer.toString()), _attributes, accuracy, _y, _prediction, lastTask);

            return Integer.valueOf(remain);
        }

        @Nonnull
        private static Text getModel(@Nonnull final AbstractModel model) throws HiveException {
        	GISModelWriter writer;
			try {
				writer = new SepDelimitedTextGISModelWriter(model, "@");
				writer.persist();
			} catch (FileNotFoundException e) {
				throw new HiveException(e.getMessage());
			} catch (IOException e) {
				throw new HiveException(e.getMessage());
			}
            return new Text(writer.toString());
        }

    }
}
