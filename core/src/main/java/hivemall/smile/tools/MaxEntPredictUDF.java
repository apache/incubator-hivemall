package hivemall.smile.tools;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.math.vector.DenseVector;
import hivemall.math.vector.SparseVector;
import hivemall.math.vector.Vector;
import hivemall.smile.classification.DecisionTree;
import hivemall.smile.classification.PredictionHandler;
import hivemall.smile.data.Attribute;
import hivemall.smile.data.Attribute.AttributeType;
import hivemall.smile.regression.RegressionTree;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.codec.Base91;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.lang.Preconditions;
import opennlp.model.AbstractModel;
import opennlp.model.GenericModelReader;
import opennlp.model.MaxentModel;
import opennlp.model.RealValueFileEventStream;

@Description(
        name = "predict_maxent_classifier",
        value = "_FUNC_(string model, string attributes, array<double> features)"
                + " - Returns best class and probability distribution among all the classes per instance.")
@UDFType(deterministic = true, stateful = false)
public class MaxEntPredictUDF extends GenericUDF {

    private StringObjectInspector modelOI;
    private StringObjectInspector attributesOI;
    private ListObjectInspector featureListOI;
    private PrimitiveObjectInspector featureElemOI;
    
    @Nullable
    private Vector featuresProbe;
    private Attribute[] attributes;
    
    @Nullable
    private transient MaxentModel evaluator;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 3) {
            throw new UDFArgumentException("_FUNC_ takes 3");
        }

        this.modelOI = HiveUtils.asStringOI(argOIs[0]);
        this.attributesOI = HiveUtils.asStringOI(argOIs[1]);
        
        ListObjectInspector listOI = HiveUtils.asListOI(argOIs[2]);
        this.featureListOI = listOI;
        ObjectInspector elemOI = listOI.getListElementObjectInspector();
        if (HiveUtils.isNumberOI(elemOI)) {
            this.featureElemOI = HiveUtils.asDoubleCompatibleOI(elemOI);
        }  else {
            throw new UDFArgumentException(
                "_FUNC_ takes array<double> for the second argument: "
                        + listOI.getTypeName());
        }

        List<String> fieldNames = new ArrayList<String>(2);
        List<ObjectInspector> fieldOIs = new ArrayList<ObjectInspector>(2);
        fieldNames.add("value");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);
        fieldNames.add("posteriori");
        fieldOIs.add(ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector));
        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public Object evaluate(@Nonnull DeferredObject[] arguments) throws HiveException {
        Object arg0 = arguments[0].get();
        if (arg0 == null) {
            throw new HiveException("Model was null");
        }
        
        Text model = modelOI.getPrimitiveWritableObject(arg0);
        try {
			evaluator = new SepDelimitedTextGISModelReader(model).constructModel();
		} catch (IOException e) {
			throw new HiveException(e.getMessage());
		}
        
        Object arg1 = arguments[1].get();
        if (arg1 == null) {
            throw new HiveException("arguments were null");
        }
        attributes = SmileExtUtils.resolveAttributes(attributesOI.getPrimitiveWritableObject(arg1).toString());
        
        Object arg2 = arguments[2].get();
        if (arg2 == null) {
            throw new HiveException("array<double> features was null");
        }
        this.featuresProbe = parseFeatures(arg2, featuresProbe);

        double[] obs = this.featuresProbe.toArray();
        
        String[] names = new String[obs.length];
  	    float[] values = new float[obs.length];
	    for (int i = 0; i < obs.length; i++){
	    	  if (attributes[i].type == AttributeType.NOMINAL){
	    		  names[i] = i + "_" + String.valueOf(obs[i]).toString();
	    		  values[i] = Double.valueOf(1.0).floatValue();
	    	  }else{
	    		  names[i] = String.valueOf(i);
	    		  values[i] = Double.valueOf(obs[i]).floatValue();
	    	  }
	    }
        
        double[] ocs = evaluator.eval(names,values);
        String klass = evaluator.getBestOutcome(ocs);
        
        List<DoubleWritable> ocss = new ArrayList<DoubleWritable>();
        for (int i = 0; i < ocs.length; i++){
        	ocss.add(new DoubleWritable(ocs[i]));
        }
        
        return new Object[]{new Text(klass), ocss};
    }

    @Nonnull
    private Vector parseFeatures(@Nonnull final Object argObj, @Nonnull Vector probe) {
    	final int length = featureListOI.getListLength(argObj);
    	if (probe == null) {
            probe = new DenseVector(length);
        } else if (length != probe.size()) {
            probe = new DenseVector(length);
        }
    	
    	for (int i = 0; i < length; i++) {
            Object o = featureListOI.getListElement(argObj, i);
            if (o == null) {
                continue;
            }
            double v = PrimitiveObjectInspectorUtils.getDouble(o, featureElemOI);
            probe.set(i, v);
        } 
        return probe;
    }
 
    @Override
    public void close() throws IOException {
        this.modelOI = null;
        this.featureElemOI = null;
        this.featureListOI = null;
        this.evaluator = null;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "maxent_predict(" + Arrays.toString(children) + ")";
    }
}
