package hivemall.smile.tools;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredJavaObject;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

import com.google.common.primitives.Doubles;

import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.math.matrix.dense.RowMajorDenseMatrix2d;
import hivemall.smile.classification.DecisionTree;
import hivemall.smile.classification.MaxEntUDTF;
import hivemall.smile.data.Attribute;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.codec.Base91;
import hivemall.utils.lang.ArrayUtils;
import opennlp.maxent.GIS;
import opennlp.maxent.io.GISModelReader;
import opennlp.maxent.io.GISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.EventStream;
import opennlp.model.OnePassRealValueDataIndexer;
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;
import smile.math.Math;
import smile.validation.LOOCV;

public class MaxEntPredictUDFTest {
    private static final boolean DEBUG = false;

    /**
     * Test of learn method, of class DecisionTree.
     */
    @Test
    //public void testIris() throws IOException, ParseException, HiveException {
    public static void main(String[] args) throws Exception{
        String types = "C,Q,Q";
    	String[] lines = {
    			"away 0.6875 0.5 lose",
    			"away 1.0625 0.5 win",
    			"home 0.8125 0.5 lose",
    			"home 0.9375 0.5 win",
    			"away 0.6875 0.6666 lose",
    			"home 1.0625 0.3333 win",
    			"away 0.8125 0.6666 win",
    			"home 0.9375 0.3333 win",
    			"home 0.6875 0.75 win",
    			"away 1.0625 0.25 tie",
    			"away 0.8125 0.5 tie",
    			"away 0.9375 0.25 tie",
    			"home 0.6875 0.6 tie",
    			"home 1.0625 0.25 tie",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.25 lose",
    			"away 0.6875 0.6 lose",
    			"home 1.0625 0.25 lose",
    			"home 0.8125 0.6 win",
    			"home 0.9375 0.4 lose",
    			"away 0.6875 0.6666 lose",
    			"home 1.0625 0.4 lose",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.5 tie",
    			"away 0.6875 0.7142 win",
    			"away 1.0625 0.5 win",
    			"home 0.8125 0.5714 win",
    			"away 0.9375 0.5 lose",
    			"home 0.6875 0.625 win",
    			"home 1.0625 0.4285 lose",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.5714 win",
    			"home 0.6875 0.5555 lose",
    			"away 1.0625 0.5 lose",
    			"away 0.8125 0.5555 lose",
    			"away 0.9375 0.5 tie",
    			"home 0.6875 0.6 win",
    			"home 1.0625 0.5555 win",
    			"away 0.8125 0.6 tie",
    			"home 0.9375 0.5 win",
    			"home 0.6875 0.5454 win",
    			"home 1.0625 0.5 win",
    			"home 0.8125 0.6 win",
    			"home 0.9375 0.4444 lose",
    			"away 0.6875 0.5 lose",
    			"home 1.0625 0.4545 tie",
    			"home 0.8125 0.5454 tie",
    			"away 0.9375 0.5 lose",
    			"away 0.6875 0.5384 tie",
    			"away 1.0625 0.4545 lose",
    			"home 0.8125 0.5454 lose",
    			"home 0.9375 0.5454 win",
    			"home 0.6875 0.5384 lose",
    			"away 1.0625 0.5 lose",
    			"home 0.8125 0.5833 win",
    			"home 0.9375 0.5 lose",
    			"away 0.6875 0.5714 lose",
    			"away 1.0625 0.5384 win",
    			"away 0.8125 0.5384 lose",
    			"away 0.9375 0.5384 win",
    			"home 0.6875 0.6 tie",
    			"home 1.0625 0.5 tie",
    			"away 0.8125 0.5714 win",
    			"home 0.9375 0.5 win",
    			"home 0.6875 0.6 lose",
    			"away 1.0625 0.5 lose",
    			"home 0.8125 0.5333 win",
    			"home 0.9375 0.4666 win",
    			"home 0.6875 0.625 lose",
    			"away 1.0625 0.5333 tie",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.4375 win",
    			"away 0.6875 0.6470 win",
    			"home 1.0625 0.5333 lose",
    			"home 0.8125 0.5294 tie",
    			"away 0.9375 0.4117 lose",
    			"away 0.6875 0.6111 tie",
    			"away 1.0625 0.5625 lose",
    			"home 0.8125 0.5294 lose",
    			"away 0.9375 0.4444 lose",
    			"away 0.6875 0.6111 lose",
    			"home 1.0625 0.5882 tie",
    			"home 0.8125 0.5555 win",
    			"away 0.9375 0.4736 tie",
    			"home 0.6875 0.6315 win",
    			"home 1.0625 0.5882 tie",
    			"home 0.8125 0.5263 lose",
    			"home 0.9375 0.4736 win",
    			"home 0.6875 0.6 lose",
    			"home 1.0625 0.5882 tie",
    			"away 0.8125 0.55 tie",
    			"home 0.9375 0.45 win",
    			"home 0.6875 0.6190 lose",
    			"home 1.0625 0.5882 tie",
    			"away 0.8125 0.55 lose",
    			"away 0.9375 0.4285 lose",
    			"away 0.6875 0.6363 lose",
    			"home 1.0625 0.5882 lose",
    			"home 0.8125 0.5714 lose",
    			"away 0.9375 0.4545 lose"
    	};
    	
        int size = lines.length;
        int[] y = new int[size];
        MatrixBuilder matrixBuilder = new CSRMatrixBuilder(8192);
        
        for (int i = 0; i < size; i++) {
        	String[] lns = lines[i].split("\\s+");
        	matrixBuilder.nextColumn(0, place(lns[0]));
        	matrixBuilder.nextColumn(1, Double.valueOf(lns[1]));
        	matrixBuilder.nextColumn(2, Double.valueOf(lns[2]));
        	y[i] = outcome(lns[3]);
	        matrixBuilder.nextRow();
        }
        
        Matrix x = matrixBuilder.buildMatrix();
        matrixBuilder = null;
        
        EventStream es = new MatrixEventStream(x, y, SmileExtUtils.resolveAttributes(types));
        AbstractModel model;
		try {
			MatrixForTraining m = new MatrixForTraining(x,y,SmileExtUtils.resolveAttributes(types));
			model = BigGIS.trainModel(100, new OnePassBigDataIndexer(es,0), m);
		} catch (IOException e) {
			throw new HiveException(e.getMessage());
		}
        
        // write model to string
		GISModelWriter writer = new SepDelimitedTextGISModelWriter(model, "@");
		writer.persist();
		String stored_model = writer.toString();
		// read model from string
		GISModelReader reader = new SepDelimitedTextGISModelReader(new Text(stored_model));
		AbstractModel read_model = reader.constructModel();
				
		MaxEntPredictUDF udf = new MaxEntPredictUDF();
		ObjectInspector param1 = ObjectInspectorUtils.getConstantObjectInspector(
	            PrimitiveObjectInspectorFactory.javaStringObjectInspector, stored_model);
		ObjectInspector param2 = ObjectInspectorUtils.getConstantObjectInspector(
	            PrimitiveObjectInspectorFactory.javaStringObjectInspector, types);
	    
		udf.initialize(new ObjectInspector[] {
				    param1, param2,
	                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
	                });
		
		int correct = 0;
        for (int i = 0; i < x.numRows(); i++) {
        	DeferredObject[] params = new DeferredJavaObject[3];
        	params[0] = new DeferredJavaObject(new Text(stored_model));
        	params[1] = new DeferredJavaObject(new Text(types));
        	Double[] doubleArray = ArrayUtils.toObject(x.getRow(i));
        	params[2] = new DeferredJavaObject(Arrays.asList(doubleArray));
            Object[] outcome = (Object[])udf.evaluate(params);
            
            if (Integer.valueOf(String.valueOf(outcome[0])) == y[i]){
            	correct++;
            }
        }
        Assert.assertEquals(50, correct);
    }
    
    public static int outcome(String outcome){
    	if (outcome.equals("lose")){
    		return 0;
    	}else if (outcome.equals("win")){
    		return 2;
    	}
    	return 1;	
    }
    
    public static double place(String outcome){
    	if (outcome.equals("home")){
    		return 1;
    	}
    	return 2;	
    }
}
