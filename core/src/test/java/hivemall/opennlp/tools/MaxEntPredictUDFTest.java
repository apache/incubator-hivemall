package hivemall.opennlp.tools;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Field;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredJavaObject;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.matrix.builders.MatrixBuilder;
import hivemall.smile.utils.SmileExtUtils;
import hivemall.utils.lang.ArrayUtils;
import opennlp.maxent.GIS;
import opennlp.maxent.PlainTextByLineDataStream;
import opennlp.maxent.RealBasicEventStream;
import opennlp.maxent.io.GISModelReader;
import opennlp.maxent.io.GISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.Context;
import opennlp.model.EvalParameters;
import opennlp.model.EventStream;
import opennlp.model.IndexHashTable;
import opennlp.model.OnePassRealValueDataIndexer;


public class MaxEntPredictUDFTest {
    /**
     * Test of learn method, of Max Entropy on realTeam data file used by openNLP.
     */
    @Test
    public void testIris() throws IOException, ParseException, HiveException {
    //public static void main(String[] args) throws Exception{
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
		System.out.println(stored_model);
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
        Assert.assertEquals(52, correct);
    }
    
    public static int outcome(String outcome){
    	if (outcome.equals("lose")){
    		return 0;
    	}else if (outcome.equals("win")){
    		return 2;
    	}
    	return 1;	
    }
    
    
    /**
     * Compare MaxEntropy in HiveMall with that of OpenNLP 3.0.0
     */
    @Test
    public void testResemblenceToOpenNLP() throws Exception{
    	
        GIS.SMOOTHING_OBSERVATION = 0.1;
        boolean USE_SMOOTHING = false;
        
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
        AbstractModel openNLPModel = GIS.trainModel(100, new OnePassRealValueDataIndexer(es,0, false), USE_SMOOTHING);
        
        EventStream matrixEs = new MatrixEventStream(x, y, SmileExtUtils.resolveAttributes(types));
        AbstractModel hivemallModel;
		try {
			MatrixForTraining m = new MatrixForTraining(x,y,SmileExtUtils.resolveAttributes(types));
			hivemallModel = BigGIS.trainModel(100, new OnePassBigDataIndexer(matrixEs,0), m);
		} catch (IOException e) {
			throw new HiveException(e.getMessage());
		}
		
		List<String> allOpenNLPKeys = new LinkedList<String>();
		List<String> allOpenNLPOutcomes = new LinkedList<String>();
		List<Double> openNLPParameters = new LinkedList<Double>();
		
		if (openNLPModel != null){
			Field pmap_field = AbstractModel.class.getDeclaredField("pmap");
			pmap_field.setAccessible(true);
			IndexHashTable<String> pmap = (IndexHashTable<String>)pmap_field.get(openNLPModel);
	        
			Field outcomes_field = AbstractModel.class.getDeclaredField("outcomeNames");
			outcomes_field.setAccessible(true);
			String[] outcomes = (String[])outcomes_field.get(openNLPModel);
			
			for (String outcome : outcomes){
				if (!allOpenNLPOutcomes.contains(outcome)){
					allOpenNLPOutcomes.add(outcome);
				}
			}
			
			Field keys_field = IndexHashTable.class.getDeclaredField("keys");
			keys_field.setAccessible(true);
			Object[] keys = (Object[])keys_field.get(pmap);

	        for (Object key : keys){
	        	if (key != null){
	        		
	        		if (!allOpenNLPKeys.contains(String.valueOf(key))){
	        			allOpenNLPKeys.add(String.valueOf(key));
	        		}
	        	}
	        }
	        
			Field paramsField = AbstractModel.class.getDeclaredField("evalParams");
			paramsField.setAccessible(true);
			EvalParameters params = (EvalParameters)paramsField.get(openNLPModel);
	        Context[] contexts = params.getParams();
	        
	        for (String key : allOpenNLPKeys){
		        int index = pmap.get(String.valueOf(key));
        		Context context = contexts[index];
        		
        		double[] ps = context.getParameters();
        		List<Integer> outs = new ArrayList<Integer>();
        		int[] os = context.getOutcomes();
        		
        		for (int o : os){
        			outs.add(o);
        		}
        		
        		for (String outcome : allOpenNLPOutcomes){
        			int i = openNLPModel.getIndex(outcome);
        			if ((i != -1) && outs.contains(i)){
        				openNLPParameters.add(ps[outs.indexOf(i)]);
        			}
        		}
	        }
		}
		
		List<String> allHiveMallKeys = new LinkedList<String>();
		List<String> allHiveMallOutcomes = new LinkedList<String>();
		List<Double> hivemallParameters = new LinkedList<Double>();
		
		if (hivemallModel != null){
			Field pmap_field = AbstractModel.class.getDeclaredField("pmap");
			pmap_field.setAccessible(true);
			IndexHashTable<String> pmap = (IndexHashTable<String>)pmap_field.get(hivemallModel);
	        
			Field outcomes_field = AbstractModel.class.getDeclaredField("outcomeNames");
			outcomes_field.setAccessible(true);
			String[] outcomes = (String[])outcomes_field.get(hivemallModel);
			
			for (String outcome : outcomes){
				if (!allHiveMallOutcomes.contains(outcome)){
					allHiveMallOutcomes.add(outcome);
				}
			}
			
			Field keys_field = IndexHashTable.class.getDeclaredField("keys");
			keys_field.setAccessible(true);
			Object[] keys = (Object[])keys_field.get(pmap);

	        for (Object key : keys){
	        	if (key != null){
	        		
	        		if (!allHiveMallKeys.contains(String.valueOf(key))){
	        			allHiveMallKeys.add(String.valueOf(key));
	        		}
	        	}
	        }
	        
			Field paramsField = AbstractModel.class.getDeclaredField("evalParams");
			paramsField.setAccessible(true);
			EvalParameters params = (EvalParameters)paramsField.get(hivemallModel);
	        Context[] contexts = params.getParams();
	        
	        
	        for (String key : allHiveMallKeys){
		        int index = pmap.get(key);
        		Context context = contexts[index];
        		
        		double[] ps = context.getParameters();
        		List<Integer> outs = new ArrayList<Integer>();
        		int[] os = context.getOutcomes();
        		
        		for (int o : os){
        			outs.add(o);
        		}
        		
        		for (String outcome : allHiveMallOutcomes){
        			int i = hivemallModel.getIndex(outcome);
        			if ((i != -1) && outs.contains(i)){
        				hivemallParameters.add(ps[outs.indexOf(i)]);
        			}
        		}
	        }
		}
		
	     Assert.assertEquals(allOpenNLPOutcomes.size(), allHiveMallOutcomes.size());
	     Assert.assertEquals(allOpenNLPKeys.size(), allHiveMallKeys.size());
	     
	     Assert.assertEquals(openNLPParameters.size(), hivemallParameters.size());
	     Collections.sort(openNLPParameters);
	     Collections.sort(hivemallParameters);
	     
	     for (int i = 0 ; i < openNLPParameters.size(); i++){
	    	 BigDecimal openNLP = new BigDecimal(openNLPParameters.get(i)).setScale(6, RoundingMode.HALF_EVEN);
	    	 BigDecimal hiveMall = new BigDecimal(hivemallParameters.get(i)).setScale(6, RoundingMode.HALF_EVEN);
	    	 Assert.assertEquals(openNLP, hiveMall);
	     }
    }
    
    public static double place(String outcome){
    	if (outcome.equals("home")){
    		return 1;
    	}
    	return 2;	
    }
}
