package hivemall.opennlp.tools;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import hivemall.math.matrix.Matrix;
import hivemall.smile.data.Attribute;
import hivemall.smile.data.Attribute.AttributeType;
import opennlp.model.ComparableEvent;
import opennlp.model.Event;

public class MatrixForTraining {
	Matrix ds;
	int[] y;
	Attribute[] attrs;
	
	public int numRows(){
		return ds.numRows();
	}
	
	public int getOutcome(int rowNum){
		return y[rowNum];
	}
	
	public MatrixForTraining (Matrix ds, int[] y, Attribute[] attrs) {
		    this.ds = ds;
		    this.y = y;
		    this.attrs = attrs;
    }
	
	public ComparableEvent createComparableEvent(int rowNum, Map<String,Integer> predicateIndex, Map<String,Integer> omap){
		Event ev = createEvent(rowNum);
		
		String[] econtext = ev.getContext();
        ComparableEvent evt = null;
	    
        String oc = ev.getOutcome();
        int ocID = omap.get(oc);
		
		List<Integer> indexedContext = new ArrayList<Integer>();
        for (int i=0; i<econtext.length; i++) {
            String pred = econtext[i];
            if (predicateIndex.containsKey(pred)) {
               indexedContext.add(predicateIndex.get(pred));
            }
        }

           // drop events with no active features
           if (indexedContext.size() > 0) {
               int[] cons = new int[indexedContext.size()];
               for (int ci=0;ci<cons.length;ci++) {
                 cons[ci] = indexedContext.get(ci);
               }
               evt = new ComparableEvent(ocID, cons, ev.getValues());
           }
         
        return evt;   
	}
	
	  private Event createEvent(int rowNum) {
		    double[] obs = this.ds.getRow(rowNum);
		    int y = this.y[rowNum];
		    if (obs == null) 
		      return null;
		    else{
		    	  String[] names = new String[obs.length];
		    	  float[] values = new float[obs.length];
			      for (int i = 0; i < obs.length; i++){
			    	  if (attrs[i].type == AttributeType.NOMINAL){
			    		  names[i] = i + "_" + String.valueOf(obs[i]).toString();
			    		  values[i] = Double.valueOf(1.0).floatValue();
			    	  }else{
			    		  names[i] = String.valueOf(i);
			    		  values[i] = Double.valueOf(obs[i]).floatValue();
			    	  }
			      }	
			      return new Event(String.valueOf(y), names, values);
		      }
		  }
}
