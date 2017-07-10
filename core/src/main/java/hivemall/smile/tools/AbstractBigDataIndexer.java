package hivemall.smile.tools;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import hivemall.math.matrix.Matrix;
import hivemall.math.matrix.builders.CSRMatrixBuilder;
import hivemall.math.matrix.builders.MatrixBuilder;
import opennlp.model.ComparableEvent;
import opennlp.model.Event;
import opennlp.model.EventStream;

public abstract class AbstractBigDataIndexer implements BigDataIndexer {

	  private int numEvents;
	  /** The integer contexts associated with each unique event. */ 
	  protected Matrix contexts;
	  /** The double values associated with each unique event. */ 
	  protected Matrix values;
	  /** The integer outcome associated with each unique event. */ 
	  protected Matrix outcomeList;
	  /** The number of times an event occured in the training data. */
	  protected Matrix numTimesEventsSeen;
	  /** The predicate/context names. */
	  protected String[] predLabels;
	  /** The names of the outcomes. */
	  protected String[] outcomeLabels;
	  /** The number of times each predicate occured. */
	  protected int[] predCounts;

	  public Matrix getContexts() {
	    return contexts;
	  }
	  
	  public Matrix getValues() {
		    return values;
	  }

	  public Matrix getNumTimesEventsSeen() {
	    return numTimesEventsSeen;
	  }

	  public Matrix getOutcomeList() {
	    return outcomeList;
	  }

	  public String[] getPredLabels() {
	    return predLabels;
	  }

	  public String[] getOutcomeLabels() {
	    return outcomeLabels;
	  }
	  
	  

	  public int[] getPredCounts() {
	    return predCounts;
	  }

	  /**
	   * Sorts and uniques the array of comparable events and return the number of unique events.
	   * This method will alter the eventsToCompare array -- it does an in place
	   * sort, followed by an in place edit to remove duplicates.
	   *
	   * @param eventsToCompare a <code>ComparableEvent[]</code> value
	   * @return The number of unique events in the specified list.
	   * @since maxent 1.2.6
	   */
	  protected int sortAndMerge(EventStream eventStream, Map<String,Integer> predicateIndex) throws IOException {
	    int numUniqueEvents = numEvents;

	    ((MatrixEventStream)eventStream).reset();
	    
	    MatrixBuilder contextsMatrixBuilder = new CSRMatrixBuilder(8192);
	    MatrixBuilder outcomesMatrixBuilder = new CSRMatrixBuilder(8192);
	    MatrixBuilder valuesMatrixBuilder = new CSRMatrixBuilder(8192);
	    MatrixBuilder numTimesMatrixBuilder = new CSRMatrixBuilder(8192);

	    Map<String,Integer> omap = new HashMap<String,Integer>();
	    int outcomeCount = 0;
	    
	    
	    while (eventStream.hasNext()) {
	         Event ev = eventStream.next();
	         String[] econtext = ev.getContext();
	         ComparableEvent evt = null;
		    
	         int ocID;
	         String oc = ev.getOutcome();
		    
	         if (omap.containsKey(oc)) {
	            ocID = omap.get(oc);
	         } else {
	            ocID = outcomeCount++;
	            omap.put(oc, ocID);
	         }
	         
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
	            else {
	              System.err.println("Dropped event "+ev.getOutcome()+":"+Arrays.asList(ev.getContext()));
	            }	
	    	
	    	
	    	
	      if (null == evt) {
	        continue; // this was a dupe, skip over it.
	      }
	      
	      outcomesMatrixBuilder.nextColumn(0, evt.outcome);
	      numTimesMatrixBuilder.nextColumn(0, evt.seen);
	      
	      int jj = 0;
	      for (int pIndex : evt.predIndexes){
	    	  contextsMatrixBuilder.nextColumn(jj, pIndex);
	    	  jj++;
      	  }
	      
	      jj = 0;
	      for (float pIndex : evt.values){
	    	  valuesMatrixBuilder.nextColumn(jj, pIndex);
	    	  jj++;
      	  }
	      
	      contextsMatrixBuilder.nextRow();
	      valuesMatrixBuilder.nextRow();
	      outcomesMatrixBuilder.nextRow();
	      numTimesMatrixBuilder.nextRow();
	    }
	    contexts = contextsMatrixBuilder.buildMatrix();
	    values = valuesMatrixBuilder.buildMatrix();
	    outcomeList = outcomesMatrixBuilder.buildMatrix();
	    numTimesEventsSeen = numTimesMatrixBuilder.buildMatrix();
	    
	    outcomeLabels = toIndexedStringArray(omap);
	    predLabels = toIndexedStringArray(predicateIndex);
	    
	    return numUniqueEvents;
	  }
	  
	  
	  public int getNumEvents() {
	    return numEvents;
	  }
	  
	  /**
	   * Updates the set of predicated and counter with the specified event contexts and cutoff. 
	   * @param ec The contexts/features which occur in a event.
	   * @param predicateSet The set of predicates which will be used for model building.
	   * @param counter The predicate counters.
	   * @param cutoff The cutoff which determines whether a predicate is included.
	   */
	   protected static void update(String[] ec, Set predicateSet, Map<String,Integer> counter, int cutoff) {
	    for (int j=0; j<ec.length; j++) {
	      Integer i = counter.get(ec[j]);
	      if (i == null) {
	        counter.put(ec[j], 1);
	      }
	      else {
	        counter.put(ec[j], i+1);
	      }
	      if (!predicateSet.contains(ec[j]) && counter.get(ec[j]) >= cutoff) {
	        predicateSet.add(ec[j]);
	      }
	    }
	  }

	  /**
	   * Utility method for creating a String[] array from a map whose
	   * keys are labels (Strings) to be stored in the array and whose
	   * values are the indices (Integers) at which the corresponding
	   * labels should be inserted.
	   *
	   * @param labelToIndexMap a <code>TObjectIntHashMap</code> value
	   * @return a <code>String[]</code> value
	   * @since maxent 1.2.6
	   */
	  protected static String[] toIndexedStringArray(Map<String,Integer> labelToIndexMap) {
	    final String[] array = new String[labelToIndexMap.size()];
	    for (String label : labelToIndexMap.keySet()) {
	      array[labelToIndexMap.get(label)] = label;
	    }
	    return array;
	  }
	}
