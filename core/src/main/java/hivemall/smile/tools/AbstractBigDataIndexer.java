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

	  /** The predicate/context names. */
	  protected String[] predLabels;
	  /** The names of the outcomes. */
	  protected String[] outcomeLabels;
	  /** The number of times each predicate occured. */
	  protected int[] predCounts;

	  public String[] getPredLabels() {
	    return predLabels;
	  }

	  public String[] getOutcomeLabels() {
	    return outcomeLabels;
	  }
	  
	  

	  public int[] getPredCounts() {
	    return predCounts;
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
