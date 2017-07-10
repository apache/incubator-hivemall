package hivemall.smile.tools;

import java.util.Map;

import hivemall.math.matrix.Matrix;

public interface BigDataIndexer {
	 /**
	   * Returns the array of predicates seen in each event. 
	   * @return a 2-D array whose first dimension is the event index and array this refers to contains
	   * the contexts for that event. 
	   */
	  public Matrix getContexts();
	  
	  public Map<String,Integer> getOMap();
	  
	  public Map<String,Integer> getPredicateIndex();
	  /**
	   * Returns an array indicating the number of times a particular event was seen.
	   * @return an array indexed by the event index indicating the number of times a particular event was seen.
	   */
	  public Matrix getNumTimesEventsSeen();
	  
	  /**
	   * Returns an array indicating the outcome index for each event.
	   * @return an array indicating the outcome index for each event.
	   */
	  public Matrix getOutcomeList();
	  
	  /**
	   * Returns an array of predicate/context names.
	   * @return an array of predicate/context names indexed by context index.  These indices are the
	   * value of the array returned by <code>getContexts</code>.
	   */
	  public String[] getPredLabels();
	  
	  /**
	   * Returns an array of the count of each predicate in the events.
	   * @return an array of the count of each predicate in the events.
	   */
	  public int[] getPredCounts();
	  
	  /**
	    * Returns an array of outcome names.
	    * @return an array of outcome names indexed by outcome index.
	    */
	  public String[] getOutcomeLabels(); 
	  
	  /**
	   * Returns the values associated with each event context or null if integer values are to be used. 
	   * @return the values associated with each event context.
	   */
	  public Matrix getValues();
	  
	  /**
	   * Returns the number of total events indexed.
	   * @return The number of total events indexed.
	   */
	  public int getNumEvents();

}
