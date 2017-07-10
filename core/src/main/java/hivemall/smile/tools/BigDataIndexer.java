package hivemall.smile.tools;

import java.util.Map;

import hivemall.math.matrix.Matrix;

public interface BigDataIndexer {
	  
	  public Map<String,Integer> getOMap();
	  
	  public Map<String,Integer> getPredicateIndex();
	  
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
	   * Returns the number of total events indexed.
	   * @return The number of total events indexed.
	   */
	  public int getNumEvents();

}
