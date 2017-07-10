package hivemall.smile.tools;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import opennlp.model.ComparableEvent;
import opennlp.model.Event;
import opennlp.model.EventStream;

public class OnePassBigDataIndexer extends AbstractBigDataIndexer  {
	int eventsSize = 0;
	Map<String,Integer> omap = new HashMap<String,Integer>();
	Map<String,Integer> predicatesInOut = new HashMap<String,Integer>();
    /**
     * One argument constructor for DataIndexer which calls the two argument
     * constructor assuming no cutoff.
     *
     * @param eventStream An Event[] which contains the a list of all the Events
     *               seen in the training data.
     */     
    public OnePassBigDataIndexer(EventStream eventStream) throws IOException {
        this(eventStream, 0);
    }

    public Map<String,Integer> getOMap() {
    	return omap;
    }
    
    public Map<String,Integer> getPredicateIndex() {
    	return predicatesInOut;
    }
    
    /**
     * Two argument constructor for DataIndexer.
     *
     * @param eventStream An Event[] which contains the a list of all the Events
     *               seen in the training data.
     * @param cutoff The minimum number of times a predicate must have been
     *               observed in order to be included in the model.
     */
    public OnePassBigDataIndexer(EventStream eventStream, int cutoff) throws IOException {
        computeEventCounts(eventStream,cutoff);
        //sortAndMerge(eventStream, predicateIndex);
    }


    
    /**
     * Reads events from <tt>eventStream</tt> into a linked list.  The
     * predicates associated with each event are counted and any which
     * occur at least <tt>cutoff</tt> times are added to the
     * <tt>predicatesInOut</tt> map along with a unique integer index.
     *
     * @param eventStream an <code>EventStream</code> value
     * @param predicatesInOut a <code>TObjectIntHashMap</code> value
     * @param cutoff an <code>int</code> value
     * @return a <code>TLinkedList</code> value
     */
    private void computeEventCounts(EventStream eventStream,
        int cutoff) throws IOException {
    	
      //Map<String,Integer> omap = new HashMap<String,Integer>();
      int outcomeCount = 0;
      //Map<String,Integer> predicatesInOut = new HashMap<String,Integer>();
      
      Set predicateSet = new HashSet();
      Map<String,Integer> counter = new HashMap<String,Integer>();
      while (eventStream.hasNext()) {
        Event ev = eventStream.next();
        update(ev.getContext(),predicateSet,counter,cutoff);
        eventsSize++;
        
        int ocID;
        String oc = ev.getOutcome();
	    
        if (omap.containsKey(oc)) {
           ocID = omap.get(oc);
        } else {
           ocID = outcomeCount++;
           omap.put(oc, ocID);
        }
      }
      predCounts = new int[predicateSet.size()];
      int index = 0;
      for (Iterator pi=predicateSet.iterator();pi.hasNext();index++) {
        String predicate = (String) pi.next();
        predCounts[index] = counter.get(predicate);
        predicatesInOut.put(predicate,index);
      }
      
	  outcomeLabels = toIndexedStringArray(omap);
	  predLabels = toIndexedStringArray(predicatesInOut);
    }
}
