package hivemall.tools.map;

import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.junit.Assert;
import org.junit.Test;

import java.util.*;

public class MapFindMaxProbUDFTest {

    @Test
    public void testFindMaxProb() throws IllegalAccessException, InstantiationException {
        MapFindMaxProbUDF fmp = new MapFindMaxProbUDF();
        Map<Object, Integer> solve = new HashMap<>();
        solve.put("Tom", 0);
        solve.put("Jerry", 0);
        solve.put("Amy", 0);
        solve.put("Wong", 0);
        solve.put("Zhao", 0);
        int T = 1000000;
        while(T-- > 0){
            Map<Object, DoubleWritable> m = new HashMap<>();
            m.put("Tom", new DoubleWritable(0.1));
            m.put("Jerry", new DoubleWritable(0.2));
            m.put("Amy", new DoubleWritable(0.1));
            m.put("Wong", new DoubleWritable(0.1));
            m.put("Zhao", new DoubleWritable(0.5));
            Object key = fmp.evaluate(m);
            solve.put(key, solve.get(key) + 1);
        }
        List<Map.Entry<Object, Integer>> solveList = new ArrayList<>(solve.entrySet());
        Collections.sort(solveList, new KvComparator());
        Object highestSolve = solveList.get(solveList.size() - 1).getKey();
        Assert.assertEquals(highestSolve, "Zhao");
        Object secondarySolve = solveList.get(solveList.size() - 2).getKey();
        Assert.assertEquals(secondarySolve, "Jerry");
    }
    private class KvComparator implements Comparator<Map.Entry<Object, Integer>> {

        @Override
        public int compare(Map.Entry<Object, Integer> o1, Map.Entry<Object, Integer> o2) {
            return o1.getValue().compareTo(o2.getValue());
        }
    }
}
