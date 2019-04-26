package hivemall.tools.map;

import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.junit.Assert;
import org.junit.Test;

import java.util.*;

public class MapFindMaxProbUDFTest {

    @Test
    public void testFindMaxProb() throws IllegalAccessException, InstantiationException {
        MapFindMaxProbUDF fmp = new MapFindMaxProbUDF();
        Map<WritableComparable, Integer> solve = new HashMap<>();
        solve.put(new Text("Tom"), 0);
        solve.put(new Text("Jerry"), 0);
        solve.put(new Text("Amy"), 0);
        solve.put(new Text("Wong"), 0);
        solve.put(new Text("Zhao"), 0);
        int T = 1000000;
        while(T-- > 0){
            Map<WritableComparable, DoubleWritable> m = new HashMap<>();
            m.put(new Text("Tom"), new DoubleWritable(0.1));
            m.put(new Text("Jerry"), new DoubleWritable(0.2));
            m.put(new Text("Amy"), new DoubleWritable(0.1));
            m.put(new Text("Wong"), new DoubleWritable(0.1));
            m.put(new Text("Zhao"), new DoubleWritable(0.5));
            WritableComparable key = fmp.evaluate(m);
            solve.put(key, solve.get(key) + 1);
        }
        List<Map.Entry<WritableComparable, Integer>> solveList = new ArrayList<>(solve.entrySet());
        Collections.sort(solveList, new KvComparator());
        Object highestSolve = solveList.get(solveList.size() - 1).getKey();
        Assert.assertEquals(highestSolve.toString(), "Zhao");
        Object secondarySolve = solveList.get(solveList.size() - 2).getKey();
        Assert.assertEquals(secondarySolve.toString(), "Jerry");
    }
    private class KvComparator implements Comparator<Map.Entry<WritableComparable, Integer>> {

        @Override
        public int compare(Map.Entry<WritableComparable, Integer> o1, Map.Entry<WritableComparable, Integer> o2) {
            return o1.getValue().compareTo(o2.getValue());
        }
    }
}
