package hivemall.tools.map;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;
import org.junit.Assert;
import org.junit.Test;

import java.util.*;

public class MapRouletteUDFTest {

    @Test
    public void testFindMaxProb() throws IllegalAccessException, InstantiationException, HiveException {
        MapRouletteUDF fmp = new MapRouletteUDF();
        fmp.initialize(new ObjectInspector[] {ObjectInspectorFactory.getStandardMapObjectInspector(
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)});
        Map<Object, Integer> solve = new HashMap<>();
        solve.put("Tom", 0);
        solve.put("Jerry", 0);
        solve.put("Amy", 0);
        solve.put("Wong", 0);
        solve.put("Zhao", 0);
        int T = 1000000;
        while(T-- > 0){
            Map<Object, Double> m = new HashMap<>();
            m.put("Tom", 0.1);
            m.put("Jerry", 0.2);
            m.put("Amy", 0.1);
            m.put("Wong", 0.1);
            m.put("Zhao", 0.5);
            GenericUDF.DeferredObject[] arguments =
                    new GenericUDF.DeferredObject[] {new GenericUDF.DeferredJavaObject(m)};
            Object key = fmp.evaluate(arguments);
            solve.put(key, solve.get(key) + 1);
        }
        List<Map.Entry<Object, Integer>> solveList = new ArrayList<>(solve.entrySet());
        Collections.sort(solveList, new KvComparator());
        Object highestSolve = solveList.get(solveList.size() - 1).getKey();
        Assert.assertEquals(highestSolve.toString(), "Zhao");
        Object secondarySolve = solveList.get(solveList.size() - 2).getKey();
        Assert.assertEquals(secondarySolve.toString(), "Jerry");
    }
    private class KvComparator implements Comparator<Map.Entry<Object, Integer>> {

        @Override
        public int compare(Map.Entry<Object, Integer> o1, Map.Entry<Object, Integer> o2) {
            return o1.getValue().compareTo(o2.getValue());
        }
    }
}
