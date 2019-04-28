package hivemall.tools.map;

import hivemall.TestUtils;
import hivemall.tools.array.ArrayFlattenUDF;
import hivemall.tools.array.ArraySliceUDF;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.WritableComparable;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.*;

public class MapRouletteUDFTest {

    /**
     * Tom, Jerry, Amy, Wong, Zhao joined a roulette.
     * Jerry has 0.2 weight to win.
     * Zhao's weight is highest, he has more chance to win.
     * During data processing ,Tom 's weight was Lost. Algorithm treat Tom 's weight as average.
     * After 1000000 times of roulette, Zhao wins the most. Jerry wins less than Zhao but more than the other.
     *
     * @throws HiveException
     * fmp.initialize may throws UDFArgumentException when checking parameter,
     * org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector#getMap(java.lang.Object) may throw
     * Hive Exception
     */
    @Test
    public void testRoulette() throws HiveException {
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
            m.put("Tom", null);
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

    @Test
    public void testSerialization() throws HiveException, IOException {
        Map<Object, Double> m = new HashMap<>();
        m.put("Tom", 0.1);
        m.put("Jerry", 0.2);
        m.put("Amy", 0.1);
        m.put("Wong", 0.1);
        m.put("Zhao", null);

        TestUtils.testGenericUDFSerialization(
                MapRouletteUDF.class,
                new ObjectInspector[] {
                        ObjectInspectorFactory.getStandardMapObjectInspector(
                                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector)},
                new Object[] {m});
        byte[] serialized = TestUtils.serializeObjectByKryo(new MapRouletteUDFTest());
        TestUtils.deserializeObjectByKryo(serialized, MapRouletteUDFTest.class);
    }

}
