package hivemall.tools.map;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;

import java.util.*;

@Description(name = "find_max_prob", value = "_FUNC_(map<K, double> map)"
        + " - Returns the key K which determine to its value , the bigger value is ,the more probability K will return , " +
        "double can store a probability value or a positive number")
@UDFType(deterministic = false) // it is false because it return value base on probability
public class MapFindMaxProbUDF extends UDF {

    /**
     * The map passed in saved all the value and its probability or the weight
     *
     * @param m A map contains a lot of value as key, with their weight as value
     * @return The key that computer selected by key's weight
     */
    public Object evaluate(Map<Object, DoubleWritable> m){
        // normalize the weight
        double sum = 0;
        for (Map.Entry<Object, DoubleWritable> entry: m.entrySet()) {
            sum += entry.getValue().get();
        }
        for (Map.Entry<Object, DoubleWritable> entry: m.entrySet()){
            entry.getValue().set(entry.getValue().get() / sum);
        }

        // sort and generate a number axis
        List<Map.Entry<Object, DoubleWritable>> entryList = new ArrayList<Map.Entry<Object, DoubleWritable>>(m.entrySet());
        Collections.sort(entryList, new MapFindMaxProbUDF.KvComparator());
        double tmp = 0;
        for (Map.Entry<Object, DoubleWritable> entry: entryList) {
            tmp += entry.getValue().get();
            entry.getValue().set(tmp);
        }

        // judge last value
        if(entryList.get(entryList.size() - 1).getValue().get() > 1.0){
            entryList.get(entryList.size() - 1).getValue().set(1.0);
        }

        // pick a Object base on its weight
        double cursor = Math.random();
        for (Map.Entry<Object, DoubleWritable> entry: entryList) {
            if(cursor < entry.getValue().get()){
                return entry.getKey();
            }
        }
        return null;
    }
    private class KvComparator implements Comparator<Map.Entry<Object, DoubleWritable>> {

        @Override
        public int compare(Map.Entry<Object, DoubleWritable> o1, Map.Entry<Object, DoubleWritable> o2) {
            return o1.getValue().compareTo(o2.getValue());
        }
    }

}
