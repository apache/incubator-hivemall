package hivemall.tools.map;

import hivemall.utils.hadoop.HiveUtils;
import org.apache.hadoop.hive.ql.exec.*;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

import java.util.*;

import static hivemall.HivemallConstants.*;

@Description(name = "map_roulette", value = "_FUNC_(Map<K, double> map)"
        + " - Returns the key K which determine to its value , the bigger value is ,the more probability K will return , " +
        "double can store a probability value or a positive number")
@UDFType(deterministic = false) // it is false because it return value base on probability
public class MapRouletteUDF extends GenericUDF {

    /**
     * The map passed in saved all the value and its probability or the weight
     *
     * @param m A map contains a lot of value as key, with their weight as value
     * @return The key that computer selected by key's weight
     */
    private Object algorithm(Map<Object, Double> m){
        // normalize the weight
        double sum = 0;
        for (Map.Entry<Object, Double> entry: m.entrySet()) {
            sum += entry.getValue();
        }
        for (Map.Entry<Object, Double> entry: m.entrySet()){
            entry.setValue(entry.getValue() / sum);
        }

        // sort and generate a number axis
        List<Map.Entry<Object, Double>> entryList = new ArrayList<>(m.entrySet());
        Collections.sort(entryList, new MapRouletteUDF.KvComparator());
        double tmp = 0;
        for (Map.Entry<Object, Double> entry: entryList) {
            tmp += entry.getValue();
            entry.setValue(tmp);
        }

        // judge last value
        if(entryList.get(entryList.size() - 1).getValue() > 1.0){
            entryList.get(entryList.size() - 1).setValue(1.0);
        }

        // pick a Object base on its weight
        double cursor = Math.random();
        for (Map.Entry<Object, Double> entry: entryList) {
            if(cursor < entry.getValue()){
                return entry.getKey();
            }
        }
        return null;
    }

    private transient MapObjectInspector mapOI;
    private transient PrimitiveObjectInspector valueOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] arguments) throws UDFArgumentException {
        if(arguments.length != 1)
            throw new UDFArgumentLengthException(
                    "Expected one arguments for map_find_max_prob: " + arguments.length);
        if (arguments[0].getCategory() != ObjectInspector.Category.MAP) {
            throw new UDFArgumentTypeException(0,
                    "Only map type arguments are accepted for the key but "
                            + arguments[0].getTypeName() + " was passed as parameter 1.");
        }
        mapOI = HiveUtils.asMapOI(arguments[0]);
        ObjectInspector keyOI = mapOI.getMapKeyObjectInspector();

        //judge valueOI is a number
        valueOI = (PrimitiveObjectInspector) mapOI.getMapValueObjectInspector();
        switch (valueOI.getTypeName()){
            case INT_TYPE_NAME:
            case DOUBLE_TYPE_NAME:
            case BIGINT_TYPE_NAME:
            case FLOAT_TYPE_NAME:
            case SMALLINT_TYPE_NAME:
            case TINYINT_TYPE_NAME:
            case DECIMAL_TYPE_NAME:
            case STRING_TYPE_NAME:
                // An empty map or a map full of {null, null} will get string type
                // An number in string format like "3.5" also support
                break;
            default:
                throw new UDFArgumentException(
                        "Expected a number but get: " + valueOI.getTypeName());
        }
        return keyOI;
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        Map<Object, Double> input = processObjectDoubleMap(arguments[0]);
        if(input == null){
            return null;
        }
        // handle empty map
        if(input.isEmpty()){
            return null;
        }
        return algorithm(input);
    }

    private Map<Object, Double> processObjectDoubleMap(DeferredObject argument) throws HiveException {
        // get
        Map<?, ?> m = mapOI.getMap(argument.get());
        if(m == null){
            return null;
        }
        if(m.size() == 0){
            return null;
        }
        // convert
        Map<Object, Double> input = new HashMap<>();
        Double avg = 0.0;
        for (Map.Entry<?, ?> entry: m.entrySet()) {
            Object key = entry.getKey();
            Double value = null;
            if(entry.getValue() != null){
                value = PrimitiveObjectInspectorUtils.convertPrimitiveToDouble(entry.getValue(), valueOI);
                if(value < 0){
                    throw new UDFArgumentException(
                            entry.getValue() + " < 0"
                    );
                }
                avg += value;
            }
            input.put(key, value);
        }
        avg /= m.size();
        for(Map.Entry<?, ?> entry: input.entrySet()){
            if(entry.getValue() == null){
                Object key = entry.getKey();
                input.put(key, avg);
            }
        }
        return input;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "map_roulette("+ Arrays.toString(children) +")";
    }

    private class KvComparator implements Comparator<Map.Entry<Object, Double>> {

        @Override
        public int compare(Map.Entry<Object, Double> o1, Map.Entry<Object, Double> o2) {
            return o1.getValue().compareTo(o2.getValue());
        }
    }

}
