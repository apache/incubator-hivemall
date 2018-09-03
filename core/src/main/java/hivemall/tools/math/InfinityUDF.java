package hivemall.tools.math;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;

@Description(name = "infinity",
        value = "_FUNC_() - Returns the constant representing positive infinity.")
public final class InfinityUDF extends UDF {
    public double evaluate() {
        return Double.POSITIVE_INFINITY;
    }
}
