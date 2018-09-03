package hivemall.tools.math;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;

@Description(name = "nan", value = "_FUNC_() - Returns the constant representing not-a-number.")
public final class NanUDF extends UDF {
    public double evaluate() {
        return Double.NaN;
    }
}
