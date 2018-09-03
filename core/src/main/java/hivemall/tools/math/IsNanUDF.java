package hivemall.tools.math;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;

@Description(name = "is_nan", value = "_FUNC_(x) - Determine if x is not-a-number.")
public final class IsNanUDF extends UDF {
    public Boolean evaluate(Double num) {
        if (num == null) {
            return null;
        } else {
            return num.isNaN(num);
        }
    }
}
