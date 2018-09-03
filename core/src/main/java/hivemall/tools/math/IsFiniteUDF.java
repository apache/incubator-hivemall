package hivemall.tools.math;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;

@Description(name = "is_finite", value = "_FUNC_(x) - Determine if x is infinite.")
public final class IsFiniteUDF extends UDF {
    public Boolean evaluate(Double num) {
        if (num == null) {
            return null;
        } else {
            return !num.isNaN() && !num.isInfinite();
        }
    }
}
