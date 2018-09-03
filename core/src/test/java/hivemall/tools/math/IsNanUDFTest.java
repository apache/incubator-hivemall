package hivemall.tools.math;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class IsNanUDFTest {
    private IsNanUDF udf;

    @Before
    public void setUp() {
        this.udf = new IsNanUDF();
    }

    @Test
    public void testNull() {
        Assert.assertEquals(null, udf.evaluate(null));
    }

    @Test
    public void testDouble() {
        Assert.assertEquals(false, udf.evaluate(1.0));
    }

    @Test
    public void testNan() {
        Assert.assertEquals(true, udf.evaluate(Double.NaN));
    }

    @Test
    public void testInfinityNumber() {
        Assert.assertEquals(false, udf.evaluate(Double.POSITIVE_INFINITY));
    }
}
