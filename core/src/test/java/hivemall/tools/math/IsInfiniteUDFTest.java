package hivemall.tools.math;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class IsInfiniteUDFTest {
    private IsInfiniteUDF udf;

    @Before
    public void setUp() {
        this.udf = new IsInfiniteUDF();
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
    public void testInfinityNumber() {
        Assert.assertEquals(true, udf.evaluate(Double.POSITIVE_INFINITY));
    }

    @Test
    public void testNan() {
        Assert.assertEquals(false, udf.evaluate(Double.NaN));
    }

}
