package hivemall.tools.math;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class IsFiniteUDFTest {

    private IsFiniteUDF udf;

    @Before
    public void setUp() {
        this.udf = new IsFiniteUDF();
    }

    @Test
    public void testNull() {
        Assert.assertEquals(null, udf.evaluate(null));
    }

    @Test
    public void testDouble() {
        Assert.assertEquals(true, udf.evaluate(1.0));
    }

    @Test
    public void testInfinityNumber() {
        Assert.assertEquals(false, udf.evaluate(Double.POSITIVE_INFINITY));
    }

    @Test
    public void testNan() {
        Assert.assertEquals(false, udf.evaluate(Double.NaN));
    }
}
