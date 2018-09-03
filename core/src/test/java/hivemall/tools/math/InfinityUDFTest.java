package hivemall.tools.math;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class InfinityUDFTest {
    private InfinityUDF udf;

    @Before
    public void setUp() {
        this.udf = new InfinityUDF();
    }

    @Test
    public void test() {
        Assert.assertEquals(true, Double.isInfinite(udf.evaluate()));
    }
}
