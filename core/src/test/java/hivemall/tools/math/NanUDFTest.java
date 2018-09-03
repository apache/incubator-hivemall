package hivemall.tools.math;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class NanUDFTest {
    private NanUDF udf;

    @Before
    public void setUp() {
        this.udf = new NanUDF();
    }

    @Test
    public void test() {
        Assert.assertEquals(true, Double.isNaN(udf.evaluate()));
    }
}
