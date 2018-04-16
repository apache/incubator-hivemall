package hivemall.ftvec.trans;

import hivemall.TestUtils;
import hivemall.utils.hadoop.WritableUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;

import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class QuantifiedFeaturesUDTFTest {

    @Test
    public void testSerialization() throws HiveException {
        final QuantifiedFeaturesUDTF udtf = new QuantifiedFeaturesUDTF();

        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorUtils.getConstantObjectInspector(
                    PrimitiveObjectInspectorFactory.javaBooleanObjectInspector, true),
                PrimitiveObjectInspectorFactory.javaStringObjectInspector,
                PrimitiveObjectInspectorFactory.javaDoubleObjectInspector});

        final List<Object[]> featuresList = new ArrayList<>();
        udtf.setCollector(new Collector() {
            public void collect(Object input) throws HiveException {
                featuresList.add((Object[]) input);
            }
        });

        udtf.process(new Object[] {WritableUtils.val(true), "aaa", 1.0});
        udtf.process(new Object[] {WritableUtils.val(true), "bbb", 2.0});

        // test Kryo serialization
        byte[] serialized = TestUtils.serializeObjectByKryo(udtf);
        TestUtils.deserializeObjectByKryo(serialized, QuantifiedFeaturesUDTF.class);

        udtf.close();

        Assert.assertEquals(2, featuresList.size());

        Object[] features = featuresList.get(0);
        Assert.assertEquals(0, ((Integer) features[0]).intValue());
        Assert.assertEquals(1, ((Double) features[1]).intValue());

        features = featuresList.get(1);
        Assert.assertEquals(1, ((Integer) features[0]).intValue());
        Assert.assertEquals(2, ((Double) features[1]).intValue());
    }
}
