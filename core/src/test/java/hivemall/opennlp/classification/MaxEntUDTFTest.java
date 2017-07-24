package hivemall.opennlp.classification;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

import hivemall.classifier.KernelExpansionPassiveAggressiveUDTF;
import hivemall.utils.codec.Base91;
import hivemall.utils.lang.mutable.MutableInt;
import smile.data.AttributeDataset;
import smile.data.parser.ArffParser;

public class MaxEntUDTFTest {
    @Test
    public void testRealTeam() throws IOException, ParseException, HiveException {
    //public static void main(String[] args) throws IOException, ParseException, HiveException {
    	String types = "C,Q,Q";
    	String[] lines = {
    			"away 0.6875 0.5 lose",
    			"away 1.0625 0.5 win",
    			"home 0.8125 0.5 lose",
    			"home 0.9375 0.5 win",
    			"away 0.6875 0.6666 lose",
    			"home 1.0625 0.3333 win",
    			"away 0.8125 0.6666 win",
    			"home 0.9375 0.3333 win",
    			"home 0.6875 0.75 win",
    			"away 1.0625 0.25 tie",
    			"away 0.8125 0.5 tie",
    			"away 0.9375 0.25 tie",
    			"home 0.6875 0.6 tie",
    			"home 1.0625 0.25 tie",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.25 lose",
    			"away 0.6875 0.6 lose",
    			"home 1.0625 0.25 lose",
    			"home 0.8125 0.6 win",
    			"home 0.9375 0.4 lose",
    			"away 0.6875 0.6666 lose",
    			"home 1.0625 0.4 lose",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.5 tie",
    			"away 0.6875 0.7142 win",
    			"away 1.0625 0.5 win",
    			"home 0.8125 0.5714 win",
    			"away 0.9375 0.5 lose",
    			"home 0.6875 0.625 win",
    			"home 1.0625 0.4285 lose",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.5714 win",
    			"home 0.6875 0.5555 lose",
    			"away 1.0625 0.5 lose",
    			"away 0.8125 0.5555 lose",
    			"away 0.9375 0.5 tie",
    			"home 0.6875 0.6 win",
    			"home 1.0625 0.5555 win",
    			"away 0.8125 0.6 tie",
    			"home 0.9375 0.5 win",
    			"home 0.6875 0.5454 win",
    			"home 1.0625 0.5 win",
    			"home 0.8125 0.6 win",
    			"home 0.9375 0.4444 lose",
    			"away 0.6875 0.5 lose",
    			"home 1.0625 0.4545 tie",
    			"home 0.8125 0.5454 tie",
    			"away 0.9375 0.5 lose",
    			"away 0.6875 0.5384 tie",
    			"away 1.0625 0.4545 lose",
    			"home 0.8125 0.5454 lose",
    			"home 0.9375 0.5454 win",
    			"home 0.6875 0.5384 lose",
    			"away 1.0625 0.5 lose",
    			"home 0.8125 0.5833 win",
    			"home 0.9375 0.5 lose",
    			"away 0.6875 0.5714 lose",
    			"away 1.0625 0.5384 win",
    			"away 0.8125 0.5384 lose",
    			"away 0.9375 0.5384 win",
    			"home 0.6875 0.6 tie",
    			"home 1.0625 0.5 tie",
    			"away 0.8125 0.5714 win",
    			"home 0.9375 0.5 win",
    			"home 0.6875 0.6 lose",
    			"away 1.0625 0.5 lose",
    			"home 0.8125 0.5333 win",
    			"home 0.9375 0.4666 win",
    			"home 0.6875 0.625 lose",
    			"away 1.0625 0.5333 tie",
    			"away 0.8125 0.5 lose",
    			"home 0.9375 0.4375 win",
    			"away 0.6875 0.6470 win",
    			"home 1.0625 0.5333 lose",
    			"home 0.8125 0.5294 tie",
    			"away 0.9375 0.4117 lose",
    			"away 0.6875 0.6111 tie",
    			"away 1.0625 0.5625 lose",
    			"home 0.8125 0.5294 lose",
    			"away 0.9375 0.4444 lose",
    			"away 0.6875 0.6111 lose",
    			"home 1.0625 0.5882 tie",
    			"home 0.8125 0.5555 win",
    			"away 0.9375 0.4736 tie",
    			"home 0.6875 0.6315 win",
    			"home 1.0625 0.5882 tie",
    			"home 0.8125 0.5263 lose",
    			"home 0.9375 0.4736 win",
    			"home 0.6875 0.6 lose",
    			"home 1.0625 0.5882 tie",
    			"away 0.8125 0.55 tie",
    			"home 0.9375 0.45 win",
    			"home 0.6875 0.6190 lose",
    			"home 1.0625 0.5882 tie",
    			"away 0.8125 0.55 lose",
    			"away 0.9375 0.4285 lose",
    			"away 0.6875 0.6363 lose",
    			"home 1.0625 0.5882 lose",
    			"home 0.8125 0.5714 lose",
    			"away 0.9375 0.4545 lose"
    	};
    	
        int size = lines.length;
        double[][] x = new double[size][3];
        int[] y = new int[size];

        MaxEntUDTF udtf = new MaxEntUDTF();
        ObjectInspector param = ObjectInspectorUtils.getConstantObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, "-real true -attrs " + types);
        udtf.initialize(new ObjectInspector[] {
                ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaDoubleObjectInspector),
                PrimitiveObjectInspectorFactory.javaIntObjectInspector, param});

        final List<Double> xi = new ArrayList<Double>(x[0].length);
        for (int i = 0; i < size; i++) {
        	String[] obs = lines[i].split("\\s+");
        	xi.add(0, place(obs[0]));
        	xi.add(1, Double.valueOf(obs[1]));
        	xi.add(2, Double.valueOf(obs[2]));
        	y[i] = outcome(obs[3]);
            udtf.process(new Object[] {xi, y[i]});
            xi.clear();
        }

        final MutableInt count = new MutableInt(0);
        Collector collector = new Collector() {
            public void collect(Object input) throws HiveException {
                count.addValue(1);
            }
        };

        udtf.setCollector(collector);
        udtf.close();

        Assert.assertEquals(1, count.getValue());
    }
    
    public static int outcome(String outcome){
    	if (outcome.equals("lose")){
    		return 0;
    	}else if (outcome.equals("tie")){
    		return 1;
    	}
    	return 2;	
    }
    
    public static double place(String outcome){
    	if (outcome.equals("home")){
    		return 1;
    	}
    	return 2;	
    }
}
