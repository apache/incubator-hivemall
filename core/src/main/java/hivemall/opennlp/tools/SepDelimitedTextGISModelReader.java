package hivemall.opennlp.tools;

import java.io.BufferedReader;

import org.apache.hadoop.io.Text;

import opennlp.maxent.io.GISModelReader;
import opennlp.model.DataReader;

public class SepDelimitedTextGISModelReader extends GISModelReader{

    public SepDelimitedTextGISModelReader (Text in) {
      super(new SepDelimitedTextDataReader(in));
    }
	

}
