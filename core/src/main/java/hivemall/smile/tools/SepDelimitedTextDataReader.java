package hivemall.smile.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.StringReader;

import org.apache.hadoop.io.Text;

import opennlp.model.DataReader;

public class SepDelimitedTextDataReader implements DataReader {
	private BufferedReader input;
	String separator = "@";
	StringReader sr;
	
	public SepDelimitedTextDataReader(Text in) {
	    String i = in.toString().replaceAll(separator, "\n");
		sr = new StringReader(i);
	    input = new BufferedReader(sr);
	    try {
	    	// read model name, should be GIS in our case
			input.readLine();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	  
	public double readDouble() throws IOException {
	    return Double.parseDouble(input.readLine());
	}

	public int readInt() throws IOException {
		String line = input.readLine();
	    return Integer.parseInt(line);
	}

	public String readUTF() throws IOException {
	    return input.readLine();
	}
}
