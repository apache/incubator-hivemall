package hivemall.opennlp.tools;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.StringWriter;
import java.util.zip.GZIPOutputStream;

import opennlp.maxent.io.GISModelWriter;
import opennlp.model.AbstractModel;

public class SepDelimitedTextGISModelWriter extends GISModelWriter {
	  BufferedWriter output;
	  String separator;
	  StringWriter sw;
	  /**
	   * Constructor which takes a GISModel and a File and prepares itself to
	   * write the model to that file. Detects whether the file is gzipped or not
	   * based on whether the suffix contains ".gz".
	   *
	   * @param model The GISModel which is to be persisted.
	   * @param f The File in which the model is to be persisted.
	   */
	  public SepDelimitedTextGISModelWriter (AbstractModel model, String separator)
			  throws IOException, FileNotFoundException {
			    super(model);
			    sw = new StringWriter();
			    output = new BufferedWriter(sw);
			    this.separator = separator;
			  }

			  /**
			   * Constructor which takes a GISModel and a BufferedWriter and prepares
			   * itself to write the model to that writer.
			   *
			   * @param model The GISModel which is to be persisted.
			   * @param bw The BufferedWriter which will be used to persist the model.
			   */
			  public SepDelimitedTextGISModelWriter (AbstractModel model, BufferedWriter bw) {
			    super(model);
			    output = bw;
			  }

			  public void writeUTF (String s) throws java.io.IOException {
			    output.write(s);
			    output.write(this.separator);
			  }

			  public void writeInt (int i) throws java.io.IOException {
			    output.write(Integer.toString(i));
			    output.write(this.separator);
			  }

			  public void writeDouble (double d) throws java.io.IOException {
			    output.write(Double.toString(d));
			    output.write(this.separator);
			  }

			  public void close () throws java.io.IOException {
			    output.flush();
			    output.close();
			  }
			  
			  public String toString(){
				  StringBuffer sb = sw.getBuffer();
				  return sb.toString();
			  }
}

