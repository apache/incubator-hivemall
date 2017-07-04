package hivemall.smile.tools;

import java.io.FileNotFoundException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.UDAF;
import org.apache.hadoop.hive.ql.exec.UDAFEvaluator;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.io.Text;

import opennlp.maxent.io.GISModelWriter;
import opennlp.model.AbstractModel;
import opennlp.model.DataReader;
import opennlp.model.MaxentModel;

public class MaxEntMixtureWeightUDAF extends UDAF{
	public static class DdiffEvaluator implements UDAFEvaluator {

		public static class partial_result{
			public List<Text> models = new ArrayList<Text>(); 
			
			public void add(Text model){
				models.add(model);
			}
		}
		
		private partial_result partialResult;
		
		@Override
		public void init() {
			partialResult = null;
		}
		
		public boolean iterate(String model){
			if (partialResult == null){
				partialResult = new partial_result();
			}
			
			partialResult.add(new Text(model));
	        return true;
		}
		
		public partial_result terminatePartial(){
			return partialResult;
		}
		
		public boolean merge(partial_result other){
			
			if (other == null){
				return true;
			}
			
			if (partialResult == null){
				partialResult = new partial_result();
			}
			
			for (Text diff : other.models){
				partialResult.add(diff);
			}
			return true;
		}
		
		public String terminate() throws HiveException {
			if (partialResult == null || partialResult.models.isEmpty()){
				return null;
			}
			
			MaxentModel mixedWeightModel;
			try {
				mixedWeightModel = new SepDelimitedTextGISModelReader(partialResult.models.get(0)).constructModel();
			} catch (IOException e1) {
				throw new HiveException(e1.getMessage());
			}
			
			if (partialResult.models.size() == 1){
				GISModelWriter writer;
				try {
					writer = new SepDelimitedTextGISModelWriter((AbstractModel)mixedWeightModel, "@");
					writer.persist();
				} catch (FileNotFoundException e) {
					throw new HiveException(e.getMessage());
				} catch (IOException e) {
					throw new HiveException(e.getMessage());
				}
				
				return writer.toString();
			}
			
			List<DataReader> readers = new LinkedList<DataReader>();
			
			for (Text model : partialResult.models){
				readers.add(new SepDelimitedTextDataReader(model));
			}
			
			AggregatedGISModelReader aggregatedReader = new AggregatedGISModelReader(readers);
			AbstractModel aggregatedModel;
			try {
				aggregatedModel = aggregatedReader.constructModel();
			} catch (IOException e) {
				throw new HiveException(e.getMessage());
			}
			
			GISModelWriter writer;
			try {
				writer = new SepDelimitedTextGISModelWriter(aggregatedModel, "@");
				writer.persist();
			} catch (FileNotFoundException e) {
				throw new HiveException(e.getMessage());
			} catch (IOException e) {
				throw new HiveException(e.getMessage());
			}
			return writer.toString();
		}
	}
}
