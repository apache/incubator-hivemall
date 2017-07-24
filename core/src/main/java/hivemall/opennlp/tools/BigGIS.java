package hivemall.opennlp.tools;

import java.io.IOException;

import opennlp.maxent.GISModel;
import opennlp.model.EventStream;
import opennlp.model.Prior;

public class BigGIS {
    /**
     * Set this to false if you don't want messages about the progress of
     * model training displayed. Alternately, you can use the overloaded
     * version of trainModel() to conditionally enable progress messages.
     */
    public static boolean PRINT_MESSAGES = true;

    /** If we are using smoothing, this is used as the "number" of
     * times we want the trainer to imagine that it saw a feature that it
     * actually didn't see.  Defaulted to 0.1.
     */
    public static double SMOOTHING_OBSERVATION = 0.1;
    



    


    
    public static GISModel trainModel(int iterations, BigDataIndexer indexer, MatrixForTraining matrix) {
    	BigGISTrainer trainer = new BigGISTrainer(true);
        trainer.setSmoothing(false);
        trainer.setSmoothingObservation(SMOOTHING_OBSERVATION);
    	return trainer.trainModel(iterations, indexer, matrix);
    }
    

    
 

    
    /**
     * Train a model using the GIS algorithm.
     * @param iterations The number of GIS iterations to perform.
     * @param indexer The object which will be used for event compilation.
     * @param printMessagesWhileTraining Determines whether training status messages are written to STDOUT.
     * @param smoothing Defines whether the created trainer will use smoothing while training the model.
     * @param modelPrior The prior distribution for the model.
     * @param cutoff The number of times a predicate must occur to be used in a model.
     * @return The newly trained model, which can be used immediately or saved
     *         to disk using an opennlp.maxent.io.GISModelWriter object.
     */
    
    /*
    public static GISModel trainModel(int iterations, BigDataIndexer indexer, boolean printMessagesWhileTraining, boolean smoothing, Prior modelPrior, int cutoff) {
      BigGISTrainer trainer = new BigGISTrainer(printMessagesWhileTraining);
      trainer.setSmoothing(smoothing);
      trainer.setSmoothingObservation(SMOOTHING_OBSERVATION);
      if (modelPrior != null) {
        return trainer.trainModel(iterations, indexer, modelPrior,cutoff);
      }
      else {
        return trainer.trainModel(iterations, indexer,cutoff);
      }
    }  
*/
}
