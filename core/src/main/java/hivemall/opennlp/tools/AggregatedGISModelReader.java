/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *   http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package hivemall.opennlp.tools;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import opennlp.maxent.GISModel;
import opennlp.model.AbstractModel;
import opennlp.model.Context;
import opennlp.model.DataReader;

public class AggregatedGISModelReader extends AbstractAggregatedModelReader {

    int correctionConstant;
    double correctionParam;
    List<String> allOutcomeLabels = new ArrayList<String>();
    Map<String, List<Context>> allPredLabels = new HashMap<String, List<Context>>();

    public AggregatedGISModelReader(List<DataReader> dataReaders) {
        super(dataReaders);
    }

    /**
     * Retrieve a model from disk. It assumes that models are saved in the following sequence:
     * 
     * <br>
     * GIS (model type identifier) <br>
     * 1. # of parameters (int) <br>
     * 2. the correction constant (int) <br>
     * 3. the correction constant parameter (double) <br>
     * 4. # of outcomes (int) <br>
     * * list of outcome names (String) <br>
     * 5. # of different types of outcome patterns (int) <br>
     * * list of (int int[]) <br>
     * [# of predicates for which outcome pattern is true] [outcome pattern] <br>
     * 6. # of predicates (int) <br>
     * * list of predicate names (String)
     *
     * <p>
     * If you are creating a reader for a format which won't work with this (perhaps a database or xml file), override this method and ignore the
     * other methods provided in this abstract class.
     *
     * @return The GISModel stored in the format and location specified to this GISModelReader (usually via its the constructor).
     */
    public AbstractModel constructModel() throws IOException {
        correctionConstant = getCorrectionConstant();
        correctionParam = getCorrectionParameter();
        for (int d = 0; d < dataReaders.size(); d++) {
            String[] modelOutcomeLabels = getOutcomes(d);

            for (String outcomeLabel : modelOutcomeLabels) {
                if (!this.allOutcomeLabels.contains(outcomeLabel)) {
                    this.allOutcomeLabels.add(outcomeLabel);
                }
            }

            int[][] outcomePatterns = getOutcomePatterns(d);
            String[] predLabels = getPredicates(d);
            Context[] params = getParameters(d, outcomePatterns);

            for (int pl = 0; pl < predLabels.length; pl++) {
                Context c = params[pl];
                int[] newIndexes = new int[c.getOutcomes().length];
                for (int ci = 0; ci < c.getOutcomes().length; ci++) {
                    newIndexes[ci] = this.allOutcomeLabels.indexOf(modelOutcomeLabels[c.getOutcomes()[ci]]);
                }

                Context newerC = new Context(newIndexes, c.getParameters());

                if (this.allPredLabels.containsKey(predLabels[pl])) {
                    this.allPredLabels.get(predLabels[pl]).add(newerC);
                } else {
                    List<Context> contexts = new LinkedList<Context>();
                    contexts.add(newerC);
                    this.allPredLabels.put(predLabels[pl], contexts);
                }
            }

        }
        // end of going through models
        String[] outcomeLabels = new String[this.allOutcomeLabels.size()];
        outcomeLabels = this.allOutcomeLabels.toArray(outcomeLabels);
        String[] predLabels = new String[this.allPredLabels.size()];
        Context[] params = new Context[this.allPredLabels.size()];

        int paramIndex = 0;
        for (String l : this.allPredLabels.keySet()) {
            List<Context> contexts = this.allPredLabels.get(l);
            Map<Integer, List<Double>> finalContext = new HashMap<Integer, List<Double>>();

            for (Context c : contexts) {
                int[] outcomes = c.getOutcomes();
                for (int outcomeIndex = 0; outcomeIndex < outcomes.length; outcomeIndex++) {
                    String outcomeLabel = this.allOutcomeLabels.get(outcomes[outcomeIndex]);
                    int index = 0;
                    for (int ol = 0; ol < outcomeLabels.length; ol++) {
                        if (outcomeLabels[ol].equals(outcomeLabel)) {
                            index = ol;
                        }
                    }
                    if (finalContext.containsKey(index)) {
                        finalContext.get(index).add(c.getParameters()[outcomeIndex]);
                    } else {
                        List<Double> parameters = new LinkedList<Double>();
                        parameters.add(c.getParameters()[outcomeIndex]);
                        finalContext.put(index, parameters);
                    }
                }
            }

            int[] labelOutcomes = new int[finalContext.size()];
            double[] labelParameters = new double[finalContext.size()];

            int index = 0;
            for (Entry<Integer, List<Double>> entry : finalContext.entrySet()) {
                labelOutcomes[index] = entry.getKey();

                double sum = 0;
                for (Double d : entry.getValue()) {
                    sum += d;
                }
                labelParameters[index] = sum / (double) entry.getValue().size();
                index++;
            }

            Context c = new Context(labelOutcomes, labelParameters);
            predLabels[paramIndex] = l;
            params[paramIndex] = c;
            paramIndex++;
        }

        return new GISModel(params, predLabels, outcomeLabels, correctionConstant, correctionParam);
    }

    public void checkModelType() throws java.io.IOException {
        for (int d = 0; d < dataReaders.size(); d++) {
            String modelType = readUTF(d);
            if (!modelType.equals("GIS"))
                System.out.println("Error: attempting to load a " + modelType
                        + " model as a GIS model." + " You should expect problems.");
        }
    }

    protected int getCorrectionConstant() throws java.io.IOException {
        int correctionConstant = readInt(0);
        for (int d = 1; d < dataReaders.size(); d++) {
            int constant = readInt(d);
            if (correctionConstant != constant) {
                System.out.println("Error: distinct correction constants.");
            }
        }

        return correctionConstant;
    }

    protected double getCorrectionParameter() throws java.io.IOException {
        double correctionParameter = readDouble(0);
        for (int d = 1; d < dataReaders.size(); d++) {
            double parameter = readDouble(d);
            correctionParameter += parameter;
        }

        return correctionParameter / (double) dataReaders.size();
    }
}
