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
package hivemall.common;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public final class ConversionState {
    private static final Log logger = LogFactory.getLog(ConversionState.class);

    /** Whether to check conversion */
    private final boolean conversionCheck;
    /** Threshold to determine convergence */
    private final double convergenceRate;

    /** being ready to end iteration */
    private boolean readyToFinishIterations;

    /** The cumulative errors in the training */
    private double totalErrors;
    /** The cumulative losses in an iteration */
    private double currLosses, prevLosses;

    private int curIter;

    public ConversionState() {
        this(true, 0.005d);
    }

    public ConversionState(boolean conversionCheck, double convergenceRate) {
        this.conversionCheck = conversionCheck;
        this.convergenceRate = convergenceRate;
        this.readyToFinishIterations = false;
        this.totalErrors = 0.d;
        this.currLosses = 0.d;
        this.prevLosses = Double.POSITIVE_INFINITY;
        this.curIter = 1;
    }

    public double getTotalErrors() {
        return totalErrors;
    }

    public double getCumulativeLoss() {
        return currLosses;
    }

    public double getAverageLoss(@Nonnegative final long numInstances) {
        if (numInstances == 0) {
            return 0.d;
        }
        return currLosses / numInstances;
    }

    public double getPreviousLoss() {
        return prevLosses;
    }

    public void incrError(double error) {
        this.totalErrors += error;
    }

    public void incrLoss(double loss) {
        this.currLosses += loss;
    }

    public void multiplyLoss(double multi) {
        this.currLosses = currLosses * multi;
    }

    public boolean isLossIncreased() {
        return currLosses > prevLosses;
    }

    public boolean isConverged(final long observedTrainingExamples) {
        if (conversionCheck == false) {
            return false;
        }

        if (currLosses > prevLosses) {
            if (logger.isInfoEnabled()) {
                logger.info("Iteration #" + curIter + " current cumulative loss `" + currLosses
                        + "` > previous cumulative loss `" + prevLosses + '`');
            }
            this.readyToFinishIterations = false;
            return false;
        }

        final double changeRate = getChangeRate();
        if (changeRate < convergenceRate) {
            if (readyToFinishIterations) {
                // NOTE: never be true at the first iteration where prevLosses == Double.POSITIVE_INFINITY
                if (logger.isInfoEnabled()) {
                    logger.info("Training converged at " + curIter + "-th iteration!\n"
                            + getInfo(observedTrainingExamples));
                }
                return true;
            } else {
                if (logger.isInfoEnabled()) {
                    logger.info(getInfo(observedTrainingExamples));
                }
                this.readyToFinishIterations = true;
            }
        } else {
            if (logger.isInfoEnabled()) {
                logger.info(getInfo(observedTrainingExamples));
            }
            this.readyToFinishIterations = false;
        }

        return false;
    }

    double getChangeRate() {
        return (prevLosses - currLosses) / prevLosses;
    }

    public void next() {
        this.prevLosses = currLosses;
        this.currLosses = 0.d;
        this.curIter++;
    }

    public int getCurrentIteration() {
        return curIter;
    }

    @Nonnull
    public String getInfo(@Nonnegative final long observedTrainingExamples) {
        final StringBuilder buf = new StringBuilder();
        buf.append("Iteration #").append(curIter).append(" | ");
        buf.append("average loss=").append(getAverageLoss(observedTrainingExamples)).append(", ");
        buf.append("current cumulative loss=").append(currLosses).append(", ");
        buf.append("previous cumulative loss=").append(prevLosses).append(", ");
        buf.append("change rate=").append(getChangeRate()).append(", ");
        buf.append("#trainingExamples=").append(observedTrainingExamples);
        return buf.toString();
    }

}
