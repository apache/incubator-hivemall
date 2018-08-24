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
package hivemall.optimizer;

import hivemall.utils.math.MathUtils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

/**
 * @link https://github.com/JohnLangford/vowpal_wabbit/wiki/Loss-functions
 */
public final class LossFunctions {

    public enum LossType {
        SquaredLoss, QuantileLoss, EpsilonInsensitiveLoss, SquaredEpsilonInsensitiveLoss, HuberLoss,
        HingeLoss, LogLoss, SquaredHingeLoss, ModifiedHuberLoss
    }

    @Nonnull
    public static LossFunction getLossFunction(@Nullable final String type) {
        if ("SquaredLoss".equalsIgnoreCase(type) || "squared".equalsIgnoreCase(type)) {
            return new SquaredLoss();
        } else if ("QuantileLoss".equalsIgnoreCase(type) || "quantile".equalsIgnoreCase(type)) {
            return new QuantileLoss();
        } else if ("EpsilonInsensitiveLoss".equalsIgnoreCase(type)
                || "epsilon_insensitive".equalsIgnoreCase(type)) {
            return new EpsilonInsensitiveLoss();
        } else if ("SquaredEpsilonInsensitiveLoss".equalsIgnoreCase(type)
                || "squared_epsilon_insensitive".equalsIgnoreCase(type)) {
            return new SquaredEpsilonInsensitiveLoss();
        } else if ("HuberLoss".equalsIgnoreCase(type) || "huber".equalsIgnoreCase(type)) {
            return new HuberLoss();
        } else if ("HingeLoss".equalsIgnoreCase(type) || "hinge".equalsIgnoreCase(type)) {
            return new HingeLoss();
        } else if ("LogLoss".equalsIgnoreCase(type) || "log".equalsIgnoreCase(type)
                || "LogisticLoss".equalsIgnoreCase(type) || "logistic".equalsIgnoreCase(type)) {
            return new LogLoss();
        } else if ("SquaredHingeLoss".equalsIgnoreCase(type)
                || "squared_hinge".equalsIgnoreCase(type)) {
            return new SquaredHingeLoss();
        } else if ("ModifiedHuberLoss".equalsIgnoreCase(type)
                || "modified_huber".equalsIgnoreCase(type)) {
            return new ModifiedHuberLoss();
        }
        throw new IllegalArgumentException("Unsupported loss function name: " + type);
    }

    @Nonnull
    public static LossFunction getLossFunction(@Nonnull final LossType type) {
        switch (type) {
            case SquaredLoss:
                return new SquaredLoss();
            case QuantileLoss:
                return new QuantileLoss();
            case EpsilonInsensitiveLoss:
                return new EpsilonInsensitiveLoss();
            case SquaredEpsilonInsensitiveLoss:
                return new SquaredEpsilonInsensitiveLoss();
            case HuberLoss:
                return new HuberLoss();
            case HingeLoss:
                return new HingeLoss();
            case LogLoss:
                return new LogLoss();
            case SquaredHingeLoss:
                return new SquaredHingeLoss();
            case ModifiedHuberLoss:
                return new ModifiedHuberLoss();
            default:
                throw new IllegalArgumentException("Unsupported loss function name: " + type);
        }
    }

    public interface LossFunction {

        /**
         * Evaluate the loss function.
         *
         * @param p The prediction, p = w^T x
         * @param y The true value (aka target)
         * @return The loss evaluated at `p` and `y`.
         */
        public float loss(float p, float y);

        public double loss(double p, double y);

        /**
         * Evaluate the derivative of the loss function with respect to the prediction `p`.
         *
         * @param p The prediction, p = w^T x
         * @param y The true value (aka target)
         * @return The derivative of the loss function w.r.t. `p`.
         */
        public float dloss(float p, float y);

        public boolean forBinaryClassification();

        public boolean forRegression();

        @Nonnull
        public LossType getType();

    }

    public static abstract class RegressionLoss implements LossFunction {

        @Override
        public boolean forBinaryClassification() {
            return false;
        }

        @Override
        public boolean forRegression() {
            return true;
        }
    }

    public static abstract class BinaryLoss implements LossFunction {

        protected static void checkTarget(final float y) {
            if (!(y == 1.f || y == -1.f)) {
                throw new IllegalArgumentException("target must be [+1,-1]: " + y);
            }
        }

        protected static void checkTarget(final double y) {
            if (!(y == 1.d || y == -1.d)) {
                throw new IllegalArgumentException("target must be [+1,-1]: " + y);
            }
        }

        @Override
        public boolean forBinaryClassification() {
            return true;
        }

        @Override
        public boolean forRegression() {
            return false;
        }
    }

    /**
     * Squared loss for regression problems.
     *
     * If you're trying to minimize the mean error, use squared-loss.
     */
    public static final class SquaredLoss extends RegressionLoss {

        @Override
        public float loss(final float p, final float y) {
            final float z = p - y;
            return z * z * 0.5f;
        }

        @Override
        public double loss(final double p, final double y) {
            final double z = p - y;
            return z * z * 0.5d;
        }

        @Override
        public float dloss(final float p, final float y) {
            return p - y; // 2 (p - y) / 2
        }

        @Override
        public LossType getType() {
            return LossType.SquaredLoss;
        }
    }

    /**
     * Quantile loss is useful to predict rank/order and you do not mind the mean error to increase
     * as long as you get the relative order correct.
     *
     * @link http://en.wikipedia.org/wiki/Quantile_regression
     */
    public static final class QuantileLoss extends RegressionLoss {

        private float tau;

        public QuantileLoss() {
            this.tau = 0.5f;
        }

        public QuantileLoss(float tau) {
            setTau(tau);
        }

        public void setTau(float tau) {
            if (tau <= 0 || tau >= 1.0) {
                throw new IllegalArgumentException("tau must be in range (0, 1): " + tau);
            }
            this.tau = tau;
        }

        @Override
        public float loss(final float p, final float y) {
            float e = y - p;
            if (e > 0.f) {
                return tau * e;
            } else {
                return -(1.f - tau) * e;
            }
        }

        @Override
        public double loss(final double p, final double y) {
            double e = y - p;
            if (e > 0.d) {
                return tau * e;
            } else {
                return -(1.d - tau) * e;
            }
        }

        @Override
        public float dloss(final float p, final float y) {
            float e = y - p;
            if (e == 0.f) {
                return 0.f;
            }
            return (e > 0.f) ? -tau : (1.f - tau);
        }

        @Override
        public LossType getType() {
            return LossType.QuantileLoss;
        }
    }

    /**
     * Epsilon-Insensitive loss used by Support Vector Regression (SVR).
     * <code>loss = max(0, |y - p| - epsilon)</code>
     */
    public static final class EpsilonInsensitiveLoss extends RegressionLoss {

        private float epsilon;

        public EpsilonInsensitiveLoss() {
            this(0.1f);
        }

        public EpsilonInsensitiveLoss(float epsilon) {
            this.epsilon = epsilon;
        }

        public void setEpsilon(float epsilon) {
            this.epsilon = epsilon;
        }

        @Override
        public float loss(final float p, final float y) {
            float loss = Math.abs(y - p) - epsilon;
            return (loss > 0.f) ? loss : 0.f;
        }

        @Override
        public double loss(final double p, final double y) {
            double loss = Math.abs(y - p) - epsilon;
            return (loss > 0.d) ? loss : 0.d;
        }

        @Override
        public float dloss(final float p, final float y) {
            if ((y - p) > epsilon) {// real value > predicted value - epsilon
                return -1.f;
            } else if ((p - y) > epsilon) {// real value < predicted value - epsilon
                return 1.f;
            } else {
                return 0.f;
            }
        }

        @Override
        public LossType getType() {
            return LossType.EpsilonInsensitiveLoss;
        }
    }

    /**
     * Squared Epsilon-Insensitive loss. <code>loss = max(0, |y - p| - epsilon)^2</code>
     */
    public static final class SquaredEpsilonInsensitiveLoss extends RegressionLoss {

        private float epsilon;

        public SquaredEpsilonInsensitiveLoss() {
            this(0.1f);
        }

        public SquaredEpsilonInsensitiveLoss(float epsilon) {
            this.epsilon = epsilon;
        }

        public void setEpsilon(float epsilon) {
            this.epsilon = epsilon;
        }

        @Override
        public float loss(final float p, final float y) {
            float d = Math.abs(y - p) - epsilon;
            return (d > 0.f) ? (d * d) : 0.f;
        }

        @Override
        public double loss(final double p, final double y) {
            double d = Math.abs(y - p) - epsilon;
            return (d > 0.d) ? (d * d) : 0.d;
        }

        @Override
        public float dloss(final float p, final float y) {
            final float z = y - p;
            if (z > epsilon) {
                return -2 * (z - epsilon);
            } else if (-z > epsilon) {
                return 2 * (-z - epsilon);
            } else {
                return 0.f;
            }
        }

        @Override
        public LossType getType() {
            return LossType.SquaredEpsilonInsensitiveLoss;
        }
    }

    /**
     * Huber regression loss.
     *
     * Variant of the SquaredLoss which is robust to outliers.
     *
     * @link https://en.wikipedia.org/wiki/Huber_Loss_Function
     */
    public static final class HuberLoss extends RegressionLoss {

        private float c;

        public HuberLoss() {
            this(1.f); // i.e., beyond 1 standard deviation, the loss becomes linear
        }

        public HuberLoss(float c) {
            this.c = c;
        }

        public void setC(float c) {
            this.c = c;
        }

        @Override
        public float loss(final float p, final float y) {
            final float r = p - y;
            final float rAbs = Math.abs(r);
            if (rAbs <= c) {
                return 0.5f * r * r;
            }
            return c * rAbs - (0.5f * c * c);
        }

        @Override
        public double loss(final double p, final double y) {
            final double r = p - y;
            final double rAbs = Math.abs(r);
            if (rAbs <= c) {
                return 0.5d * r * r;
            }
            return c * rAbs - (0.5d * c * c);
        }

        @Override
        public float dloss(final float p, final float y) {
            final float r = p - y;
            final float rAbs = Math.abs(r);
            if (rAbs <= c) {
                return r;
            } else if (r > 0.f) {
                return c;
            }
            return -c;
        }

        @Override
        public LossType getType() {
            return LossType.HuberLoss;
        }
    }

    /**
     * Hinge loss for binary classification tasks with y in {-1,1}.
     */
    public static final class HingeLoss extends BinaryLoss {

        private float threshold;

        public HingeLoss() {
            this(1.f);
        }

        /**
         * @param threshold Margin threshold. When threshold=1.0, one gets the loss used by SVM.
         *        When threshold=0.0, one gets the loss used by the Perceptron.
         */
        public HingeLoss(float threshold) {
            this.threshold = threshold;
        }

        public void setThreshold(float threshold) {
            this.threshold = threshold;
        }

        @Override
        public float loss(final float p, final float y) {
            float loss = hingeLoss(p, y, threshold);
            return (loss > 0.f) ? loss : 0.f;
        }

        @Override
        public double loss(final double p, final double y) {
            double loss = hingeLoss(p, y, threshold);
            return (loss > 0.d) ? loss : 0.d;
        }

        @Override
        public float dloss(final float p, final float y) {
            float loss = hingeLoss(p, y, threshold);
            return (loss > 0.f) ? -y : 0.f;
        }

        @Override
        public LossType getType() {
            return LossType.HingeLoss;
        }
    }

    /**
     * Logistic regression loss for binary classification with y in {-1, 1}.
     */
    public static final class LogLoss extends BinaryLoss {

        /**
         * <code>logloss(p,y) = log(1+exp(-p*y))</code>
         */
        @Override
        public float loss(final float p, final float y) {
            checkTarget(y);

            final float z = y * p;
            if (z > 18.f) {
                return (float) Math.exp(-z);
            }
            if (z < -18.f) {
                return -z;
            }
            return (float) Math.log(1.d + Math.exp(-z));
        }

        @Override
        public double loss(final double p, final double y) {
            checkTarget(y);

            final double z = y * p;
            if (z > 18.d) {
                return Math.exp(-z);
            }
            if (z < -18.d) {
                return -z;
            }
            return Math.log(1.d + Math.exp(-z));
        }

        @Override
        public float dloss(final float p, final float y) {
            checkTarget(y);

            float z = y * p;
            if (z > 18.f) {
                return (float) Math.exp(-z) * -y;
            }
            if (z < -18.f) {
                return -y;
            }
            return -y / ((float) Math.exp(z) + 1.f);
        }

        @Override
        public LossType getType() {
            return LossType.LogLoss;
        }
    }

    /**
     * Squared Hinge loss for binary classification tasks with y in {-1,1}.
     */
    public static final class SquaredHingeLoss extends BinaryLoss {

        @Override
        public float loss(final float p, final float y) {
            return squaredHingeLoss(p, y);
        }

        @Override
        public double loss(final double p, final double y) {
            return squaredHingeLoss(p, y);
        }

        @Override
        public float dloss(final float p, final float y) {
            checkTarget(y);

            float d = 1 - (y * p);
            return (d > 0.f) ? -2.f * d * y : 0.f;
        }

        @Override
        public LossType getType() {
            return LossType.SquaredHingeLoss;
        }
    }

    /**
     * Modified Huber loss for binary classification with y in {-1, 1}.
     *
     * Equivalent to quadratically smoothed SVM with gamma = 2.
     */
    public static final class ModifiedHuberLoss extends BinaryLoss {

        @Override
        public float loss(final float p, final float y) {
            final float z = p * y;
            if (z >= 1.f) {
                return 0.f;
            } else if (z >= -1.f) {
                return (1.f - z) * (1.f - z);
            }
            return -4.f * z;
        }

        @Override
        public double loss(final double p, final double y) {
            final double z = p * y;
            if (z >= 1.d) {
                return 0.d;
            } else if (z >= -1.d) {
                return (1.d - z) * (1.d - z);
            }
            return -4.d * z;
        }

        @Override
        public float dloss(final float p, final float y) {
            final float z = p * y;
            if (z >= 1.f) {
                return 0.f;
            } else if (z >= -1.f) {
                return 2.f * (1.f - z) * -y;
            }
            return -4.f * y;
        }

        @Override
        public LossType getType() {
            return LossType.ModifiedHuberLoss;
        }
    }

    /**
     * logistic loss function where target is 0 (negative) or 1 (positive).
     */
    public static float logisticLoss(final float target, final float predicted) {
        if (predicted > -100.d) {
            return target - (float) MathUtils.sigmoid(predicted);
        } else {
            return target;
        }
    }

    public static float logLoss(final float p, final float y) {
        BinaryLoss.checkTarget(y);

        final float z = y * p;
        if (z > 18.f) {
            return (float) Math.exp(-z);
        }
        if (z < -18.f) {
            return -z;
        }
        return (float) Math.log(1.d + Math.exp(-z));
    }

    public static double logLoss(final double p, final double y) {
        BinaryLoss.checkTarget(y);

        final double z = y * p;
        if (z > 18.d) {
            return Math.exp(-z);
        }
        if (z < -18.d) {
            return -z;
        }
        return Math.log(1.d + Math.exp(-z));
    }

    public static float squaredLoss(final float p, final float y) {
        final float z = p - y;
        return z * z * 0.5f;
    }

    public static double squaredLoss(final double p, final double y) {
        final double z = p - y;
        return z * z * 0.5d;
    }

    public static float hingeLoss(final float p, final float y, final float threshold) {
        BinaryLoss.checkTarget(y);

        float z = y * p;
        return threshold - z;
    }

    public static double hingeLoss(final double p, final double y, final double threshold) {
        BinaryLoss.checkTarget(y);

        double z = y * p;
        return threshold - z;
    }

    public static float hingeLoss(final float p, final float y) {
        return hingeLoss(p, y, 1.f);
    }

    public static double hingeLoss(final double p, final double y) {
        return hingeLoss(p, y, 1.d);
    }

    public static float squaredHingeLoss(final float p, final float y) {
        BinaryLoss.checkTarget(y);

        float z = y * p;
        float d = 1.f - z;
        return (d > 0.f) ? (d * d) : 0.f;
    }

    public static double squaredHingeLoss(final double p, final double y) {
        BinaryLoss.checkTarget(y);

        double z = y * p;
        double d = 1.d - z;
        return (d > 0.d) ? d * d : 0.d;
    }

    /**
     * Math.abs(target - predicted) - epsilon
     */
    public static float epsilonInsensitiveLoss(final float predicted, final float target,
            final float epsilon) {
        return Math.abs(target - predicted) - epsilon;
    }
}
