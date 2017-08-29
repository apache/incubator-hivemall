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
package hivemall.fm;

import hivemall.fm.FMHyperParameters.FFMHyperParameters;
import hivemall.utils.collections.arrays.DoubleArray3D;
import hivemall.utils.collections.lists.IntArrayList;
import hivemall.utils.lang.NumberUtils;
import hivemall.utils.math.MathUtils;

import java.util.Arrays;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.metadata.HiveException;

public abstract class FieldAwareFactorizationMachineModel extends FactorizationMachineModel {

    @Nonnull
    protected final FFMHyperParameters _params;
    protected final float _eta0;
    protected final float _eps;

    protected final boolean _useAdaGrad;
    protected final boolean _useFTRL;

    // FTEL
    private final float _alpha;
    private final float _beta;
    private final float _lambda1;
    private final float _lamdda2;

    public FieldAwareFactorizationMachineModel(@Nonnull FFMHyperParameters params) {
        super(params);
        this._params = params;
        if (params.useAdaGrad) {
            this._eta0 = 1.0f;
        } else {
            this._eta0 = params.eta.eta0();
        }
        this._eps = params.eps;
        this._useAdaGrad = params.useAdaGrad;
        this._useFTRL = params.useFTRL;
        this._alpha = params.alphaFTRL;
        this._beta = params.betaFTRL;
        this._lambda1 = params.lambda1;
        this._lamdda2 = params.lamdda2;
    }

    public abstract float getV(@Nonnull Feature x, @Nonnull int yField, int f);

    @Deprecated
    protected abstract void setV(@Nonnull Feature x, @Nonnull int yField, int f, float nextVif);

    @Override
    public float getV(Feature x, int f) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected void setV(Feature x, int f, float nextVif) {
        throw new UnsupportedOperationException();
    }

    @Override
    protected final double predict(@Nonnull final Feature[] x) throws HiveException {
        // w0
        double ret = getW0();
        // W
        for (Feature e : x) {
            double xi = e.getValue();
            float wi = getW(e);
            double wx = wi * xi;
            ret += wx;
        }
        // V
        for (int i = 0; i < x.length; i++) {
            final Feature ei = x[i];
            final double xi = ei.getValue();
            final int iField = ei.getField();
            for (int j = i + 1; j < x.length; j++) {
                final Feature ej = x[j];
                final double xj = ej.getValue();
                final int jField = ej.getField();
                for (int f = 0, k = _factor; f < k; f++) {
                    float vijf = getV(ei, jField, f);
                    float vjif = getV(ej, iField, f);
                    ret += vijf * vjif * xi * xj;
                    assert (!Double.isNaN(ret));
                }
            }
        }
        if (!NumberUtils.isFinite(ret)) {
            throw new HiveException("Detected " + ret
                    + " in predict. We recommend to normalize training examples.\n"
                    + "Dumping variables ...\n" + varDump(x));
        }
        return ret;
    }

    void updateWi(final double dloss, @Nonnull final Feature x, final long t) {
        if (_useFTRL) {
            updateWi_FTRL(dloss, x);
            return;
        }

        final double Xi = x.getValue();
        float gradWi = (float) (dloss * Xi);

        final Entry theta = getEntryW(x);
        float wi = theta.getW();

        final float eta = eta(theta, t, gradWi);
        float nextWi = wi - eta * (gradWi + 2.f * _lambdaW * wi);
        if (!NumberUtils.isFinite(nextWi)) {
            throw new IllegalStateException("Got " + nextWi + " for next W[" + x.getFeature()
                    + "]\n" + "Xi=" + Xi + ", gradWi=" + gradWi + ", wi=" + wi + ", dloss=" + dloss
                    + ", eta=" + eta + ", t=" + t);
        }
        if (MathUtils.closeToZero(nextWi, 1E-9f)) {
            removeEntry(theta);
            return;
        }
        theta.setW(nextWi);
    }

    /**
     * Update Wi using Follow-the-Regularized-Leader
     */
    private void updateWi_FTRL(final double dloss, @Nonnull final Feature x) {
        final double Xi = x.getValue();
        float gradWi = (float) (dloss * Xi);

        final Entry theta = getEntryW(x);

        final float z = theta.updateZ(gradWi, _alpha);
        final double n = theta.updateN(gradWi);

        if (Math.abs(z) <= _lambda1) {
            removeEntry(theta);
            return;
        }

        final float nextWi = (float) ((MathUtils.sign(z) * _lambda1 - z) / ((_beta + Math.sqrt(n))
                / _alpha + _lamdda2));
        if (!NumberUtils.isFinite(nextWi)) {
            throw new IllegalStateException("Got " + nextWi + " for next W[" + x.getFeature()
                    + "]\n" + "Xi=" + Xi + ", gradWi=" + gradWi + ", wi=" + theta.getW()
                    + ", dloss=" + dloss + ", n=" + n + ", z=" + z);
        }
        if (MathUtils.closeToZero(nextWi, 1E-9f)) {
            removeEntry(theta);
            return;
        }
        theta.setW(nextWi);
    }

    protected abstract void removeEntry(@Nonnull final Entry entry);

    void updateV(final double dloss, @Nonnull final Feature x, @Nonnull final int yField,
            final int f, final double sumViX, long t) {
        if (_useFTRL) {
            updateV_FTRL(dloss, x, yField, f, sumViX);
            return;
        }

        final Entry theta = getEntryV(x, yField);
        if (theta == null) {
            return;
        }

        final double Xi = x.getValue();
        final double h = Xi * sumViX;
        final float gradV = (float) (dloss * h);
        final float lambdaVf = getLambdaV(f);

        final float currentV = theta.getV(f);
        final float eta = eta(theta, f, t, gradV);
        final float nextV = currentV - eta * (gradV + 2.f * lambdaVf * currentV);
        if (!NumberUtils.isFinite(nextV)) {
            throw new IllegalStateException("Got " + nextV + " for next V" + f + '['
                    + x.getFeatureIndex() + "]\n" + "Xi=" + Xi + ", Vif=" + currentV + ", h=" + h
                    + ", gradV=" + gradV + ", lambdaVf=" + lambdaVf + ", dloss=" + dloss
                    + ", sumViX=" + sumViX + ", t=" + t);
        }
        if (MathUtils.closeToZero(nextV, 1E-9f)) {
            theta.setV(f, 0.f);
            if (theta.removable()) { // Whether other factors are zero filled or not? Remove if zero filled
                removeEntry(theta);
            }
            return;
        }
        theta.setV(f, nextV);
    }

    private void updateV_FTRL(final double dloss, @Nonnull final Feature x,
            @Nonnull final int yField, final int f, final double sumViX) {
        final Entry theta = getEntryV(x, yField);
        if (theta == null) {
            return;
        }

        final double Xi = x.getValue();
        final double h = Xi * sumViX;
        final float gradV = (float) (dloss * h);

        float oldV = theta.getV(f);
        final float z = theta.updateZ(f, oldV, gradV, _alpha);
        final double n = theta.updateN(f, gradV);

        if (Math.abs(z) <= _lambda1) {
            theta.setV(f, 0.f);
            if (theta.removable()) { // Whether other factors are zero filled or not? Remove if zero filled
                removeEntry(theta);
            }
            return;
        }

        final float nextV = (float) ((MathUtils.sign(z) * _lambda1 - z) / ((_beta + Math.sqrt(n))
                / _alpha + _lamdda2));
        if (!NumberUtils.isFinite(nextV)) {
            throw new IllegalStateException("Got " + nextV + " for next V" + f + '['
                    + x.getFeatureIndex() + "]\n" + "Xi=" + Xi + ", Vif=" + theta.getV(f) + ", h="
                    + h + ", gradV=" + gradV + ", dloss=" + dloss + ", sumViX=" + sumViX + ", n="
                    + n + ", z=" + z);
        }
        if (MathUtils.closeToZero(nextV, 1E-9f)) {
            theta.setV(f, 0.f);
            if (theta.removable()) { // Whether other factors are zero filled or not? Remove if zero filled
                removeEntry(theta);
            }
            return;
        }
        theta.setV(f, nextV);
    }

    protected final float eta(@Nonnull final Entry theta, final long t, final float grad) {
        return eta(theta, 0, t, grad);
    }

    protected final float eta(@Nonnull final Entry theta, @Nonnegative final int f, final long t,
            final float grad) {
        if (_useAdaGrad) {
            double gg = theta.getSumOfSquaredGradients(f);
            theta.addGradient(f, grad);
            return (float) (_eta0 / Math.sqrt(_eps + gg));
        } else {
            return _eta.eta(t);
        }
    }

    /**
     * sum{XiViaf} where a is field index of Xi
     */
    @Nonnull
    final DoubleArray3D sumVfX(@Nonnull final Feature[] x, @Nonnull final IntArrayList fieldList,
            @Nullable DoubleArray3D cached) {
        final int xSize = x.length;
        final int fieldSize = fieldList.size();
        final int factors = _factor;

        final DoubleArray3D mdarray;
        if (cached == null) {
            mdarray = new DoubleArray3D();
            mdarray.setSanityCheck(false);
        } else {
            mdarray = cached;
        }
        mdarray.configure(xSize, fieldSize, factors);

        for (int i = 0; i < xSize; i++) {
            for (int fieldIndex = 0; fieldIndex < fieldSize; fieldIndex++) {
                final int yField = fieldList.get(fieldIndex);
                for (int f = 0; f < factors; f++) {
                    double val = sumVfX(x, i, yField, f);
                    mdarray.set(i, fieldIndex, f, val);
                }
            }
        }

        return mdarray;
    }

    private double sumVfX(@Nonnull final Feature[] x, final int i, @Nonnull final int yField,
            final int f) {
        final Feature xi = x[i];
        final int xiFeature = xi.getFeatureIndex();
        final double xiValue = xi.getValue();
        final int xiField = xi.getField();
        double ret = 0.d;
        // find all other features whose field matches field
        for (Feature e : x) {
            if (e.getFeatureIndex() == xiFeature) { // ignore x[i] = e
                continue;
            }
            if (e.getField() == yField) { // multiply x_e and v_d,field(e),f
                float Vjf = getV(e, xiField, f);
                ret += Vjf * xiValue;
            }
        }
        if (!NumberUtils.isFinite(ret)) {
            throw new IllegalStateException("Got " + ret + " for sumV[ " + i + "][ " + f + "]X.\n"
                    + "x = " + Arrays.toString(x));
        }
        return ret;
    }

    @Nonnull
    protected abstract Entry getEntryW(@Nonnull Feature x);

    @Nullable
    protected abstract Entry getEntryV(@Nonnull Feature x, @Nonnull int yField);

    @Override
    protected final String varDump(@Nonnull final Feature[] x) {
        final StringBuilder buf1 = new StringBuilder(1024);
        final StringBuilder buf2 = new StringBuilder(1024);

        // X
        for (int i = 0; i < x.length; i++) {
            Feature e = x[i];
            String j = e.getFeature();
            double xj = e.getValue();
            if (i != 0) {
                buf1.append(", ");
            }
            buf1.append("x[").append(j).append("] = ").append(xj);
        }
        buf1.append("\n");

        // w0        
        double ret = getW0();
        buf1.append("predict(x) = w0");
        buf2.append("predict(x) = ").append(ret);

        // W
        for (Feature e : x) {
            String i = e.getFeature();
            double xi = e.getValue();
            float wi = getW(e);

            buf1.append(" + (w[").append(i).append("] * x[").append(i).append("])");
            buf2.append(" + (").append(wi).append(" * ").append(xi).append(')');

            double wx = wi * xi;
            ret += wx;

            if (!NumberUtils.isFinite(ret)) {
                return buf1.append(" + ... = ")
                           .append(ret)
                           .append('\n')
                           .append(buf2)
                           .append(" + ... = ")
                           .append(ret)
                           .toString();
            }
        }

        // V
        for (int i = 0; i < x.length; i++) {
            final Feature ei = x[i];
            final String fi = ei.getFeature();
            final double xi = ei.getValue();
            final int iField = ei.getField();
            for (int j = i + 1; j < x.length; j++) {
                final Feature ej = x[j];
                final String fj = ej.getFeature();
                final double xj = ej.getValue();
                final int jField = ej.getField();
                for (int f = 0, k = _factor; f < k; f++) {
                    float vijf = getV(ei, jField, f);
                    float vjif = getV(ej, iField, f);

                    buf1.append(" + (v[i")
                        .append(fi)
                        .append("-j")
                        .append(jField)
                        .append("-f")
                        .append(f)
                        .append("] * v[j")
                        .append(fj)
                        .append("-i")
                        .append(iField)
                        .append("-f")
                        .append(f)
                        .append("] * x[")
                        .append(fi)
                        .append("] * x[")
                        .append(fj)
                        .append("])");

                    buf2.append(" + (")
                        .append(vijf)
                        .append(" * ")
                        .append(vjif)
                        .append(" * ")
                        .append(xi)
                        .append(" * ")
                        .append(xj)
                        .append(')');

                    ret += vijf * vjif * xi * xj;

                    if (!NumberUtils.isFinite(ret)) {
                        return buf1.append(" + ... = ")
                                   .append(ret)
                                   .append('\n')
                                   .append(buf2)
                                   .append(" + ... = ")
                                   .append(ret)
                                   .toString();
                    }
                }
            }
        }

        return buf1.append(" = ")
                   .append(ret)
                   .append('\n')
                   .append(buf2)
                   .append(" = ")
                   .append(ret)
                   .toString();
    }
}
