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
package hivemall.model;

import hivemall.utils.hashing.MurmurHash3;
import hivemall.utils.lang.Preconditions;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.io.Text;

public final class FeatureValue {

    private/* final */Object feature; // possible types: String, Text, Integer, Long
    private/* final */double value;

    public FeatureValue() {}// used for Probe

    public FeatureValue(Object f, float v) {
        this.feature = f;
        this.value = v;
    }

    public FeatureValue(Object f, double v) {
        this.feature = f;
        this.value = v;
    }

    @SuppressWarnings("unchecked")
    public <T> T getFeature() {
        return (T) feature;
    }

    public int getFeatureAsInt() {
        Preconditions.checkNotNull(feature);
        Preconditions.checkArgument(feature instanceof Integer);
        return ((Integer) feature).intValue();
    }

    public String getFeatureAsString() {
        return feature.toString();
    }

    public double getValue() {
        return value;
    }

    public float getValueAsFloat() {
        return (float) value;
    }

    public void setValue(float value) {
        this.value = value;
    }

    public void setValue(double value) {
        this.value = value;
    }

    @Nullable
    public static FeatureValue parse(final Object o) throws IllegalArgumentException {
        return parse(o, false);
    }

    @Nullable
    public static FeatureValue parse(final Object o, final boolean mhash)
            throws IllegalArgumentException {
        if (o == null) {
            return null;
        }
        String s = o.toString();
        return parse(s, mhash);
    }

    @Nullable
    public static FeatureValue parse(@Nonnull final String s) throws IllegalArgumentException {
        return parse(s, false);
    }

    @Nullable
    public static FeatureValue parse(@Nonnull final String s, final boolean mhash)
            throws IllegalArgumentException {
        final int pos = s.indexOf(':');
        if (pos == 0) {
            throw new IllegalArgumentException("Invalid feature value representation: " + s);
        }

        final Object feature;
        final double weight;
        if (pos > 0) {
            String s1 = s.substring(0, pos);
            String s2 = s.substring(pos + 1);
            feature = mhash ? Integer.valueOf(MurmurHash3.murmurhash3(s1)) : new Text(s1);
            try {
                weight = Double.parseDouble(s2);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Failed to parse a feature value: " + s, e);
            }
        } else {
            feature = mhash ? Integer.valueOf(MurmurHash3.murmurhash3(s)) : new Text(s);
            weight = 1.d;
        }
        return new FeatureValue(feature, weight);
    }

    @Nonnull
    public static FeatureValue parseFeatureAsString(@Nonnull final Text t) {
        String s = t.toString();
        return parseFeatureAsString(s);
    }

    @Nonnull
    public static FeatureValue parseFeatureAsString(@Nonnull final String s)
            throws IllegalArgumentException {
        final int pos = s.indexOf(':');
        if (pos == 0) {
            throw new IllegalArgumentException("Invalid feature value representation: " + s);
        }

        final String feature;
        final double weight;
        if (pos > 0) {
            feature = s.substring(0, pos);
            String s2 = s.substring(pos + 1);
            try {
                weight = Double.parseDouble(s2);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Failed to parse a feature value: " + s, e);
            }
        } else {
            feature = s;
            weight = 1.d;
        }
        return new FeatureValue(feature, weight);
    }

    public static void parseFeatureAsString(@Nonnull final Text t, @Nonnull final FeatureValue probe) {
        String s = t.toString();
        parseFeatureAsString(s, probe);
    }

    public static void parseFeatureAsString(@Nonnull final String s,
            @Nonnull final FeatureValue probe) throws IllegalArgumentException {
        final int pos = s.indexOf(':');
        if (pos == 0) {
            throw new IllegalArgumentException("Invalid feature value representation: " + s);
        }
        if (pos > 0) {
            probe.feature = s.substring(0, pos);
            String s2 = s.substring(pos + 1);
            try {
                probe.value = Double.parseDouble(s2);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Failed to parse a feature value: " + s, e);
            }
        } else {
            probe.feature = s;
            probe.value = 1.d;
        }
    }

}
