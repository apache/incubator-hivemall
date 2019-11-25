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
package hivemall.factorization.fm;

import hivemall.utils.hashing.MurmurHash3;
import hivemall.utils.lang.NumberUtils;

import java.nio.ByteBuffer;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;

public abstract class Feature {
    public static final int DEFAULT_NUM_FIELDS = 256;
    public static final int DEFAULT_FEATURE_BITS = 21;
    public static final int DEFAULT_NUM_FEATURES = 1 << 21; // 2^21

    protected double value;

    public Feature() {}

    public Feature(double value) {
        this.value = value;
    }

    public void setFeature(@Nonnull String f) {
        throw new UnsupportedOperationException();
    }

    @Nonnull
    public String getFeature() {
        throw new UnsupportedOperationException();
    }

    public void setFeatureIndex(@Nonnegative int i) {
        throw new UnsupportedOperationException();
    }

    @Nonnegative
    public int getFeatureIndex() {
        throw new UnsupportedOperationException();
    }

    public short getField() {
        throw new UnsupportedOperationException();
    }

    public void setField(short field) {
        throw new UnsupportedOperationException();
    }

    public double getValue() {
        return value;
    }

    public abstract int bytes();

    public abstract void writeTo(@Nonnull ByteBuffer dst);

    public abstract void readFrom(@Nonnull ByteBuffer src);

    public static int requiredBytes(@Nonnull final Feature[] x) {
        int ret = 0;
        for (Feature f : x) {
            assert (f != null);
            ret += f.bytes();
        }
        return ret;
    }

    @Nullable
    public static Feature[] parseFeatures(@Nonnull final Object arg,
            @Nonnull final ListObjectInspector listOI, @Nullable final Feature[] probes,
            final boolean asIntFeature) throws HiveException {
        if (arg == null) {
            return null;
        }

        final int length = listOI.getListLength(arg);
        final Feature[] ary;
        if (probes != null && probes.length == length) {
            ary = probes;
        } else {
            ary = new Feature[length];
        }

        int j = 0;
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(arg, i);
            if (o == null) {
                continue;
            }
            String s = o.toString();
            Feature f = ary[j];
            if (f == null) {
                f = parseFeature(s, asIntFeature);
            } else {
                parseFeature(s, f, asIntFeature);
            }
            ary[j] = f;
            j++;
        }
        if (j == length) {
            return ary;
        } else {
            Feature[] dst = new Feature[j];
            System.arraycopy(ary, 0, dst, 0, j);
            return dst;
        }
    }

    @Nullable
    public static Feature[] parseFFMFeatures(@Nonnull final Object arg,
            @Nonnull final ListObjectInspector listOI, @Nullable final Feature[] probes,
            final int numFeatures, final int numFields) throws HiveException {
        if (arg == null) {
            return null;
        }

        final int length = listOI.getListLength(arg);
        final Feature[] ary;
        if (probes != null && probes.length == length) {
            ary = probes;
        } else {
            ary = new Feature[length];
        }

        int j = 0;
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(arg, i);
            if (o == null) {
                continue;
            }
            String s = o.toString();
            Feature f = ary[j];
            if (f == null) {
                f = parseFFMFeature(s, numFeatures, numFields);
            } else {
                parseFFMFeature(s, f, numFeatures, numFields);
            }
            ary[j] = f;
            j++;
        }
        if (j == length) {
            return ary;
        } else {
            Feature[] dst = new Feature[j];
            System.arraycopy(ary, 0, dst, 0, j);
            return dst;
        }
    }

    @Nonnull
    static Feature parseFeature(@Nonnull final String fv, final boolean asIntFeature)
            throws HiveException {
        final int pos1 = fv.indexOf(':');
        if (pos1 == -1) {
            if (asIntFeature) {
                int index = parseFeatureIndex(fv);
                return new IntFeature(index, 1.d);
            } else {
                if ("0".equals(fv)) {
                    throw new HiveException("Index value should not be 0: " + fv);
                }
                return new StringFeature(/* index */fv, 1.d);
            }
        } else {
            final String indexStr = fv.substring(0, pos1);
            final String valueStr = fv.substring(pos1 + 1);
            if (asIntFeature) {
                int index = parseFeatureIndex(indexStr);
                double value = parseFeatureValue(valueStr);
                return new IntFeature(index, value);
            } else {
                double value = parseFeatureValue(valueStr);
                if ("0".equals(indexStr)) {
                    throw new HiveException("Index value should not be 0: " + fv);
                }
                return new StringFeature(/* index */indexStr, value);
            }
        }
    }

    @Nonnull
    static IntFeature parseFFMFeature(@Nonnull final String fv) throws HiveException {
        return parseFFMFeature(fv, DEFAULT_NUM_FEATURES, DEFAULT_NUM_FIELDS);
    }

    @Nonnull
    static IntFeature parseFFMFeature(@Nonnull final String fv, final int numFeatures)
            throws HiveException {
        return parseFFMFeature(fv, -1, DEFAULT_NUM_FIELDS);
    }

    @Nonnull
    static IntFeature parseFFMFeature(@Nonnull final String fv, final int numFeatures,
            final int numFields) throws HiveException {
        final int pos1 = fv.indexOf(':');
        if (pos1 == -1) {
            throw new HiveException("Invalid FFM feature format: " + fv);
        }
        final String lead = fv.substring(0, pos1);
        final String rest = fv.substring(pos1 + 1);
        final int pos2 = rest.indexOf(':');
        if (pos2 == -1) {
            throw new HiveException(
                "Invalid FFM feature representation. Expected <field>:<index>:<value> but got "
                        + fv);
        }

        final short field;
        if (NumberUtils.isDigits(lead)) {
            field = parseField(lead, numFields);
        } else {
            field = NumberUtils.castToShort(MurmurHash3.murmurhash3(lead, numFields));
        }

        final int index;
        final String indexStr = rest.substring(0, pos2);
        if (numFeatures == -1 && NumberUtils.isDigits(indexStr)) {
            index = parseFeatureIndex(indexStr);
        } else {
            // +NUM_FIELD to avoid conflict to quantitative features
            index = MurmurHash3.murmurhash3(indexStr, numFeatures) + numFields;
        }
        String valueStr = rest.substring(pos2 + 1);
        double value = parseFeatureValue(valueStr);

        return new IntFeature(index, field, value);
    }

    static void parseFeature(@Nonnull final String fv, @Nonnull final Feature probe,
            final boolean asIntFeature) throws HiveException {
        final int pos1 = fv.indexOf(":");
        if (pos1 == -1) {
            if (asIntFeature) {
                int index = parseFeatureIndex(fv);
                probe.setFeatureIndex(index);
            } else {
                if ("0".equals(fv)) {
                    throw new HiveException("Index value should not be 0: " + fv);
                }
                probe.setFeature(fv);
            }
            probe.value = 1.d;
        } else {
            final String indexStr = fv.substring(0, pos1);
            final String valueStr = fv.substring(pos1 + 1);
            if (asIntFeature) {
                int index = parseFeatureIndex(indexStr);
                probe.setFeatureIndex(index);
                probe.value = parseFeatureValue(valueStr);
            } else {
                if ("0".equals(indexStr)) {
                    throw new HiveException("Index value should not be 0: " + fv);
                }
                probe.setFeature(indexStr);
                probe.value = parseFeatureValue(valueStr);
            }
        }
    }

    static void parseFFMFeature(@Nonnull final String fv, @Nonnull final Feature probe)
            throws HiveException {
        parseFFMFeature(fv, probe, DEFAULT_NUM_FEATURES, DEFAULT_NUM_FIELDS);
    }

    static void parseFFMFeature(@Nonnull final String fv, @Nonnull final Feature probe,
            final int numFeatures, final int numFields) throws HiveException {
        final int pos1 = fv.indexOf(":");
        if (pos1 == -1) {
            throw new HiveException("Invalid FFM feature format: " + fv);
        }
        final String lead = fv.substring(0, pos1);
        final String rest = fv.substring(pos1 + 1);
        final int pos2 = rest.indexOf(':');
        if (pos2 == -1) {
            throw new HiveException(
                "Invalid FFM feature representation. Expected <field>:<index>:<value> but got "
                        + fv);
        }

        final short field;
        if (NumberUtils.isDigits(lead)) {
            field = parseField(lead, numFields);
        } else {
            field = NumberUtils.castToShort(MurmurHash3.murmurhash3(lead, numFields));
        }
        final int index;
        final String indexStr = rest.substring(0, pos2);
        if (numFeatures == -1 && NumberUtils.isDigits(indexStr)) {
            index = parseFeatureIndex(indexStr);
        } else {
            // +NUM_FIELD to avoid conflict to quantitative features
            index = MurmurHash3.murmurhash3(indexStr, numFeatures) + numFields;
        }
        probe.setField(field);
        probe.setFeatureIndex(index);

        String valueStr = rest.substring(pos2 + 1);
        probe.value = parseFeatureValue(valueStr);
    }

    private static int parseFeatureIndex(@Nonnull final String indexStr) throws HiveException {
        final int index;
        try {
            index = Integer.parseInt(indexStr);
        } catch (NumberFormatException e) {
            throw new HiveException("Invalid index value: " + indexStr, e);
        }
        if (index <= 0) {
            throw new HiveException("Feature index MUST be greater than 0: " + indexStr);
        }
        return index;
    }

    private static double parseFeatureValue(@Nonnull final String value) throws HiveException {
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            throw new HiveException("Invalid feature value: " + value, e);
        }
    }

    private static short parseField(@Nonnull final String fieldStr, final int numFields)
            throws HiveException {
        final short field;
        try {
            field = Short.parseShort(fieldStr);
        } catch (NumberFormatException e) {
            throw new HiveException("Invalid field value: " + fieldStr, e);
        }
        if (field < 0 || field >= numFields) {
            throw new HiveException("Invalid field value: " + fieldStr);
        }
        return field;
    }

    public static int toIntFeature(@Nonnull final Feature x) {
        int index = x.getFeatureIndex();
        return -index;
    }

    public static int toIntFeature(@Nonnull final Feature x, @Nonnegative final int yField,
            @Nonnegative final int numFields) {
        int index = x.getFeatureIndex();
        return index * numFields + yField;
    }

    public static void l2normalize(@Nonnull final Feature[] features) {
        double squaredSum = 0.d;
        for (Feature f : features) {
            double v = f.value;
            squaredSum += (v * v);
        }
        if (squaredSum == 0.d) {
            return;
        }

        final double invNorm = 1.d / Math.sqrt(squaredSum);
        for (Feature f : features) {
            f.value *= invNorm;
        }
    }

}
