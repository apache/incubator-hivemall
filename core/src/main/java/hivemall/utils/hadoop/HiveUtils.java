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
package hivemall.utils.hadoop;

import static hivemall.HivemallConstants.BIGINT_TYPE_NAME;
import static hivemall.HivemallConstants.BINARY_TYPE_NAME;
import static hivemall.HivemallConstants.BOOLEAN_TYPE_NAME;
import static hivemall.HivemallConstants.DECIMAL_TYPE_NAME;
import static hivemall.HivemallConstants.DOUBLE_TYPE_NAME;
import static hivemall.HivemallConstants.FLOAT_TYPE_NAME;
import static hivemall.HivemallConstants.INT_TYPE_NAME;
import static hivemall.HivemallConstants.SMALLINT_TYPE_NAME;
import static hivemall.HivemallConstants.STRING_TYPE_NAME;
import static hivemall.HivemallConstants.TINYINT_TYPE_NAME;
import static hivemall.HivemallConstants.VOID_TYPE_NAME;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.Properties;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.SerDeException;
import org.apache.hadoop.hive.serde2.io.ByteWritable;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.io.HiveDecimalWritable;
import org.apache.hadoop.hive.serde2.io.ShortWritable;
import org.apache.hadoop.hive.serde2.lazy.ByteArrayRef;
import org.apache.hadoop.hive.serde2.lazy.LazyDouble;
import org.apache.hadoop.hive.serde2.lazy.LazyInteger;
import org.apache.hadoop.hive.serde2.lazy.LazyLong;
import org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe;
import org.apache.hadoop.hive.serde2.lazy.LazyString;
import org.apache.hadoop.hive.serde2.lazy.objectinspector.primitive.LazyPrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.lazy.objectinspector.primitive.LazyStringObjectInspector;
import org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryArray;
import org.apache.hadoop.hive.serde2.lazybinary.LazyBinaryMap;
import org.apache.hadoop.hive.serde2.objectinspector.ConstantObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector.Category;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils.ObjectInspectorCopyOption;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector.PrimitiveCategory;
import org.apache.hadoop.hive.serde2.objectinspector.StandardConstantListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BinaryObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BooleanObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.IntObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.LongObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.WritableConstantStringObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.ListTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;
import org.apache.hadoop.io.BooleanWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;

public final class HiveUtils {

    private HiveUtils() {}

    public static int parseInt(@Nonnull final Object o) {
        if (o instanceof Integer) {
            return ((Integer) o).intValue();
        }
        if (o instanceof IntWritable) {
            return ((IntWritable) o).get();
        }
        if (o instanceof LongWritable) {
            long l = ((LongWritable) o).get();
            if (l > 0x7fffffffL) {
                throw new IllegalArgumentException(
                    "feature index must be less than " + Integer.MAX_VALUE + ", but was " + l);
            }
            return (int) l;
        }
        String s = o.toString();
        return Integer.parseInt(s);
    }

    @Nullable
    public static Writable copyToWritable(@Nullable final Object o,
            @Nonnull final PrimitiveObjectInspector poi) {
        if (o == null) {
            return null;
        }
        Object copied = poi.copyObject(o);
        Object result = poi.getPrimitiveWritableObject(copied);
        return (Writable) result;
    }

    public static Text asText(@Nullable final Object o) {
        if (o == null) {
            return null;
        }
        if (o instanceof Text) {
            return (Text) o;
        }
        if (o instanceof LazyString) {
            LazyString l = (LazyString) o;
            return l.getWritableObject();
        }
        if (o instanceof String) {
            String s = (String) o;
            return new Text(s);
        }
        String s = o.toString();
        return new Text(s);
    }

    public static int asJavaInt(@Nullable final Object o, final int nullValue) {
        if (o == null) {
            return nullValue;
        }
        return asJavaInt(o);
    }

    public static int asJavaInt(@Nullable final Object o) {
        if (o == null) {
            throw new IllegalArgumentException();
        }
        if (o instanceof Integer) {
            return ((Integer) o).intValue();
        }
        if (o instanceof LazyInteger) {
            IntWritable i = ((LazyInteger) o).getWritableObject();
            return i.get();
        }
        if (o instanceof IntWritable) {
            return ((IntWritable) o).get();
        }
        String s = o.toString();
        return Integer.parseInt(s);
    }

    public static double asJavaDouble(@Nullable final Object o) {
        if (o == null) {
            throw new IllegalArgumentException();
        }
        if (o instanceof Double) {
            return ((Double) o).doubleValue();
        }
        if (o instanceof LazyDouble) {
            DoubleWritable d = ((LazyDouble) o).getWritableObject();
            return d.get();
        }
        if (o instanceof DoubleWritable) {
            return ((DoubleWritable) o).get();
        }
        String s = o.toString();
        return Double.parseDouble(s);
    }

    @Nullable
    public static List<String> asStringList(@Nonnull final DeferredObject arg,
            @Nonnull final ListObjectInspector listOI) throws HiveException {
        Object argObj = arg.get();
        if (argObj == null) {
            return null;
        }
        List<?> data = listOI.getList(argObj);
        int size = data.size();
        if (size == 0) {
            return Collections.emptyList();
        }
        final String[] ary = new String[size];
        for (int i = 0; i < size; i++) {
            Object o = data.get(i);
            if (o != null) {
                ary[i] = o.toString();
            }
        }
        return Arrays.asList(ary);
    }

    @Nullable
    public static String[] asStringArray(@Nonnull final DeferredObject arg,
            @Nonnull final ListObjectInspector listOI) throws HiveException {
        Object argObj = arg.get();
        if (argObj == null) {
            return null;
        }
        List<?> data = listOI.getList(argObj);
        final int size = data.size();
        final String[] arr = new String[size];
        for (int i = 0; i < size; i++) {
            Object o = data.get(i);
            if (o != null) {
                arr[i] = o.toString();
            }
        }
        return arr;
    }

    @Nonnull
    public static StructObjectInspector asStructOI(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (oi.getCategory() != Category.STRUCT) {
            throw new UDFArgumentException("Expected Struct OI but got: " + oi.getTypeName());
        }
        return (StructObjectInspector) oi;
    }

    public static boolean isPrimitiveOI(@Nonnull final ObjectInspector oi) {
        return oi.getCategory() == Category.PRIMITIVE;
    }

    public static boolean isStructOI(@Nonnull final ObjectInspector oi) {
        return oi.getCategory() == Category.STRUCT;
    }

    public static boolean isVoidOI(@Nonnull final ObjectInspector oi) {
        String typeName = oi.getTypeName();
        return VOID_TYPE_NAME.equals(typeName);
    }

    public static boolean isStringOI(@Nonnull final ObjectInspector oi) {
        String typeName = oi.getTypeName();
        return STRING_TYPE_NAME.equals(typeName);
    }

    public static boolean isIntOI(@Nonnull final ObjectInspector oi) {
        String typeName = oi.getTypeName();
        return INT_TYPE_NAME.equals(typeName);
    }

    public static boolean isBigIntOI(@Nonnull final ObjectInspector oi) {
        String typeName = oi.getTypeName();
        return BIGINT_TYPE_NAME.equals(typeName);
    }

    public static boolean isBooleanOI(@Nonnull final ObjectInspector oi) {
        String typeName = oi.getTypeName();
        return BOOLEAN_TYPE_NAME.equals(typeName);
    }

    public static boolean isBinaryOI(@Nonnull final ObjectInspector oi) {
        String typeName = oi.getTypeName();
        return BINARY_TYPE_NAME.equals(typeName);
    }

    public static boolean isNumberOI(@Nonnull final ObjectInspector argOI) {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            return false;
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case SHORT:
            case INT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
            case BYTE:
                //case TIMESTAMP:
                return true;
            default:
                return false;
        }
    }

    public static boolean isIntegerOI(@Nonnull final ObjectInspector argOI) {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            return false;
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case INT:
            case SHORT:
            case LONG:
            case BYTE:
                return true;
            default:
                return false;
        }
    }

    public static boolean isListOI(@Nonnull final ObjectInspector oi) {
        Category category = oi.getCategory();
        return category == Category.LIST;
    }

    public static boolean isStringListOI(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        Category category = oi.getCategory();
        if (category != Category.LIST) {
            throw new UDFArgumentException("Expected List OI but was: " + oi);
        }
        ListObjectInspector listOI = (ListObjectInspector) oi;
        return isStringOI(listOI.getListElementObjectInspector());
    }

    public static boolean isMapOI(@Nonnull final ObjectInspector oi) {
        return oi.getCategory() == Category.MAP;
    }

    public static boolean isNumberListOI(@Nonnull final ObjectInspector oi) {
        return isListOI(oi)
                && isNumberOI(((ListObjectInspector) oi).getListElementObjectInspector());
    }

    public static boolean isNumberListListOI(@Nonnull final ObjectInspector oi) {
        return isListOI(oi)
                && isNumberListOI(((ListObjectInspector) oi).getListElementObjectInspector());
    }

    public static boolean isConstListOI(@Nonnull final ObjectInspector oi) {
        return ObjectInspectorUtils.isConstantObjectInspector(oi) && isListOI(oi);
    }

    public static boolean isConstStringListOI(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (!isConstListOI(oi)) {
            return false;
        }
        ListObjectInspector listOI = (ListObjectInspector) oi;
        return isStringOI(listOI.getListElementObjectInspector());
    }

    public static boolean isConstString(@Nonnull final ObjectInspector oi) {
        return ObjectInspectorUtils.isConstantObjectInspector(oi) && isStringOI(oi);
    }

    public static boolean isConstInt(@Nonnull final ObjectInspector oi) {
        return ObjectInspectorUtils.isConstantObjectInspector(oi) && isIntOI(oi);
    }

    public static boolean isConstInteger(@Nonnull final ObjectInspector oi) {
        return ObjectInspectorUtils.isConstantObjectInspector(oi) && isIntegerOI(oi);
    }

    public static boolean isConstBoolean(@Nonnull final ObjectInspector oi) {
        return ObjectInspectorUtils.isConstantObjectInspector(oi) && isBooleanOI(oi);
    }

    public static boolean isPrimitiveTypeInfo(@Nonnull TypeInfo typeInfo) {
        return typeInfo.getCategory() == ObjectInspector.Category.PRIMITIVE;
    }

    public static boolean isStructTypeInfo(@Nonnull TypeInfo typeInfo) {
        return typeInfo.getCategory() == ObjectInspector.Category.STRUCT;
    }

    public static boolean isNumberTypeInfo(@Nonnull TypeInfo typeInfo) {
        if (typeInfo.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            return false;
        }
        switch (((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory()) {
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
                return true;
            default:
                return false;
        }
    }

    public static boolean isBooleanTypeInfo(@Nonnull TypeInfo typeInfo) {
        if (typeInfo.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            return false;
        }
        switch (((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory()) {
            case BOOLEAN:
                return true;
            default:
                return false;
        }
    }

    public static boolean isIntegerTypeInfo(@Nonnull TypeInfo typeInfo) {
        if (typeInfo.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            return false;
        }
        switch (((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory()) {
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
                return true;
            default:
                return false;
        }
    }

    public static boolean isIntTypeInfo(@Nonnull TypeInfo typeInfo) {
        if (typeInfo.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            return false;
        }
        return ((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory() == PrimitiveCategory.INT;
    }

    public static boolean isFloatingPointTypeInfo(@Nonnull TypeInfo typeInfo) {
        if (typeInfo.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            return false;
        }
        switch (((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory()) {
            case DOUBLE:
            case FLOAT:
            case DECIMAL:
                return true;
            default:
                return false;
        }
    }

    public static boolean isStringTypeInfo(@Nonnull TypeInfo typeInfo) {
        if (typeInfo.getCategory() != ObjectInspector.Category.PRIMITIVE) {
            return false;
        }
        return ((PrimitiveTypeInfo) typeInfo).getPrimitiveCategory() == PrimitiveCategory.STRING;
    }

    public static boolean isListTypeInfo(@Nonnull TypeInfo typeInfo) {
        return typeInfo.getCategory() == Category.LIST;
    }

    public static boolean isFloatingPointListTypeInfo(@Nonnull TypeInfo typeInfo) {
        if (typeInfo.getCategory() != Category.LIST) {
            return false;
        }
        TypeInfo elemTypeInfo = ((ListTypeInfo) typeInfo).getListElementTypeInfo();
        return isFloatingPointTypeInfo(elemTypeInfo);
    }

    @Nonnull
    public static ListTypeInfo asListTypeInfo(@Nonnull TypeInfo typeInfo)
            throws UDFArgumentException {
        if (!typeInfo.getCategory().equals(Category.LIST)) {
            throw new UDFArgumentException("Expected list type: " + typeInfo);
        }
        return (ListTypeInfo) typeInfo;
    }

    public static boolean isSameCategoryGroup(@Nonnull final PrimitiveCategory cat1,
            @Nonnull final PrimitiveCategory cat2) {
        if (cat1 == cat2) {
            return true;
        }

        switch (cat1) {
            // integers
            case BYTE:
            case SHORT:
            case INT:
            case LONG: {
                switch (cat2) {
                    case BYTE:
                    case SHORT:
                    case INT:
                    case LONG:
                        return true;
                    default:
                        return false;
                }
            }
            // floating point number
            case FLOAT:
            case DOUBLE: {
                switch (cat2) {
                    case FLOAT:
                    case DOUBLE:
                        return true;
                    default:
                        return false;
                }
            }
            // string
            case STRING:
            case CHAR:
            case VARCHAR:
                switch (cat2) {
                    case STRING:
                    case CHAR:
                    case VARCHAR:
                        return true;
                    default:
                        return false;
                }
            default:
                break;
        }
        return false;
    }

    @Nullable
    public static ArrayList<Object> copyListObject(@Nonnull final DeferredObject argument,
            @Nonnull final ListObjectInspector loi) throws HiveException {
        return copyListObject(argument, loi, ObjectInspectorCopyOption.DEFAULT);
    }

    @Nullable
    public static ArrayList<Object> copyListObject(@Nonnull final DeferredObject argument,
            @Nonnull final ListObjectInspector loi,
            @Nonnull final ObjectInspectorCopyOption objectInspectorOption) throws HiveException {
        final Object o = argument.get();
        if (o == null) {
            return null;
        }

        final int length = loi.getListLength(o);
        final ArrayList<Object> list = new ArrayList<Object>(length);
        for (int i = 0; i < length; i++) {
            Object e = ObjectInspectorUtils.copyToStandardObject(loi.getListElement(o, i),
                loi.getListElementObjectInspector(), objectInspectorOption);
            list.add(e);
        }
        return list;
    }

    public static float getFloat(@Nullable Object o, @Nonnull PrimitiveObjectInspector oi) {
        if (o == null) {
            return 0.f;
        }
        return PrimitiveObjectInspectorUtils.getFloat(o, oi);
    }

    public static double getDouble(@Nullable Object o, @Nonnull PrimitiveObjectInspector oi) {
        if (o == null) {
            return 0.d;
        }
        return PrimitiveObjectInspectorUtils.getDouble(o, oi);
    }

    public static int getInt(@Nullable Object o, @Nonnull PrimitiveObjectInspector oi) {
        if (o == null) {
            return 0;
        }
        return PrimitiveObjectInspectorUtils.getInt(o, oi);
    }

    public static long getLong(@Nullable Object o, @Nonnull PrimitiveObjectInspector oi) {
        if (o == null) {
            return 0L;
        }
        return PrimitiveObjectInspectorUtils.getLong(o, oi);
    }

    @SuppressWarnings("unchecked")
    @Nullable
    public static <T extends Writable> T getConstValue(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (!ObjectInspectorUtils.isConstantObjectInspector(oi)) {
            throw new UDFArgumentException("argument must be a constant value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        ConstantObjectInspector constOI = (ConstantObjectInspector) oi;
        Object v = constOI.getWritableConstantValue();
        return (T) v;
    }

    @Nullable
    public static String[] getConstStringArray(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (!ObjectInspectorUtils.isConstantObjectInspector(oi)) {
            throw new UDFArgumentException("argument must be a constant value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        ConstantObjectInspector constOI = (ConstantObjectInspector) oi;
        if (constOI.getCategory() != Category.LIST) {
            throw new UDFArgumentException(
                "argument must be an array: " + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        final List<?> lst = (List<?>) constOI.getWritableConstantValue();
        if (lst == null) {
            return null;
        }
        final int size = lst.size();
        final String[] ary = new String[size];
        for (int i = 0; i < size; i++) {
            Object o = lst.get(i);
            if (o != null) {
                ary[i] = o.toString();
            }
        }
        return ary;
    }

    @Nullable
    public static double[] getConstDoubleArray(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (!ObjectInspectorUtils.isConstantObjectInspector(oi)) {
            throw new UDFArgumentException("argument must be a constant value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        ConstantObjectInspector constOI = (ConstantObjectInspector) oi;
        if (constOI.getCategory() != Category.LIST) {
            throw new UDFArgumentException(
                "argument must be an array: " + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        StandardConstantListObjectInspector listOI = (StandardConstantListObjectInspector) constOI;
        PrimitiveObjectInspector elemOI =
                HiveUtils.asDoubleCompatibleOI(listOI.getListElementObjectInspector());

        final List<?> lst = listOI.getWritableConstantValue();
        if (lst == null) {
            return null;
        }
        final int size = lst.size();
        final double[] ary = new double[size];
        for (int i = 0; i < size; i++) {
            Object o = lst.get(i);
            if (o == null) {
                ary[i] = Double.NaN;
            } else {
                ary[i] = PrimitiveObjectInspectorUtils.getDouble(o, elemOI);
            }
        }
        return ary;
    }

    @Nullable
    public static String getConstString(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (!isStringOI(oi)) {
            throw new UDFArgumentException("argument must be a Text value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        Text v = getConstValue(oi);
        return v == null ? null : v.toString();
    }


    @Nullable
    public static String getConstString(@Nonnull final ObjectInspector[] argOIs, final int argIndex)
            throws UDFArgumentException {
        final ObjectInspector oi = getObjectInspector(argOIs, argIndex);
        if (!isStringOI(oi)) {
            throw new UDFArgumentTypeException(argIndex, "argument must be a Text value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        Text v = getConstValue(oi);
        return v == null ? null : v.toString();
    }

    public static boolean getConstBoolean(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (!isBooleanOI(oi)) {
            throw new UDFArgumentException("argument must be a Boolean value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        BooleanWritable v = getConstValue(oi);
        return v.get();
    }


    public static boolean getConstBoolean(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        final ObjectInspector oi = getObjectInspector(argOIs, argIndex);
        if (!isBooleanOI(oi)) {
            throw new UDFArgumentTypeException(argIndex, "argument must be a Boolean value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        BooleanWritable v = getConstValue(oi);
        return v.get();
    }

    public static int getConstInt(@Nonnull final ObjectInspector oi) throws UDFArgumentException {
        if (!isIntOI(oi)) {
            throw new UDFArgumentException("argument must be a Int value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        IntWritable v = getConstValue(oi);
        return v.get();
    }

    public static long getConstLong(@Nonnull final ObjectInspector oi) throws UDFArgumentException {
        if (!isBigIntOI(oi)) {
            throw new UDFArgumentException("argument must be a BigInt value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        LongWritable v = getConstValue(oi);
        return v.get();
    }

    public static int getAsConstInt(@Nonnull final ObjectInspector numberOI)
            throws UDFArgumentException {
        final String typeName = numberOI.getTypeName();
        if (INT_TYPE_NAME.equals(typeName)) {
            IntWritable v = getConstValue(numberOI);
            return v.get();
        } else if (BIGINT_TYPE_NAME.equals(typeName)) {
            LongWritable v = getConstValue(numberOI);
            return (int) v.get();
        } else if (SMALLINT_TYPE_NAME.equals(typeName)) {
            ShortWritable v = getConstValue(numberOI);
            return v.get();
        } else if (TINYINT_TYPE_NAME.equals(typeName)) {
            ByteWritable v = getConstValue(numberOI);
            return v.get();
        }
        throw new UDFArgumentException("Unexpected argument type to cast as INT: "
                + TypeInfoUtils.getTypeInfoFromObjectInspector(numberOI));
    }

    public static long getAsConstLong(@Nonnull final ObjectInspector numberOI)
            throws UDFArgumentException {
        final String typeName = numberOI.getTypeName();
        if (BIGINT_TYPE_NAME.equals(typeName)) {
            LongWritable v = getConstValue(numberOI);
            return v.get();
        } else if (INT_TYPE_NAME.equals(typeName)) {
            IntWritable v = getConstValue(numberOI);
            return v.get();
        } else if (SMALLINT_TYPE_NAME.equals(typeName)) {
            ShortWritable v = getConstValue(numberOI);
            return v.get();
        } else if (TINYINT_TYPE_NAME.equals(typeName)) {
            ByteWritable v = getConstValue(numberOI);
            return v.get();
        }
        throw new UDFArgumentException("Unexpected argument type to cast as long: "
                + TypeInfoUtils.getTypeInfoFromObjectInspector(numberOI));
    }

    public static float getAsConstFloat(@Nonnull final ObjectInspector numberOI)
            throws UDFArgumentException {
        final String typeName = numberOI.getTypeName();
        if (FLOAT_TYPE_NAME.equals(typeName)) {
            FloatWritable v = getConstValue(numberOI);
            return v.get();
        } else if (DOUBLE_TYPE_NAME.equals(typeName)) {
            DoubleWritable v = getConstValue(numberOI);
            return (float) v.get();
        } else if (INT_TYPE_NAME.equals(typeName)) {
            IntWritable v = getConstValue(numberOI);
            return v.get();
        } else if (BIGINT_TYPE_NAME.equals(typeName)) {
            LongWritable v = getConstValue(numberOI);
            return v.get();
        } else if (SMALLINT_TYPE_NAME.equals(typeName)) {
            ShortWritable v = getConstValue(numberOI);
            return v.get();
        } else if (TINYINT_TYPE_NAME.equals(typeName)) {
            ByteWritable v = getConstValue(numberOI);
            return v.get();
        } else if (DECIMAL_TYPE_NAME.equals(typeName)) {
            HiveDecimalWritable v = getConstValue(numberOI);
            return v.getHiveDecimal().floatValue();
        }
        throw new UDFArgumentException("Unexpected argument type to cast as double: "
                + TypeInfoUtils.getTypeInfoFromObjectInspector(numberOI));
    }

    public static double getAsConstDouble(@Nonnull final ObjectInspector numberOI)
            throws UDFArgumentException {
        final String typeName = numberOI.getTypeName();
        if (DOUBLE_TYPE_NAME.equals(typeName)) {
            DoubleWritable v = getConstValue(numberOI);
            return v.get();
        } else if (FLOAT_TYPE_NAME.equals(typeName)) {
            FloatWritable v = getConstValue(numberOI);
            return v.get();
        } else if (INT_TYPE_NAME.equals(typeName)) {
            IntWritable v = getConstValue(numberOI);
            return v.get();
        } else if (BIGINT_TYPE_NAME.equals(typeName)) {
            LongWritable v = getConstValue(numberOI);
            return v.get();
        } else if (SMALLINT_TYPE_NAME.equals(typeName)) {
            ShortWritable v = getConstValue(numberOI);
            return v.get();
        } else if (TINYINT_TYPE_NAME.equals(typeName)) {
            ByteWritable v = getConstValue(numberOI);
            return v.get();
        } else if (DECIMAL_TYPE_NAME.equals(typeName)) {
            HiveDecimalWritable v = getConstValue(numberOI);
            return v.getHiveDecimal().doubleValue();
        }
        throw new UDFArgumentException("Unexpected argument type to cast as double: "
                + TypeInfoUtils.getTypeInfoFromObjectInspector(numberOI));
    }

    @Nonnull
    public static long[] asLongArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI, @Nonnull PrimitiveObjectInspector elemOI) {
        if (argObj == null) {
            return null;
        }
        final int length = listOI.getListLength(argObj);
        final long[] ary = new long[length];
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(argObj, i);
            if (o == null) {
                continue;
            }
            ary[i] = PrimitiveObjectInspectorUtils.getLong(o, elemOI);
        }
        return ary;
    }

    @Nonnull
    public static long[] asLongArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI, @Nonnull LongObjectInspector elemOI) {
        if (argObj == null) {
            return null;
        }
        final int length = listOI.getListLength(argObj);
        final long[] ary = new long[length];
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(argObj, i);
            if (o == null) {
                continue;
            }
            ary[i] = elemOI.get(o);
        }
        return ary;
    }

    @Nullable
    public static float[] asFloatArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI,
            @Nonnull final PrimitiveObjectInspector elemOI) throws UDFArgumentException {
        return asFloatArray(argObj, listOI, elemOI, true);
    }

    @Nullable
    public static float[] asFloatArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI,
            @Nonnull final PrimitiveObjectInspector elemOI, final boolean avoidNull)
            throws UDFArgumentException {
        if (argObj == null) {
            return null;
        }
        final int length = listOI.getListLength(argObj);
        final float[] ary = new float[length];
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(argObj, i);
            if (o == null) {
                if (avoidNull) {
                    continue;
                }
                throw new UDFArgumentException("Found null at index " + i);
            }
            ary[i] = PrimitiveObjectInspectorUtils.getFloat(o, elemOI);
        }
        return ary;
    }

    @Nullable
    public static double[] asDoubleArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI,
            @Nonnull final PrimitiveObjectInspector elemOI) throws UDFArgumentException {
        return asDoubleArray(argObj, listOI, elemOI, true);
    }

    @Nullable
    public static double[] asDoubleArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI,
            @Nonnull final PrimitiveObjectInspector elemOI, final boolean avoidNull)
            throws UDFArgumentException {
        if (argObj == null) {
            return null;
        }
        final int length = listOI.getListLength(argObj);
        final double[] ary = new double[length];
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(argObj, i);
            if (o == null) {
                if (avoidNull) {
                    continue;
                }
                throw new UDFArgumentException("Found null at index " + i);
            }
            ary[i] = PrimitiveObjectInspectorUtils.getDouble(o, elemOI);
        }
        return ary;
    }

    @Nonnull
    public static void toDoubleArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI,
            @Nonnull final PrimitiveObjectInspector elemOI, @Nonnull final double[] out,
            final boolean avoidNull) throws UDFArgumentException {
        if (argObj == null) {
            return;
        }
        final int length = listOI.getListLength(argObj);
        if (out.length != length) {
            throw new UDFArgumentException(
                "Dimension mismatched. Expected: " + out.length + ", Actual: " + length);
        }
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(argObj, i);
            if (o == null) {
                if (avoidNull) {
                    continue;
                }
                throw new UDFArgumentException("Found null at index " + i);
            }
            out[i] = PrimitiveObjectInspectorUtils.getDouble(o, elemOI);
        }
        return;
    }

    @Nonnull
    public static void toDoubleArray(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI,
            @Nonnull final PrimitiveObjectInspector elemOI, @Nonnull final double[] out,
            final double nullValue) throws UDFArgumentException {
        if (argObj == null) {
            return;
        }
        final int length = listOI.getListLength(argObj);
        if (out.length != length) {
            throw new UDFArgumentException(
                "Dimension mismatched. Expected: " + out.length + ", Actual: " + length);
        }
        for (int i = 0; i < length; i++) {
            Object o = listOI.getListElement(argObj, i);
            if (o == null) {
                out[i] = nullValue;
                continue;
            }
            out[i] = PrimitiveObjectInspectorUtils.getDouble(o, elemOI);
        }
        return;
    }

    /**
     * @return the number of true bits
     */
    @Nonnull
    public static int setBits(@Nullable final Object argObj,
            @Nonnull final ListObjectInspector listOI,
            @Nonnull final PrimitiveObjectInspector elemOI, @Nonnull final BitSet bitset)
            throws UDFArgumentException {
        if (argObj == null) {
            return 0;
        }
        int count = 0;
        final int length = listOI.getListLength(argObj);
        for (int i = 0; i < length; i++) {
            final Object o = listOI.getListElement(argObj, i);
            if (o == null) {
                continue;
            }
            final int index = PrimitiveObjectInspectorUtils.getInt(o, elemOI);
            if (index < 0) {
                throw new UDFArgumentException("Negative index is not allowed: " + index);
            }
            bitset.set(index);
            count++;
        }
        return count;
    }

    @Nonnull
    public static ConstantObjectInspector asConstantObjectInspector(
            @Nonnull final ObjectInspector oi) throws UDFArgumentException {
        if (!ObjectInspectorUtils.isConstantObjectInspector(oi)) {
            throw new UDFArgumentException("argument must be a constant value: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        return (ConstantObjectInspector) oi;
    }

    public static ObjectInspector getObjectInspector(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        if (argIndex >= argOIs.length) {
            throw new UDFArgumentException("Illegal argument index:" + argIndex);
        }
        return argOIs[argIndex];
    }

    @Nonnull
    public static PrimitiveObjectInspector asPrimitiveObjectInspector(
            @Nonnull final ObjectInspector oi) throws UDFArgumentException {
        if (oi.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentException("Expecting PrimitiveObjectInspector: "
                    + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        return (PrimitiveObjectInspector) oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asPrimitiveObjectInspector(
            @Nonnull final ObjectInspector[] argOIs, final int argIndex)
            throws UDFArgumentException {
        final ObjectInspector oi = getObjectInspector(argOIs, argIndex);
        if (oi.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentException("Expecting PrimitiveObjectInspector for argOIs["
                    + argIndex + "] but got " + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        return (PrimitiveObjectInspector) oi;
    }

    @Nonnull
    public static StringObjectInspector asStringOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentException {
        if (!isStringOI(argOI)) {
            throw new UDFArgumentException("Argument type must be String: " + argOI.getTypeName());
        }
        return (StringObjectInspector) argOI;
    }

    @Nonnull
    public static StringObjectInspector asStringOI(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        final ObjectInspector oi = getObjectInspector(argOIs, argIndex);
        if (!isStringOI(oi)) {
            throw new UDFArgumentException(
                "argOIs[" + argIndex + "] type must be String: " + oi.getTypeName());
        }
        return (StringObjectInspector) oi;
    }

    @Nonnull
    public static BinaryObjectInspector asBinaryOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentException {
        if (!BINARY_TYPE_NAME.equals(argOI.getTypeName())) {
            throw new UDFArgumentException("Argument type must be Binary: " + argOI.getTypeName());
        }
        return (BinaryObjectInspector) argOI;
    }

    @Nonnull
    public static BooleanObjectInspector asBooleanOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentException {
        if (!BOOLEAN_TYPE_NAME.equals(argOI.getTypeName())) {
            throw new UDFArgumentException("Argument type must be Boolean: " + argOI.getTypeName());
        }
        return (BooleanObjectInspector) argOI;
    }

    @Nonnull
    public static BooleanObjectInspector asBooleanOI(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        ObjectInspector argOI = getObjectInspector(argOIs, argIndex);
        if (!BOOLEAN_TYPE_NAME.equals(argOI.getTypeName())) {
            throw new UDFArgumentTypeException(argIndex,
                "Argument type must be Boolean: " + argOI.getTypeName());
        }
        return (BooleanObjectInspector) argOI;
    }

    @Nonnull
    public static IntObjectInspector asIntOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentException {
        if (!INT_TYPE_NAME.equals(argOI.getTypeName())) {
            throw new UDFArgumentException("Argument type must be INT: " + argOI.getTypeName());
        }
        return (IntObjectInspector) argOI;
    }

    @Nonnull
    public static LongObjectInspector asLongOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentException {
        if (!BIGINT_TYPE_NAME.equals(argOI.getTypeName())) {
            throw new UDFArgumentException("Argument type must be BIGINT: " + argOI.getTypeName());
        }
        return (LongObjectInspector) argOI;
    }

    @Nonnull
    public static DoubleObjectInspector asDoubleOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentException {
        if (!DOUBLE_TYPE_NAME.equals(argOI.getTypeName())) {
            throw new UDFArgumentException("Argument type must be DOUBLE: " + argOI.getTypeName());
        }
        return (DoubleObjectInspector) argOI;
    }

    @Nonnull
    public static DoubleObjectInspector asDoubleOI(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        ObjectInspector argOI = getObjectInspector(argOIs, argIndex);
        if (!DOUBLE_TYPE_NAME.equals(argOI.getTypeName())) {
            throw new UDFArgumentTypeException(argIndex,
                "Argument type must be DOUBLE: " + argOI.getTypeName());
        }
        return (DoubleObjectInspector) argOI;
    }

    @Nonnull
    public static PrimitiveObjectInspector asIntCompatibleOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentTypeException {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0, "Only primitive type arguments are accepted but "
                    + argOI.getTypeName() + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case INT:
            case SHORT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
            case BOOLEAN:
            case BYTE:
            case STRING:
                break;
            default:
                throw new UDFArgumentTypeException(0,
                    "Unexpected type '" + argOI.getTypeName() + "' is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asIntCompatibleOI(
            @Nonnull final ObjectInspector[] argOIs, final int argIndex)
            throws UDFArgumentException {
        ObjectInspector argOI = getObjectInspector(argOIs, argIndex);
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(argIndex,
                "Only primitive type arguments are accepted but " + argOI.getTypeName()
                        + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case INT:
            case SHORT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
            case BOOLEAN:
            case BYTE:
            case STRING:
                break;
            default:
                throw new UDFArgumentTypeException(argIndex,
                    "Unexpected type '" + argOI.getTypeName() + "' is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asLongCompatibleOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentTypeException {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0, "Only primitive type arguments are accepted but "
                    + argOI.getTypeName() + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case LONG:
            case INT:
            case SHORT:
            case BYTE:
            case BOOLEAN:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
            case STRING:
            case TIMESTAMP:
                break;
            default:
                throw new UDFArgumentTypeException(0,
                    "Unexpected type '" + argOI.getTypeName() + "' is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asIntegerOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentTypeException {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0, "Only primitive type arguments are accepted but "
                    + argOI.getTypeName() + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case INT:
            case SHORT:
            case LONG:
            case BYTE:
                break;
            default:
                throw new UDFArgumentTypeException(0,
                    "Unexpected type '" + argOI.getTypeName() + "' is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asIntegerOI(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        final ObjectInspector argOI = getObjectInspector(argOIs, argIndex);
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(argIndex,
                "Only primitive type arguments are accepted but " + argOI.getTypeName()
                        + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case INT:
            case SHORT:
            case LONG:
            case BYTE:
                break;
            default:
                throw new UDFArgumentTypeException(argIndex,
                    "Unexpected type '" + argOI.getTypeName() + "' is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asDoubleCompatibleOI(
            @Nonnull final ObjectInspector argOI) throws UDFArgumentTypeException {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0, "Only primitive type arguments are accepted but "
                    + argOI.getTypeName() + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
            case STRING:
            case TIMESTAMP:
                break;
            default:
                throw new UDFArgumentTypeException(0,
                    "Only numeric or string type arguments are accepted but " + argOI.getTypeName()
                            + " is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asDoubleCompatibleOI(
            @Nonnull final ObjectInspector[] argOIs, final int argIndex)
            throws UDFArgumentException {
        final PrimitiveObjectInspector oi = asPrimitiveObjectInspector(argOIs, argIndex);
        switch (oi.getPrimitiveCategory()) {
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
            case STRING:
            case TIMESTAMP:
                break;
            default:
                throw new UDFArgumentTypeException(argIndex,
                    "Only numeric or string type arguments are accepted but " + oi.getTypeName()
                            + " is passed for argument index " + argIndex);
        }
        return oi;

    }

    @Nonnull
    public static PrimitiveObjectInspector asFloatingPointOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentTypeException {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0, "Only primitive type arguments are accepted but "
                    + argOI.getTypeName() + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
                break;
            default:
                throw new UDFArgumentTypeException(0, "Only floating point number is accepted but "
                        + argOI.getTypeName() + " is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asNumberOI(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        final PrimitiveObjectInspector oi = asPrimitiveObjectInspector(argOIs, argIndex);
        switch (oi.getPrimitiveCategory()) {
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
                break;
            default:
                throw new UDFArgumentTypeException(argIndex,
                    "Only numeric argument is accepted but " + oi.getTypeName() + " is passed.");
        }
        return oi;
    }

    @Nonnull
    public static PrimitiveObjectInspector asNumberOI(@Nonnull final ObjectInspector argOI)
            throws UDFArgumentTypeException {
        if (argOI.getCategory() != Category.PRIMITIVE) {
            throw new UDFArgumentTypeException(0, "Only primitive type arguments are accepted but "
                    + argOI.getTypeName() + " is passed.");
        }
        final PrimitiveObjectInspector oi = (PrimitiveObjectInspector) argOI;
        switch (oi.getPrimitiveCategory()) {
            case BYTE:
            case SHORT:
            case INT:
            case LONG:
            case FLOAT:
            case DOUBLE:
            case DECIMAL:
                break;
            default:
                throw new UDFArgumentTypeException(0,
                    "Only numeric argument is accepted but " + argOI.getTypeName() + " is passed.");
        }
        return oi;
    }

    @Nonnull
    public static ListObjectInspector asListOI(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        Category category = oi.getCategory();
        if (category != Category.LIST) {
            throw new UDFArgumentException("Expected List OI but was: " + oi);
        }
        return (ListObjectInspector) oi;
    }

    @Nonnull
    public static ListObjectInspector asListOI(@Nonnull final ObjectInspector[] argOIs,
            final int argIndex) throws UDFArgumentException {
        final ObjectInspector oi = getObjectInspector(argOIs, argIndex);
        Category category = oi.getCategory();
        if (category != Category.LIST) {
            throw new UDFArgumentException("Expecting ListObjectInspector for argOIs[" + argIndex
                    + "] but got " + TypeInfoUtils.getTypeInfoFromObjectInspector(oi));
        }
        return (ListObjectInspector) oi;
    }


    @Nonnull
    public static MapObjectInspector asMapOI(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        if (oi.getCategory() != Category.MAP) {
            throw new UDFArgumentException("Expected Map OI but was: " + oi);
        }
        return (MapObjectInspector) oi;
    }

    public static void validateFeatureOI(@Nonnull final ObjectInspector oi)
            throws UDFArgumentException {
        final String typeName = oi.getTypeName();
        if (!STRING_TYPE_NAME.equals(typeName) && !INT_TYPE_NAME.equals(typeName)
                && !BIGINT_TYPE_NAME.equals(typeName)) {
            throw new UDFArgumentException(
                "argument type for a feature must be List of key type [Int|BitInt|Text]: "
                        + typeName);
        }
    }

    @Nonnull
    public static FloatWritable[] newFloatArray(final int size, final float defaultVal) {
        final FloatWritable[] array = new FloatWritable[size];
        for (int i = 0; i < size; i++) {
            array[i] = new FloatWritable(defaultVal);
        }
        return array;
    }

    public static LazySimpleSerDe getKeyValueLineSerde(
            @Nonnull final PrimitiveObjectInspector keyOI,
            @Nonnull final PrimitiveObjectInspector valueOI) throws SerDeException {
        LazySimpleSerDe serde = new LazySimpleSerDe();
        Configuration conf = new Configuration();
        Properties tbl = new Properties();
        tbl.setProperty("columns", "key,value");
        tbl.setProperty("columns.types", keyOI.getTypeName() + "," + valueOI.getTypeName());
        serde.initialize(conf, tbl);
        return serde;
    }

    public static LazySimpleSerDe getLineSerde(@Nonnull final PrimitiveObjectInspector... OIs)
            throws SerDeException {
        if (OIs.length == 0) {
            throw new IllegalArgumentException("OIs must be specified");
        }
        LazySimpleSerDe serde = new LazySimpleSerDe();
        Configuration conf = new Configuration();
        Properties tbl = new Properties();

        StringBuilder columnNames = new StringBuilder();
        StringBuilder columnTypes = new StringBuilder();
        for (int i = 0; i < OIs.length; i++) {
            columnNames.append('c').append(i + 1).append(',');
            columnTypes.append(OIs[i].getTypeName()).append(',');
        }
        columnNames.deleteCharAt(columnNames.length() - 1);
        columnTypes.deleteCharAt(columnTypes.length() - 1);

        tbl.setProperty("columns", columnNames.toString());
        tbl.setProperty("columns.types", columnTypes.toString());
        serde.initialize(conf, tbl);
        return serde;
    }

    @Nonnull
    public static Object castLazyBinaryObject(@Nonnull final Object obj) {
        if (obj instanceof LazyBinaryMap) {
            return ((LazyBinaryMap) obj).getMap();
        } else if (obj instanceof LazyBinaryArray) {
            return ((LazyBinaryArray) obj).getList();
        }
        return obj;
    }

    @Nonnull
    public static LazyString lazyString(@Nonnull final String str) {
        return lazyString(str, (byte) '\\');
    }

    @Nonnull
    public static LazyString lazyString(@Nonnull final String str, final byte escapeChar) {
        LazyStringObjectInspector oi =
                LazyPrimitiveObjectInspectorFactory.getLazyStringObjectInspector(false, escapeChar);
        return lazyString(str, oi);
    }

    @Nonnull
    public static LazyString lazyString(@Nonnull final String str,
            @Nonnull final LazyStringObjectInspector oi) {
        LazyString lazy = new LazyString(oi);
        ByteArrayRef ref = new ByteArrayRef();
        byte[] data = str.getBytes(StandardCharsets.UTF_8);
        ref.setData(data);
        lazy.init(ref, 0, data.length);
        return lazy;
    }

    @Nonnull
    public static LazyInteger lazyInteger(@Nonnull final int v) {
        LazyInteger lazy =
                new LazyInteger(LazyPrimitiveObjectInspectorFactory.LAZY_INT_OBJECT_INSPECTOR);
        lazy.getWritableObject().set(v);
        return lazy;
    }

    @Nonnull
    public static LazyLong lazyLong(@Nonnull final long v) {
        LazyLong lazy =
                new LazyLong(LazyPrimitiveObjectInspectorFactory.LAZY_LONG_OBJECT_INSPECTOR);
        lazy.getWritableObject().set(v);
        return lazy;
    }

    @Nonnull
    public static ObjectInspector getObjectInspector(@Nonnull final String typeString,
            final boolean preferWritable) {
        TypeInfo typeInfo = TypeInfoUtils.getTypeInfoFromTypeString(typeString);
        if (preferWritable) {
            return TypeInfoUtils.getStandardWritableObjectInspectorFromTypeInfo(typeInfo);
        } else {
            return TypeInfoUtils.getStandardJavaObjectInspectorFromTypeInfo(typeInfo);
        }
    }

    @Nonnull
    public static WritableConstantStringObjectInspector getConstStringObjectInspector(
            @Nonnull final String str) {
        return (WritableConstantStringObjectInspector) PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            TypeInfoFactory.stringTypeInfo, new Text(str));
    }

    @Nullable
    public static StructField getStructFieldRef(@Nonnull String fieldName,
            @Nonnull final List<? extends StructField> fields) {
        fieldName = fieldName.toLowerCase();
        for (StructField f : fields) {
            if (f.getFieldName().equals(fieldName)) {
                return f;
            }
        }
        // For backward compatibility: fieldNames can also be integer Strings.
        try {
            final int i = Integer.parseInt(fieldName);
            if (i >= 0 && i < fields.size()) {
                return fields.get(i);
            }
        } catch (NumberFormatException e) {
            // ignore
        }
        return null;
    }
}
