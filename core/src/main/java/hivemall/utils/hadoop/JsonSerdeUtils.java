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
// This file codes borrowed from 
//   - org.apache.hive.hcatalog.data.JsonSerDe
package hivemall.utils.hadoop;

import hivemall.utils.io.FastByteArrayInputStream;
import hivemall.utils.lang.Preconditions;

import java.io.IOException;
import java.nio.charset.CharacterCodingException;
import java.sql.Date;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.annotation.CheckForNull;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.common.type.HiveChar;
import org.apache.hadoop.hive.common.type.HiveDecimal;
import org.apache.hadoop.hive.common.type.HiveVarchar;
import org.apache.hadoop.hive.conf.HiveConf;
import org.apache.hadoop.hive.serde2.SerDeException;
import org.apache.hadoop.hive.serde2.SerDeUtils;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructField;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.UnionObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BinaryObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BooleanObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.ByteObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DateObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.DoubleObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.FloatObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.HiveCharObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.HiveDecimalObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.HiveVarcharObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.IntObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.LongObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.ShortObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.TimestampObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.BaseCharTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory;
import org.apache.hadoop.io.Text;
import org.apache.hive.hcatalog.common.HCatException;
import org.apache.hive.hcatalog.data.schema.HCatFieldSchema;
import org.apache.hive.hcatalog.data.schema.HCatFieldSchema.Type;
import org.apache.hive.hcatalog.data.schema.HCatSchema;
import org.apache.hive.hcatalog.data.schema.HCatSchemaUtils;
import org.codehaus.jackson.JsonFactory;
import org.codehaus.jackson.JsonParseException;
import org.codehaus.jackson.JsonParser;
import org.codehaus.jackson.JsonToken;

public final class JsonSerdeUtils {

    /**
     * Serialize Hive objects as Text.
     */
    @Nonnull
    public static Text serialize(@Nullable final Object obj, @Nonnull final ObjectInspector oi)
            throws SerDeException {
        return serialize(obj, oi, null);
    }

    /**
     * Serialize Hive objects as Text.
     */
    @Nonnull
    public static Text serialize(@Nullable final Object obj, @Nonnull final ObjectInspector oi,
            @Nullable final List<String> columnNames) throws SerDeException {
        final StringBuilder sb = new StringBuilder();
        switch (oi.getCategory()) {
            case STRUCT:
                StructObjectInspector soi = (StructObjectInspector) oi;
                serializeStruct(sb, obj, soi, columnNames);
                break;
            case LIST:
                ListObjectInspector loi = (ListObjectInspector) oi;
                serializeList(sb, obj, loi);
                break;
            case MAP:
                MapObjectInspector moi = (MapObjectInspector) oi;
                serializeMap(sb, obj, moi);
                break;
            case PRIMITIVE:
                PrimitiveObjectInspector poi = (PrimitiveObjectInspector) oi;
                serializePrimitive(sb, obj, poi);
                break;
            default:
                throw new SerDeException("Unknown type in ObjectInspector: " + oi.getCategory());
        }

        return new Text(sb.toString());
    }

    /**
     * Serialize Hive objects as Text.
     */
    private static void serializeStruct(@Nonnull final StringBuilder sb, @Nullable final Object obj,
            @Nonnull final StructObjectInspector soi, @Nullable final List<String> columnNames)
            throws SerDeException {
        if (obj == null) {
            sb.append("null");
        } else {
            final List<? extends StructField> structFields = soi.getAllStructFieldRefs();
            sb.append(SerDeUtils.LBRACE);
            if (columnNames == null) {
                for (int i = 0, len = structFields.size(); i < len; i++) {
                    String colName = structFields.get(i).getFieldName();
                    if (i > 0) {
                        sb.append(SerDeUtils.COMMA);
                    }
                    appendWithQuotes(sb, colName);
                    sb.append(SerDeUtils.COLON);
                    buildJSONString(sb, soi.getStructFieldData(obj, structFields.get(i)),
                        structFields.get(i).getFieldObjectInspector());
                }
            } else if (columnNames.size() == structFields.size()) {
                for (int i = 0, len = structFields.size(); i < len; i++) {
                    if (i > 0) {
                        sb.append(SerDeUtils.COMMA);
                    }
                    String colName = columnNames.get(i);
                    appendWithQuotes(sb, colName);
                    sb.append(SerDeUtils.COLON);
                    buildJSONString(sb, soi.getStructFieldData(obj, structFields.get(i)),
                        structFields.get(i).getFieldObjectInspector());
                }
            } else {
                Collections.sort(columnNames);
                final List<String> found = new ArrayList<>(columnNames.size());
                for (int i = 0, len = structFields.size(); i < len; i++) {
                    String colName = structFields.get(i).getFieldName();
                    if (Collections.binarySearch(columnNames, colName) < 0) {
                        continue;
                    }
                    if (!found.isEmpty()) {
                        sb.append(SerDeUtils.COMMA);
                    }
                    appendWithQuotes(sb, colName);
                    sb.append(SerDeUtils.COLON);
                    buildJSONString(sb, soi.getStructFieldData(obj, structFields.get(i)),
                        structFields.get(i).getFieldObjectInspector());
                    found.add(colName);
                }
                if (found.size() != columnNames.size()) {
                    ArrayList<String> expected = new ArrayList<>(columnNames);
                    expected.removeAll(found);
                    throw new SerDeException("Could not find some fields: " + expected);
                }
            }
            sb.append(SerDeUtils.RBRACE);
        }
    }

    @Nonnull
    private static void serializeList(@Nonnull final StringBuilder sb, @Nullable final Object obj,
            @Nullable final ListObjectInspector loi) throws SerDeException {
        ObjectInspector listElementObjectInspector = loi.getListElementObjectInspector();
        List<?> olist = loi.getList(obj);

        if (olist == null) {
            sb.append("null");
        } else {
            sb.append(SerDeUtils.LBRACKET);
            for (int i = 0; i < olist.size(); i++) {
                if (i > 0) {
                    sb.append(SerDeUtils.COMMA);
                }
                buildJSONString(sb, olist.get(i), listElementObjectInspector);
            }
            sb.append(SerDeUtils.RBRACKET);
        }
    }

    private static void serializeMap(@Nonnull final StringBuilder sb, @Nullable final Object obj,
            @Nonnull final MapObjectInspector moi) throws SerDeException {
        ObjectInspector mapKeyObjectInspector = moi.getMapKeyObjectInspector();
        ObjectInspector mapValueObjectInspector = moi.getMapValueObjectInspector();
        Map<?, ?> omap = moi.getMap(obj);
        if (omap == null) {
            sb.append("null");
        } else {
            sb.append(SerDeUtils.LBRACE);
            boolean first = true;
            for (Object entry : omap.entrySet()) {
                if (first) {
                    first = false;
                } else {
                    sb.append(SerDeUtils.COMMA);
                }
                Map.Entry<?, ?> e = (Map.Entry<?, ?>) entry;
                StringBuilder keyBuilder = new StringBuilder();
                buildJSONString(keyBuilder, e.getKey(), mapKeyObjectInspector);
                String keyString = keyBuilder.toString().trim();
                if ((!keyString.isEmpty()) && (keyString.charAt(0) != SerDeUtils.QUOTE)) {
                    appendWithQuotes(sb, keyString);
                } else {
                    sb.append(keyString);
                }
                sb.append(SerDeUtils.COLON);
                buildJSONString(sb, e.getValue(), mapValueObjectInspector);
            }
            sb.append(SerDeUtils.RBRACE);
        }
    }

    private static void serializePrimitive(@Nonnull final StringBuilder sb,
            @Nullable final Object obj, @Nullable final PrimitiveObjectInspector poi)
            throws SerDeException {
        if (obj == null) {
            sb.append("null");
        } else {
            switch (poi.getPrimitiveCategory()) {
                case BOOLEAN: {
                    boolean b = ((BooleanObjectInspector) poi).get(obj);
                    sb.append(b ? "true" : "false");
                    break;
                }
                case BYTE: {
                    sb.append(((ByteObjectInspector) poi).get(obj));
                    break;
                }
                case SHORT: {
                    sb.append(((ShortObjectInspector) poi).get(obj));
                    break;
                }
                case INT: {
                    sb.append(((IntObjectInspector) poi).get(obj));
                    break;
                }
                case LONG: {
                    sb.append(((LongObjectInspector) poi).get(obj));
                    break;
                }
                case FLOAT: {
                    sb.append(((FloatObjectInspector) poi).get(obj));
                    break;
                }
                case DOUBLE: {
                    sb.append(((DoubleObjectInspector) poi).get(obj));
                    break;
                }
                case STRING: {
                    String s = SerDeUtils.escapeString(
                        ((StringObjectInspector) poi).getPrimitiveJavaObject(obj));
                    appendWithQuotes(sb, s);
                    break;
                }
                case BINARY:
                    byte[] b = ((BinaryObjectInspector) poi).getPrimitiveJavaObject(obj);
                    Text txt = new Text();
                    txt.set(b, 0, b.length);
                    appendWithQuotes(sb, SerDeUtils.escapeString(txt.toString()));
                    break;
                case DATE:
                    Date d = ((DateObjectInspector) poi).getPrimitiveJavaObject(obj);
                    appendWithQuotes(sb, d.toString());
                    break;
                case TIMESTAMP: {
                    Timestamp t = ((TimestampObjectInspector) poi).getPrimitiveJavaObject(obj);
                    appendWithQuotes(sb, t.toString());
                    break;
                }
                case DECIMAL:
                    sb.append(((HiveDecimalObjectInspector) poi).getPrimitiveJavaObject(obj));
                    break;
                case VARCHAR: {
                    String s = SerDeUtils.escapeString(
                        ((HiveVarcharObjectInspector) poi).getPrimitiveJavaObject(obj).toString());
                    appendWithQuotes(sb, s);
                    break;
                }
                case CHAR: {
                    //this should use HiveChar.getPaddedValue() but it's protected; currently (v0.13)
                    // HiveChar.toString() returns getPaddedValue()
                    String s = SerDeUtils.escapeString(
                        ((HiveCharObjectInspector) poi).getPrimitiveJavaObject(obj).toString());
                    appendWithQuotes(sb, s);
                    break;
                }
                default:
                    throw new SerDeException(
                        "Unknown primitive type: " + poi.getPrimitiveCategory());
            }
        }
    }

    private static void buildJSONString(@Nonnull final StringBuilder sb, @Nullable final Object obj,
            @Nonnull final ObjectInspector oi) throws SerDeException {
        switch (oi.getCategory()) {
            case PRIMITIVE: {
                PrimitiveObjectInspector poi = (PrimitiveObjectInspector) oi;
                serializePrimitive(sb, obj, poi);
                break;
            }
            case LIST: {
                ListObjectInspector loi = (ListObjectInspector) oi;
                serializeList(sb, obj, loi);
                break;
            }
            case MAP: {
                MapObjectInspector moi = (MapObjectInspector) oi;
                serializeMap(sb, obj, moi);
                break;
            }
            case STRUCT: {
                StructObjectInspector soi = (StructObjectInspector) oi;
                serializeStruct(sb, obj, soi, null);
                break;
            }
            case UNION: {
                UnionObjectInspector uoi = (UnionObjectInspector) oi;
                if (obj == null) {
                    sb.append("null");
                } else {
                    sb.append(SerDeUtils.LBRACE);
                    sb.append(uoi.getTag(obj));
                    sb.append(SerDeUtils.COLON);
                    buildJSONString(sb, uoi.getField(obj),
                        uoi.getObjectInspectors().get(uoi.getTag(obj)));
                    sb.append(SerDeUtils.RBRACE);
                }
                break;
            }
            default:
                throw new SerDeException("Unknown type in ObjectInspector: " + oi.getCategory());
        }
    }

    @Nonnull
    public static <T> T deserialize(@Nonnull final Text t) throws SerDeException {
        return deserialize(t, null, null);
    }

    /**
     * Deserialize Json array or Json primitives.
     */
    @SuppressWarnings("unchecked")
    @Nonnull
    public static <T> T deserialize(@Nonnull final Text t, @Nonnull TypeInfo columnType)
            throws SerDeException {
        final HiveJsonStructReader reader = new HiveJsonStructReader(columnType);
        reader.setIgnoreUnknownFields(true);
        final Object result;
        try {
            result = reader.parseStruct(new FastByteArrayInputStream(t.getBytes(), t.getLength()));
        } catch (IOException e) {
            throw new SerDeException(e);
        }
        return (T) result;
    }

    @SuppressWarnings("unchecked")
    @Nonnull
    public static <T> T deserialize(@Nonnull final Text t, @Nullable final List<String> columnNames,
            @Nullable final List<TypeInfo> columnTypes) throws SerDeException {
        final Object result;
        try {
            JsonParser p = new JsonFactory().createJsonParser(
                new FastByteArrayInputStream(t.getBytes(), t.getLength()));
            final JsonToken token = p.nextToken();
            if (token == JsonToken.START_OBJECT) {
                result = parseObject(p, columnNames, columnTypes);
            } else if (token == JsonToken.START_ARRAY) {
                result = parseArray(p, columnTypes);
            } else {
                result = parseValue(p);
            }
        } catch (JsonParseException e) {
            throw new SerDeException(e);
        } catch (IOException e) {
            throw new SerDeException(e);
        }
        return (T) result;
    }

    @Nonnull
    private static Object parseObject(@Nonnull final JsonParser p,
            @CheckForNull final List<String> columnNames,
            @CheckForNull final List<TypeInfo> columnTypes)
            throws JsonParseException, IOException, SerDeException {
        Preconditions.checkNotNull(columnNames, "columnNames MUST NOT be null in parseObject",
            SerDeException.class);
        Preconditions.checkNotNull(columnTypes, "columnTypes MUST NOT be null in parseObject",
            SerDeException.class);
        if (columnNames.size() != columnTypes.size()) {
            throw new SerDeException(
                "Size of columnNames and columnTypes does not match. #columnNames="
                        + columnNames.size() + ", #columnTypes=" + columnTypes.size());
        }

        TypeInfo rowTypeInfo = TypeInfoFactory.getStructTypeInfo(columnNames, columnTypes);
        final HCatSchema schema;
        try {
            schema = HCatSchemaUtils.getHCatSchema(rowTypeInfo).get(0).getStructSubSchema();
        } catch (HCatException e) {
            throw new SerDeException(e);
        }

        final List<Object> r = new ArrayList<Object>(Collections.nCopies(columnNames.size(), null));
        JsonToken token;
        while (((token = p.nextToken()) != JsonToken.END_OBJECT) && (token != null)) {
            // iterate through each token, and create appropriate object here.
            populateRecord(r, token, p, schema);
        }

        if (columnTypes.size() == 1) {
            return r.get(0);
        }
        return r;
    }

    @Nonnull
    private static List<Object> parseArray(@Nonnull final JsonParser p,
            @CheckForNull final List<TypeInfo> columnTypes)
            throws HCatException, IOException, SerDeException {
        Preconditions.checkNotNull(columnTypes, "columnTypes MUST NOT be null",
            SerDeException.class);
        if (columnTypes.size() != 1) {
            throw new IOException("Expected a single array but go " + columnTypes);
        }

        TypeInfo elemType = columnTypes.get(0);
        HCatSchema schema = HCatSchemaUtils.getHCatSchema(elemType);

        HCatFieldSchema listSchema = schema.get(0);
        HCatFieldSchema elemSchema = listSchema.getArrayElementSchema().get(0);

        final List<Object> arr = new ArrayList<Object>();
        while (p.nextToken() != JsonToken.END_ARRAY) {
            arr.add(extractCurrentField(p, elemSchema, true));
        }
        return arr;
    }

    @Nonnull
    private static Object parseValue(@Nonnull final JsonParser p)
            throws JsonParseException, IOException {
        final JsonToken t = p.getCurrentToken();
        switch (t) {
            case VALUE_FALSE:
                return Boolean.FALSE;
            case VALUE_TRUE:
                return Boolean.TRUE;
            case VALUE_NULL:
                return null;
            case VALUE_STRING:
                return p.getText();
            case VALUE_NUMBER_FLOAT:
                return p.getDoubleValue();
            case VALUE_NUMBER_INT:
                return p.getIntValue();
            default:
                throw new IOException("Unexpected token: " + t);
        }
    }

    private static void populateRecord(@Nonnull final List<Object> r,
            @Nonnull final JsonToken token, @Nonnull final JsonParser p,
            @Nonnull final HCatSchema s) throws IOException {
        if (token != JsonToken.FIELD_NAME) {
            throw new IOException("Field name expected");
        }
        String fieldName = p.getText();
        Integer fpos = s.getPosition(fieldName);
        if (fpos == null) {
            fpos = getPositionFromHiveInternalColumnName(fieldName);
            if (fpos == -1) {
                skipValue(p);
                return; // unknown field, we return. We'll continue from the next field onwards.
            }
            // If we get past this, then the column name did match the hive pattern for an internal
            // column name, such as _col0, etc, so it *MUST* match the schema for the appropriate column.
            // This means people can't use arbitrary column names such as _col0, and expect us to ignore it
            // if we find it.
            if (!fieldName.equalsIgnoreCase(getHiveInternalColumnName(fpos))) {
                throw new IOException("Hive internal column name (" + fieldName
                        + ") and position encoding (" + fpos + ") for the column name are at odds");
            }
            // If we reached here, then we were successful at finding an alternate internal
            // column mapping, and we're about to proceed.
        }
        HCatFieldSchema hcatFieldSchema = s.getFields().get(fpos);
        Object currField = extractCurrentField(p, hcatFieldSchema, false);
        r.set(fpos, currField);
    }

    @SuppressWarnings("deprecation")
    @Nullable
    private static Object extractCurrentField(@Nonnull final JsonParser p,
            @Nonnull final HCatFieldSchema hcatFieldSchema, final boolean isTokenCurrent)
            throws IOException {
        JsonToken valueToken;
        if (isTokenCurrent) {
            valueToken = p.getCurrentToken();
        } else {
            valueToken = p.nextToken();
        }

        final Object val;
        switch (hcatFieldSchema.getType()) {
            case INT:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : p.getIntValue();
                break;
            case TINYINT:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : p.getByteValue();
                break;
            case SMALLINT:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : p.getShortValue();
                break;
            case BIGINT:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : p.getLongValue();
                break;
            case BOOLEAN:
                String bval = (valueToken == JsonToken.VALUE_NULL) ? null : p.getText();
                if (bval != null) {
                    val = Boolean.valueOf(bval);
                } else {
                    val = null;
                }
                break;
            case FLOAT:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : p.getFloatValue();
                break;
            case DOUBLE:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : p.getDoubleValue();
                break;
            case STRING:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : p.getText();
                break;
            case BINARY:
                String b = (valueToken == JsonToken.VALUE_NULL) ? null : p.getText();
                if (b != null) {
                    try {
                        String t = Text.decode(b.getBytes(), 0, b.getBytes().length);
                        return t.getBytes();
                    } catch (CharacterCodingException e) {
                        throw new IOException("Error generating json binary type from object.", e);
                    }
                } else {
                    val = null;
                }
                break;
            case DATE:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : Date.valueOf(p.getText());
                break;
            case TIMESTAMP:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : Timestamp.valueOf(p.getText());
                break;
            case DECIMAL:
                val = (valueToken == JsonToken.VALUE_NULL) ? null : HiveDecimal.create(p.getText());
                break;
            case VARCHAR:
                int vLen = ((BaseCharTypeInfo) hcatFieldSchema.getTypeInfo()).getLength();
                val = (valueToken == JsonToken.VALUE_NULL) ? null
                        : new HiveVarchar(p.getText(), vLen);
                break;
            case CHAR:
                int cLen = ((BaseCharTypeInfo) hcatFieldSchema.getTypeInfo()).getLength();
                val = (valueToken == JsonToken.VALUE_NULL) ? null : new HiveChar(p.getText(), cLen);
                break;
            case ARRAY:
                if (valueToken == JsonToken.VALUE_NULL) {
                    val = null;
                    break;
                }
                if (valueToken != JsonToken.START_ARRAY) {
                    throw new IOException("Start of Array expected");
                }
                final List<Object> arr = new ArrayList<>();
                final HCatFieldSchema elemSchema = hcatFieldSchema.getArrayElementSchema().get(0);
                while ((valueToken = p.nextToken()) != JsonToken.END_ARRAY) {
                    arr.add(extractCurrentField(p, elemSchema, true));
                }
                val = arr;
                break;
            case MAP:
                if (valueToken == JsonToken.VALUE_NULL) {
                    val = null;
                    break;
                }
                if (valueToken != JsonToken.START_OBJECT) {
                    throw new IOException("Start of Object expected");
                }
                final Map<Object, Object> map = new LinkedHashMap<>();
                final HCatFieldSchema valueSchema = hcatFieldSchema.getMapValueSchema().get(0);
                while ((valueToken = p.nextToken()) != JsonToken.END_OBJECT) {
                    Object k = getObjectOfCorrespondingPrimitiveType(p.getCurrentName(),
                        hcatFieldSchema.getMapKeyTypeInfo());
                    Object v = extractCurrentField(p, valueSchema, false);
                    map.put(k, v);
                }
                val = map;
                break;
            case STRUCT:
                if (valueToken == JsonToken.VALUE_NULL) {
                    val = null;
                    break;
                }
                if (valueToken != JsonToken.START_OBJECT) {
                    throw new IOException("Start of Object expected");
                }
                HCatSchema subSchema = hcatFieldSchema.getStructSubSchema();
                int sz = subSchema.getFieldNames().size();

                List<Object> struct = new ArrayList<>(Collections.nCopies(sz, null));
                while ((valueToken = p.nextToken()) != JsonToken.END_OBJECT) {
                    populateRecord(struct, valueToken, p, subSchema);
                }
                val = struct;
                break;
            default:
                throw new IOException("Unknown type found: " + hcatFieldSchema.getType());
        }
        return val;
    }

    @Nonnull
    private static Object getObjectOfCorrespondingPrimitiveType(String s,
            PrimitiveTypeInfo mapKeyType) throws IOException {
        switch (Type.getPrimitiveHType(mapKeyType)) {
            case INT:
                return Integer.valueOf(s);
            case TINYINT:
                return Byte.valueOf(s);
            case SMALLINT:
                return Short.valueOf(s);
            case BIGINT:
                return Long.valueOf(s);
            case BOOLEAN:
                return (s.equalsIgnoreCase("true"));
            case FLOAT:
                return Float.valueOf(s);
            case DOUBLE:
                return Double.valueOf(s);
            case STRING:
                return s;
            case BINARY:
                try {
                    String t = Text.decode(s.getBytes(), 0, s.getBytes().length);
                    return t.getBytes();
                } catch (CharacterCodingException e) {
                    throw new IOException("Error generating json binary type from object.", e);
                }
            case DATE:
                return Date.valueOf(s);
            case TIMESTAMP:
                return Timestamp.valueOf(s);
            case DECIMAL:
                return HiveDecimal.create(s);
            case VARCHAR:
                return new HiveVarchar(s, ((BaseCharTypeInfo) mapKeyType).getLength());
            case CHAR:
                return new HiveChar(s, ((BaseCharTypeInfo) mapKeyType).getLength());
            default:
                throw new IOException(
                    "Could not convert from string to map type " + mapKeyType.getTypeName());
        }
    }

    private static int getPositionFromHiveInternalColumnName(String internalName) {
        Pattern internalPattern = Pattern.compile("_col([0-9]+)");
        Matcher m = internalPattern.matcher(internalName);
        if (!m.matches()) {
            return -1;
        } else {
            return Integer.parseInt(m.group(1));
        }
    }

    private static void skipValue(@Nonnull final JsonParser p)
            throws JsonParseException, IOException {
        JsonToken valueToken = p.nextToken();
        if ((valueToken == JsonToken.START_ARRAY) || (valueToken == JsonToken.START_OBJECT)) {
            // if the currently read token is a beginning of an array or object, move stream forward
            // skipping any child tokens till we're at the corresponding END_ARRAY or END_OBJECT token
            p.skipChildren();
        }
    }

    @Nonnull
    private static String getHiveInternalColumnName(int fpos) {
        return HiveConf.getColumnInternalName(fpos);
    }

    @Nonnull
    private static StringBuilder appendWithQuotes(@Nonnull final StringBuilder sb,
            @Nonnull final String value) {
        return sb.append(SerDeUtils.QUOTE).append(value).append(SerDeUtils.QUOTE);
    }

}
