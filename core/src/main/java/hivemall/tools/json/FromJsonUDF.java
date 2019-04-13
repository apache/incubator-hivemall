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
package hivemall.tools.json;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.hadoop.JsonSerdeUtils;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.ExceptionUtils;
import hivemall.utils.lang.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;
import org.apache.hadoop.io.Text;
import org.apache.hive.hcatalog.data.HCatRecordObjectInspectorFactory;

// @formatter:off
@Description(name = "from_json",
        value = "_FUNC_(string jsonString, const string returnTypes [, const array<string>|const string columnNames])"
                + " - Return Hive object.",
        extended = "SELECT\n" + 
                "  from_json(to_json(map('one',1,'two',2)), 'map<string,int>'),\n" +
                "  from_json(\n" + 
                "    '{ \"person\" : { \"name\" : \"makoto\" , \"age\" : 37 } }',\n" + 
                "    'struct<name:string,age:int>', \n" + 
                "    array('person')\n" + 
                "  ),\n" + 
                "  from_json(\n" + 
                "    '[0.1,1.1,2.2]',\n" + 
                "    'array<double>'\n" + 
                "  ),\n" + 
                "  from_json(to_json(\n" + 
                "    ARRAY(\n" + 
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"tokyo\"), \n" + 
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"osaka\")\n" + 
                "    )\n" + 
                "  ),'array<struct<country:string,city:string>>'),\n" + 
                "  from_json(to_json(\n" + 
                "    ARRAY(\n" + 
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"tokyo\"), \n" + 
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"osaka\")\n" + 
                "    ),\n" + 
                "    array('city')\n" + 
                "  ), 'array<struct<country:string,city:string>>'),\n" + 
                "  from_json(to_json(\n" + 
                "    ARRAY(\n" + 
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"tokyo\"), \n" + 
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"osaka\")\n" + 
                "    )\n" + 
                "  ),'array<struct<city:string>>');\n"
                + "```\n\n" +
                "```\n" +
                " {\"one\":1,\"two\":2}\n" +
                " {\"name\":\"makoto\",\"age\":37}\n" + 
                " [0.1,1.1,2.2]\n" + 
                " [{\"country\":\"japan\",\"city\":\"tokyo\"},{\"country\":\"japan\",\"city\":\"osaka\"}]\n" + 
                " [{\"country\":\"japan\",\"city\":\"tokyo\"},{\"country\":\"japan\",\"city\":\"osaka\"}]\n" + 
                " [{\"city\":\"tokyo\"},{\"city\":\"osaka\"}]")
//@formatter:on
@UDFType(deterministic = true, stateful = false)
public final class FromJsonUDF extends GenericUDF {

    private StringObjectInspector jsonOI;

    private List<TypeInfo> columnTypes;
    @Nullable
    private List<String> columnNames;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentException(
                "from_json takes two or three arguments: " + argOIs.length);
        }

        this.jsonOI = HiveUtils.asStringOI(argOIs[0]);

        String typeString = HiveUtils.getConstString(argOIs[1]);
        this.columnTypes = TypeInfoUtils.getTypeInfosFromTypeString(typeString);

        if (argOIs.length == 3) {
            final ObjectInspector argOI2 = argOIs[2];
            if (HiveUtils.isConstString(argOI2)) {
                String names = HiveUtils.getConstString(argOI2);
                this.columnNames = ArrayUtils.asKryoSerializableList(names.split(","));
            } else if (HiveUtils.isConstStringListOI(argOI2)) {
                this.columnNames =
                        ArrayUtils.asKryoSerializableList(HiveUtils.getConstStringArray(argOI2));
            } else {
                throw new UDFArgumentException("Expected `const array<string>` or `const string`"
                        + " but got an unexpected OI type for the third argument: " + argOI2);
            }
        }

        return getObjectInspector(columnTypes, columnNames);
    }

    @Nonnull
    private static ObjectInspector getObjectInspector(@Nonnull final List<TypeInfo> columnTypes,
            @Nullable List<String> columnNames) throws UDFArgumentException {
        if (columnTypes.isEmpty()) {
            throw new UDFArgumentException("Returning columnTypes MUST NOT be null");
        }

        final ObjectInspector returnOI;
        final int numColumns = columnTypes.size();
        if (numColumns == 1) {
            TypeInfo type = columnTypes.get(0);
            returnOI =
                    HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(type);
        } else {
            if (columnNames == null) {
                columnNames = new ArrayList<>(numColumns);
                for (int i = 0; i < numColumns; i++) {
                    columnNames.add("c" + i);
                }
            } else {
                if (columnNames.size() != numColumns) {
                    throw new UDFArgumentException("#columnNames != #columnTypes. #columnName="
                            + columnNames.size() + ", #columnTypes=" + numColumns);
                }
            }
            final ObjectInspector[] fieldOIs = new ObjectInspector[numColumns];
            for (int i = 0; i < fieldOIs.length; i++) {
                TypeInfo type = columnTypes.get(i);
                fieldOIs[i] =
                        HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(
                            type);
            }
            returnOI = ObjectInspectorFactory.getStandardStructObjectInspector(columnNames,
                Arrays.asList(fieldOIs));
        }
        return returnOI;
    }

    @Override
    public Object evaluate(DeferredObject[] args) throws HiveException {
        Object arg0 = args[0].get();
        if (arg0 == null) {
            return null;
        }
        Text jsonString = jsonOI.getPrimitiveWritableObject(arg0);

        final Object result;
        try {
            if (columnNames == null && columnTypes != null && columnTypes.size() == 1) {
                result = JsonSerdeUtils.deserialize(jsonString, columnTypes.get(0));
            } else {
                result = JsonSerdeUtils.deserialize(jsonString, columnNames, columnTypes);
            }
        } catch (Throwable e) {
            throw new HiveException("Failed to deserialize Json: \n" + jsonString.toString() + '\n'
                    + ExceptionUtils.prettyPrintStackTrace(e),
                e);
        }
        return result;
    }

    @Override
    public String getDisplayString(String[] args) {
        return "from_json(" + StringUtils.join(args, ',') + ")";
    }

}
