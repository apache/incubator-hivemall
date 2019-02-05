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

import java.util.List;

import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.io.Text;

// @formatter:off
@Description(name = "to_json",
        value = "_FUNC_(ANY object [, const array<string>|const string columnNames]) - Returns Json string",
        extended = "SELECT \n" +
                "  NAMED_STRUCT(\"Name\", \"John\", \"age\", 31),\n" +
                "  to_json(\n" +
                "     NAMED_STRUCT(\"Name\", \"John\", \"age\", 31)\n" +
                "  ),\n" +
                "  to_json(\n" +
                "     NAMED_STRUCT(\"Name\", \"John\", \"age\", 31),\n" +
                "     array('Name', 'age')\n" +
                "  ),\n" +
                "  to_json(\n" +
                "     NAMED_STRUCT(\"Name\", \"John\", \"age\", 31),\n" +
                "     array('name', 'age')\n" +
                "  ),\n" +
                "  to_json(\n" +
                "     NAMED_STRUCT(\"Name\", \"John\", \"age\", 31),\n" +
                "     array('age')\n" +
                "  ),\n" +
                "  to_json(\n" +
                "     NAMED_STRUCT(\"Name\", \"John\", \"age\", 31),\n" +
                "     array()\n" +
                "  ),\n" +
                "  to_json(\n" +
                "     null,\n" +
                "     array()\n" +
                "  ),\n" +
                "  to_json(\n" +
                "    struct(\"123\", \"456\", 789, array(314,007)),\n" +
                "    array('ti','si','i','bi')\n" +
                "  ),\n" +
                "  to_json(\n" +
                "    struct(\"123\", \"456\", 789, array(314,007)),\n" +
                "    'ti,si,i,bi'\n" +
                "  ),\n" +
                "  to_json(\n" +
                "    struct(\"123\", \"456\", 789, array(314,007))\n" +
                "  ),\n" +
                "  to_json(\n" +
                "    NAMED_STRUCT(\"country\", \"japan\", \"city\", \"tokyo\")\n" +
                "  ),\n" +
                "  to_json(\n" +
                "    NAMED_STRUCT(\"country\", \"japan\", \"city\", \"tokyo\"), \n" +
                "    array('city')\n" +
                "  ),\n" +
                "  to_json(\n" +
                "    ARRAY(\n" +
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"tokyo\"), \n" +
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"osaka\")\n" +
                "    )\n" +
                "  ),\n" +
                "  to_json(\n" +
                "    ARRAY(\n" +
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"tokyo\"), \n" +
                "      NAMED_STRUCT(\"country\", \"japan\", \"city\", \"osaka\")\n" +
                "    ),\n" +
                "    array('city')\n" +
                "  );\n" +
                "```\n\n" +
                "```\n" +
                " {\"name\":\"John\",\"age\":31}\n"
                + " {\"name\":\"John\",\"age\":31}\n"
                + " {\"Name\":\"John\",\"age\":31}\n"
                + " {\"name\":\"John\",\"age\":31}\n"
                + " {\"age\":31}\n"
                + " {}\n"
                + " NULL\n"
                + " {\"ti\":\"123\",\"si\":\"456\",\"i\":789,\"bi\":[314,7]}\n"
                + " {\"ti\":\"123\",\"si\":\"456\",\"i\":789,\"bi\":[314,7]}\n"
                + " {\"col1\":\"123\",\"col2\":\"456\",\"col3\":789,\"col4\":[314,7]}\n"
                + " {\"country\":\"japan\",\"city\":\"tokyo\"}\n"
                + " {\"city\":\"tokyo\"}\n"
                + " [{\"country\":\"japan\",\"city\":\"tokyo\"},{\"country\":\"japan\",\"city\":\"osaka\"}]\n"
                + " [{\"country\":\"japan\",\"city\":\"tokyo\"},{\"country\":\"japan\",\"city\":\"osaka\"}]")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class ToJsonUDF extends GenericUDF {

    private ObjectInspector objOI;

    @Nullable
    private List<String> columnNames;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1 && argOIs.length != 2) {
            throw new UDFArgumentException(
                "from_json takes one or two arguments: " + argOIs.length);
        }

        this.objOI = argOIs[0];
        if (argOIs.length == 2) {
            final ObjectInspector argOI1 = argOIs[1];
            if (HiveUtils.isConstString(argOI1)) {
                String names = HiveUtils.getConstString(argOI1);
                this.columnNames = ArrayUtils.asKryoSerializableList(names.split(","));
            } else if (HiveUtils.isConstStringListOI(argOI1)) {
                this.columnNames =
                        ArrayUtils.asKryoSerializableList(HiveUtils.getConstStringArray(argOI1));
            } else {
                throw new UDFArgumentException("Expected `const array<string>` or `const string`"
                        + " but got an unexpected OI type for the third argument: " + argOI1);
            }
        }

        return PrimitiveObjectInspectorFactory.writableStringObjectInspector;
    }

    @Override
    public Text evaluate(DeferredObject[] args) throws HiveException {
        Object obj = args[0].get();
        if (obj == null) {
            return null;
        }

        try {
            return JsonSerdeUtils.serialize(obj, objOI, columnNames);
        } catch (Throwable e) {
            throw new HiveException(
                "Failed to serialize: " + obj + '\n' + ExceptionUtils.prettyPrintStackTrace(e), e);
        }
    }

    @Override
    public String getDisplayString(String[] children) {
        return "to_json(" + StringUtils.join(children, ',') + ")";
    }

}
