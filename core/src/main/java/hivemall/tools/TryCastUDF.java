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
package hivemall.tools;

import hivemall.utils.hadoop.HiveUtils;

import java.util.Arrays;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorConverters;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorConverters.Converter;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;

@Description(name = "try_cast",
        value = "_FUNC_(ANY src, const string typeName)"
                + " - Explicitly cast a value as a type. Returns null if cast fails.",
        extended = "SELECT try_cast(array(1.0,2.0,3.0), 'array<string>')\n"
                + "SELECT try_cast(map('A',10,'B',20,'C',30), 'map<string,double>')")
@UDFType(deterministic = true, stateful = false)
public final class TryCastUDF extends GenericUDF {

    private ObjectInspector inputOI;
    private Converter converter;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentException(
                "try_cast(ANY src, const string typeName) expects exactly two arguments");
        }

        this.inputOI = argOIs[0];
        String typeString = HiveUtils.getConstString(argOIs, 1);

        ObjectInspector outputOI = HiveUtils.getObjectInspector(typeString, true);
        this.converter = ObjectInspectorConverters.getConverter(inputOI, outputOI);

        return outputOI;
    }

    @Override
    public Object evaluate(DeferredObject[] args) throws HiveException {
        Object arg0 = args[0].get();
        if (arg0 == null) {
            return null;
        }

        Object input = ObjectInspectorUtils.copyToStandardObject(arg0, inputOI);
        try {
            return converter.convert(input);
        } catch (Exception e) {
            return null;
        }
    }

    @Override
    public String getDisplayString(String[] args) {
        return "try_cast(" + Arrays.toString(args) + ")";
    }

}
