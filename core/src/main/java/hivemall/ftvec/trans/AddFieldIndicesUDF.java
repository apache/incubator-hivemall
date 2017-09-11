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
package hivemall.ftvec.trans;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.Preconditions;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@Description(name = "add_field_indicies", value = "_FUNC_(array<string> features) "
        + "- Returns arrays of string that field indicies (<field>:<feature>)* are argumented")
@UDFType(deterministic = true, stateful = false)
public final class AddFieldIndicesUDF extends GenericUDF {

    private ListObjectInspector listOI;

    @Override
    public ObjectInspector initialize(@Nonnull ObjectInspector[] argOIs)
            throws UDFArgumentException {
        if (argOIs.length != 1) {
            throw new UDFArgumentException("Expected a single argument: " + argOIs.length);
        }

        this.listOI = HiveUtils.asListOI(argOIs[0]);
        if (!HiveUtils.isStringOI(listOI.getListElementObjectInspector())) {
            throw new UDFArgumentException("Expected array<string> but got " + argOIs[0]);
        }

        return ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector);
    }

    @Override
    public List<String> evaluate(@Nonnull DeferredObject[] args) throws HiveException {
        Preconditions.checkArgument(args.length == 1);

        final String[] features = HiveUtils.asStringArray(args[0], listOI);
        if (features == null) {
            return null;
        }

        final List<String> argumented = new ArrayList<>(features.length);
        for (int i = 0; i < features.length; i++) {
            final String f = features[i];
            if (f == null) {
                continue;
            }
            argumented.add((i + 1) + ":" + f);
        }

        return argumented;
    }

    @Override
    public String getDisplayString(String[] args) {
        return "add_field_indicies( " + Arrays.toString(args) + " )";
    }


}
