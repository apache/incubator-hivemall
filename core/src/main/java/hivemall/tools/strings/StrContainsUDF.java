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
package hivemall.tools.strings;

import hivemall.utils.hadoop.HiveUtils;

import java.util.List;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentTypeException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BooleanObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.StringObjectInspector;

//@formatter:off
@Description(name = "str_contains",
        value = "_FUNC_(string query, array<string> searchTerms [, boolean orQuery=false])"
                + " - Returns true if the given query contains search terms",
        extended = "select\n" + 
                "  str_contains('There are apple and orange', array('apple')), -- or=false\n" + 
                "  str_contains('There are apple and orange', array('apple', 'banana'), true), -- or=true\n" + 
                "  str_contains('There are apple and orange', array('apple', 'banana'), false); -- or=false\n" + 
                "> true, true, false")
//@formatter:on
@UDFType(deterministic = true, stateful = false)
public final class StrContainsUDF extends GenericUDF {

    private StringObjectInspector queryOI;
    private ListObjectInspector searchTermsOI;
    private BooleanObjectInspector orQueryOI;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2 && argOIs.length != 3) {
            throw new UDFArgumentLengthException("str_contains expects two or three arguments");
        }

        this.queryOI = HiveUtils.asStringOI(argOIs, 0);
        if (!HiveUtils.isStringListOI(argOIs[1])) {
            throw new UDFArgumentTypeException(1,
                "Expected array<string> for the second argument but got "
                        + argOIs[1].getTypeName());
        }
        this.searchTermsOI = HiveUtils.asListOI(argOIs, 1);

        if (argOIs.length == 3) {
            this.orQueryOI = HiveUtils.asBooleanOI(argOIs, 2);
        }

        return PrimitiveObjectInspectorFactory.javaBooleanObjectInspector;
    }

    @Override
    public Boolean evaluate(DeferredObject[] args) throws HiveException {
        final String query = queryOI.getPrimitiveJavaObject(args[0].get());
        if (query == null) {
            return null;
        }

        final List<String> searchTerms = HiveUtils.asStringList(args[1], searchTermsOI);
        if (searchTerms == null || searchTerms.isEmpty()) {
            return Boolean.FALSE;
        }

        boolean orQuery = false;
        if (args.length == 3) {
            orQuery = orQueryOI.get(args[2].get());
        }

        if (orQuery) {
            for (String term : searchTerms) {
                if (query.contains(term)) {
                    return Boolean.TRUE;
                }
            }
            return Boolean.FALSE;
        } else {
            for (String term : searchTerms) {
                if (!query.contains(term)) {
                    return Boolean.FALSE;
                }
            }
            return Boolean.TRUE;
        }
    }

    @Override
    public String getDisplayString(String[] args) {
        return "str_contains(" + StringUtils.join(args, ',') + ')';
    }


}
