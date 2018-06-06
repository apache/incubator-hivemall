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
package hivemall.tools.sanity;

import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@Description(name = "raise_error", value = "_FUNC_() or _FUNC_(string msg) - Throws an error",
        extended = "SELECT product_id, price, raise_error('Found an invalid record') FROM xxx WHERE price < 0.0")
@UDFType(deterministic = false, stateful = false)
public class RaiseErrorUDF extends GenericUDF {

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 0 && argOIs.length != 1) {
            throw new UDFArgumentLengthException(
                "Expected one or two arguments for raise_error UDF: " + argOIs.length);
        }

        return PrimitiveObjectInspectorFactory.writableBooleanObjectInspector;
    }

    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        if (arguments.length == 1) {
            Object arg0 = arguments[0].get();
            if (arg0 == null) {
                throw new HiveException();
            }
            String msg = arg0.toString();
            throw new HiveException(msg);
        } else {
            throw new HiveException();
        }
    }

    @Override
    public String getDisplayString(String[] children) {
        return "raise_error(" + StringUtils.join(children, ',') + ')';
    }

}
