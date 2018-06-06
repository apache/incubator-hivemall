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

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;

@Description(name = "assert",
        value = "_FUNC_(boolean condition) or _FUNC_(boolean condition, string errMsg)"
                + "- Throws HiveException if condition is not met",
        extended = "SELECT count(1) FROM stock_price WHERE assert(price > 0.0);\n"
                + "SELECT count(1) FROM stock_price WHERE assert(price > 0.0, 'price MUST be more than 0.0')")
@UDFType(deterministic = false, stateful = false)
public final class AssertUDF extends UDF {

    public boolean evaluate(boolean condition) throws HiveException {
        if (!condition) {
            throw new HiveException();
        }
        return true;
    }

    public boolean evaluate(boolean condition, String errMsg) throws HiveException {
        if (!condition) {
            throw new HiveException(errMsg);
        }
        return true;
    }

}
