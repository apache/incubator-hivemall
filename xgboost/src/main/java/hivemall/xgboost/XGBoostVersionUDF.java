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
package hivemall.xgboost;

import hivemall.xgboost.utils.XGBoostUtils;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;

@Description(name = "xgboost_version", value = "_FUNC_() - Returns the version of xgboost",
        extended = "SELECT xgboost_version();")
@UDFType(deterministic = true, stateful = false)
public final class XGBoostVersionUDF extends UDF {

    @Nullable
    private String version;

    @Nonnull
    public String evaluate() throws HiveException {
        if (version == null) {
            this.version = XGBoostUtils.getVersion();
        }
        return version;
    }

}
