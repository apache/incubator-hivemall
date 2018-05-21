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
package hivemall.tools.text;

import java.text.Normalizer;

import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;

@Description(name = "normalize_unicode",
        value = "_FUNC_(string str [, string form]) - Transforms `str` with the specified normalization form. "
                + "The `form` takes one of NFC (default), NFD, NFKC, or NFKD",
        extended = "SELECT normalize_unicode('ﾊﾝｶｸｶﾅ','NFKC');\n" + " ハンカクカナ\n" + "\n"
                + "SELECT normalize_unicode('㈱㌧㌦Ⅲ','NFKC');\n" + " (株)トンドルIII")
@UDFType(deterministic = true, stateful = false)
public final class NormalizeUnicodeUDF extends UDF {

    @Nullable
    public String evaluate(@Nullable String str) {
        return evaluate(str, null);
    }

    @Nullable
    public String evaluate(@Nullable String str, @Nullable String form) {
        if (str == null) {
            return null;
        }
        if (form == null) {
            return Normalizer.normalize(str, Normalizer.Form.NFC);
        } else if ("NFC".equals(form)) {
            return Normalizer.normalize(str, Normalizer.Form.NFC);
        } else if ("NFD".equals(form)) {
            return Normalizer.normalize(str, Normalizer.Form.NFD);
        } else if ("NFKC".equals(form)) {
            return Normalizer.normalize(str, Normalizer.Form.NFKC);
        } else if ("NFKD".equals(form)) {
            return Normalizer.normalize(str, Normalizer.Form.NFKD);
        } else {
            return Normalizer.normalize(str, Normalizer.Form.NFC);
        }
    }

}
