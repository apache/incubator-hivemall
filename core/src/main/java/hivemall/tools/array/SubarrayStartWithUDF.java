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
package hivemall.tools.array;

import java.util.List;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

@Description(name = "subarray_startwith",
        value = "_FUNC_(array<int|text> original, int|text key)"
                + " - Returns an array that starts with the specified key",
        extended = "SELECT subarray_startwith(array(1,2,3,4), 2);\n" + " [2,3,4]")
@UDFType(deterministic = true, stateful = false)
public class SubarrayStartWithUDF extends UDF {

    public List<IntWritable> evaluate(List<IntWritable> original, IntWritable key) {
        if (original == null) {
            return null;
        }
        int fromIndex = original.indexOf(key);
        if (fromIndex == -1) {
            return null;
        }
        int toIndex = original.size();
        return original.subList(fromIndex, toIndex);
    }

    public List<Text> evaluate(List<Text> original, Text key) {
        if (original == null) {
            return null;
        }
        int fromIndex = original.indexOf(key);
        if (fromIndex == -1) {
            return null;
        }
        int toIndex = original.size();
        return original.subList(fromIndex, toIndex);
    }

}
