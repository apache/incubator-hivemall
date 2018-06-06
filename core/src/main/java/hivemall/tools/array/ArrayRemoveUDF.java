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

@Description(name = "array_remove",
        value = "_FUNC_(array<int|text> original, int|text|array<int> target)"
                + " - Returns an array that the target is removed " + "from the original array",
        extended = "SELECT array_remove(array(1,null,3),array(null));\n" + " [3]\n" + "\n"
                + "SELECT array_remove(array(\"aaa\",\"bbb\"),\"bbb\");\n" + " [\"aaa\"]")
@UDFType(deterministic = true, stateful = false)
public class ArrayRemoveUDF extends UDF {

    public List<IntWritable> evaluate(List<IntWritable> original, IntWritable target) {
        while (original.remove(target));
        return original;
    }

    public List<IntWritable> evaluate(List<IntWritable> original, List<IntWritable> targets) {
        original.removeAll(targets);
        return original;
    }

    public List<Text> evaluate(List<Text> original, Text target) {
        while (original.remove(target));
        return original;
    }

}
