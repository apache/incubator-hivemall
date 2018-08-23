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
package hivemall.tools.mapred;

import hivemall.utils.hadoop.HadoopUtils;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.LongWritable;

@Description(name = "rownum",
        value = "_FUNC_() - Returns a generated row number `sprintf(`%d%04d`,sequence,taskId)` in long",
        extended = "SELECT rownum() as rownum, xxx from ...")
@UDFType(deterministic = false, stateful = true)
public final class RowNumberUDF extends UDF {

    private long sequence;
    private int taskId;
    @Nonnull
    private final LongWritable result;

    public RowNumberUDF() {
        this.sequence = 0L;
        this.taskId = -1;
        this.result = new LongWritable(Double.doubleToLongBits(Double.NaN));
    }

    @Nonnull
    public LongWritable evaluate() throws HiveException {
        if (taskId == -1) {
            this.taskId = HadoopUtils.getTaskId() + 1;
            if (taskId > 9999) {
                throw new HiveException(
                    "TaskId out of range `" + taskId + "`. rownum() supports 9999 tasks at max");
            }
        }
        sequence++;

        String rowid = String.format("%d%04d", sequence, taskId);
        final long l;
        try {
            l = Long.parseLong(rowid);
        } catch (NumberFormatException e) {
            throw new HiveException("failed to parse `" + rowid + "` as long", e);
        }

        result.set(l);
        return result;
    }
}
