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
package hivemall.tools.datetime;

import java.util.UUID;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

//@formatter:off
@Description(name = "sessionize",
        value = "_FUNC_(long timeInSec, long thresholdInSec [, String subject])"
                + "- Returns a UUID string of a session.",
        extended = "SELECT \n" + 
                "  sessionize(time, 3600, ip_addr) as session_id, \n" + 
                "  time, ip_addr\n" + 
                "FROM (\n" + 
                "  SELECT time, ipaddr \n" + 
                "  FROM weblog \n" + 
                "  DISTRIBUTE BY ip_addr, time SORT BY ip_addr, time DESC\n" + 
                ") t1")
//@formatter:on
@UDFType(deterministic = false, stateful = true)
public final class SessionizeUDF extends UDF {

    private long lastTime;
    @Nullable
    private Text lastSubject;
    @Nonnull
    private final Text sessionId = new Text();

    @Nullable
    public Text evaluate(@Nullable LongWritable time, @Nullable LongWritable threshold) {
        if (time == null || threshold == null) {
            return null;
        }

        final long thisTime = time.get();
        final long diff = thisTime - lastTime;
        if (diff < threshold.get()) {
            this.lastTime = thisTime;
            return sessionId;
        }

        sessionId.set(UUID.randomUUID().toString());
        this.lastTime = time.get();
        return sessionId;
    }

    @Nullable
    public Text evaluate(@Nullable LongWritable time, @Nullable LongWritable threshold,
            @Nullable Text subject) {
        if (time == null || threshold == null || subject == null) {
            return null;
        }

        if (subject.equals(lastSubject)) {
            final long thisTime = time.get();
            final long diff = thisTime - lastTime;
            if (diff < threshold.get()) {
                this.lastTime = thisTime;
                return sessionId;
            }
        } else {
            if (lastSubject == null) {
                lastSubject = new Text(subject);
            } else {
                lastSubject.set(subject);
            }
        }

        sessionId.set(UUID.randomUUID().toString());
        this.lastTime = time.get();
        return sessionId;
    }

}
