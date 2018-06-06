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

import static hivemall.utils.hadoop.WritableUtils.val;

import hivemall.TestUtils;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.io.Text;
import org.junit.Assert;
import org.junit.Test;

public class SessionizeUDFTest {

    @Test
    public void testTwoArgs() {
        SessionizeUDF udf = new SessionizeUDF();

        Text session1 = new Text(udf.evaluate(val(30L), val(10L)));
        Assert.assertNotNull(session1);

        Text session2 = new Text(udf.evaluate(val(35L), val(10L)));
        Assert.assertEquals(session1, session2);

        Text session3 = new Text(udf.evaluate(val(40L), val(10L)));
        Assert.assertEquals(session2, session3);

        Text session4 = new Text(udf.evaluate(val(50L), val(10L)));
        Assert.assertNotEquals(session3, session4);
    }

    @Test
    public void testThreeArgs() {
        SessionizeUDF udf = new SessionizeUDF();

        Text session1 = new Text(udf.evaluate(val(30L), val(10L), val("subject1")));
        Assert.assertNotNull(session1);

        Text session2 = new Text(udf.evaluate(val(35L), val(10L), val("subject1")));
        Assert.assertEquals(session1, session2);

        Text session3 = new Text(udf.evaluate(val(40L), val(10L), val("subject2")));
        Assert.assertNotEquals(session2, session3);

        Text session4 = new Text(udf.evaluate(val(45L), val(10L), val("subject2")));
        Assert.assertEquals(session3, session4);
    }

    @Test
    public void testSerialization() throws HiveException {
        SessionizeUDF udf = new SessionizeUDF();

        udf.evaluate(val((long) (System.currentTimeMillis() / 1000.0d)), val(30L));
        udf.evaluate(val((long) (System.currentTimeMillis() / 1000.0d)), val(30L));

        byte[] serialized = TestUtils.serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, SessionizeUDF.class);
    }
}
