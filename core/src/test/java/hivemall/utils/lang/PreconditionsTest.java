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
package hivemall.utils.lang;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Assert;
import org.junit.Test;

public class PreconditionsTest {

    @Test(expected = UDFArgumentException.class)
    public void testCheckNotNullTClassOfE() throws UDFArgumentException {
        Preconditions.checkNotNull(null, UDFArgumentException.class);
    }

    @Test
    public void testCheckNotNullTClassOfE2() {
        final String msg = "safdfvzfd";
        try {
            Preconditions.checkNotNull(null, msg, UDFArgumentException.class);
        } catch (UDFArgumentException e) {
            if (e.getMessage().equals(msg)) {
                return;
            }
        }
        Assert.fail("should not reach");
    }

    @Test(expected = HiveException.class)
    public void testCheckArgumentBooleanClassOfE() throws HiveException {
        Preconditions.checkArgument(false, HiveException.class);
    }

    @Test
    public void testCheckArgumentBooleanClassOfE2() {
        final String msg = "safdfvzfd";
        try {
            Preconditions.checkArgument(false, HiveException.class, msg);
        } catch (HiveException e) {
            if (e.getMessage().equals(msg)) {
                return;
            }
        }
        Assert.fail("should not reach");
    }

}
