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
package hivemall.ftvec.scaling;

import static org.junit.Assert.assertEquals;
import hivemall.utils.hadoop.WritableUtils;

import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Before;
import org.junit.Test;

public class RescaleUDFTest {

    RescaleUDF udf = null;

    @Before
    public void init() {
        udf = new RescaleUDF();
    }

    @Test
    public void test() throws Exception {
        assertEquals(WritableUtils.val(0.5f), udf.evaluate(1f, 1f, 1f));
        assertEquals(WritableUtils.val(0.5f), udf.evaluate(0.1d, 0.1d, 0.1d));
        assertEquals(WritableUtils.val("1:0.5"), udf.evaluate("1:1", 1f, 1f));
        assertEquals(WritableUtils.val("1:0.5"), udf.evaluate("1:1", 0.1d, 0.1d));
    }

    @Test(expected = HiveException.class)
    public void testFloatMinIsNull() throws Exception {
        udf.evaluate(1f, null, 1f);
    }

    @Test(expected = HiveException.class)
    public void testFloatMaxIsNull() throws Exception {
        udf.evaluate(1f, 1f, null);
    }

    @Test(expected = HiveException.class)
    public void testDoubleMinIsNull() throws Exception {
        udf.evaluate(0.1, null, 0.1);
    }

    @Test(expected = HiveException.class)
    public void testDoubleMaxIsNull() throws Exception {
        udf.evaluate(0.1, 0.1, null);
    }

    @Test(expected = HiveException.class)
    public void testBothNull() throws Exception {
        udf.evaluate(0.1, null, null);
    }

    @Test(expected = HiveException.class)
    public void testIllegalArgumentException1() throws Exception {
        udf.evaluate("1:", 0.1d, 0.1d);
    }

    @Test(expected = HiveException.class)
    public void testStringMaxNull() throws Exception {
        udf.evaluate("1:1", null, 1d);
    }

    @Test(expected = HiveException.class)
    public void testStringMinNull() throws Exception {
        udf.evaluate("1:1", 1d, null);
    }

    @Test(expected = HiveException.class)
    public void testCannotParseNumber() throws Exception {
        udf.evaluate("1:string", 0.1d, 0.1d);
    }

    @Test
    public void testMinMaxEquals() throws Exception {
        assertEquals(WritableUtils.val(0.5f), udf.evaluate(0.1d, 0.1d, 0.1d));
    }

    @Test
    public void testMinMaxCornercase() throws Exception {
        assertEquals(WritableUtils.val(1.0f), udf.evaluate(1.1f, 0.0f, 1.0f));
        assertEquals(WritableUtils.val(0.0f), udf.evaluate(-0.1f, 0.0f, 1.0f));
        assertEquals(WritableUtils.val(1.0f), udf.evaluate(4.1f, 0.0f, 3.0f));
        assertEquals(WritableUtils.val(0.0f), udf.evaluate(-2.1f, -1.0f, 1.0f));
    }

    @Test(expected = HiveException.class)
    public void testInvalidMinMax() throws Exception {
        udf.evaluate(0.1d, 0.2d, 0.1d);
    }

}
