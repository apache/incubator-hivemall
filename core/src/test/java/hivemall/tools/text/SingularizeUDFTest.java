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

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class SingularizeUDFTest {

    private SingularizeUDF udf;

    @Before
    public void setUp() {
        this.udf = new SingularizeUDF();
    }

    @Test
    public void testNull() {
        Assert.assertEquals(null, udf.evaluate(null));
    }

    @Test
    public void testEmpty() {
        Assert.assertEquals("", udf.evaluate(""));
    }

    @Test
    public void testUnchanged() {
        Assert.assertEquals("christmas", udf.evaluate("christmas"));
    }

    @Test
    public void testCompound() {
        Assert.assertEquals("mother-in-law", udf.evaluate("mothers-in-law"));
    }

    @Test
    public void testTailSingleQuote() {
        Assert.assertEquals("dog's", udf.evaluate("dogs'"));
    }

    @Test
    public void testIrregular() {
        Assert.assertEquals("child", udf.evaluate("children"));
    }

    @Test
    public void testRule() {
        Assert.assertEquals("apple", udf.evaluate("apples"));
        Assert.assertEquals("bus", udf.evaluate("buses"));
        Assert.assertEquals("candy", udf.evaluate("candies"));
    }

}
