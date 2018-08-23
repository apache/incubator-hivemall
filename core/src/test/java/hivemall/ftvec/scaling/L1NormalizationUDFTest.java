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

import hivemall.utils.hadoop.WritableUtils;
import hivemall.utils.math.MathUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.io.Text;
import org.junit.Test;

import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class L1NormalizationUDFTest {

    @Test
    public void test() throws HiveException {
        L1NormalizationUDF udf = new L1NormalizationUDF();

        assertEquals(null, udf.evaluate(null));

        assertEquals(WritableUtils.val(new String[] {}),
            udf.evaluate(WritableUtils.val(new String[] {})));

        assertEquals(WritableUtils.val(new String[] {"aaa:1.0"}),
            udf.evaluate(WritableUtils.val(new String[] {"aaa"})));

        assertEquals(WritableUtils.val(new String[] {"aaa:1.0"}),
            udf.evaluate(WritableUtils.val(new String[] {"aaa:1"})));

        assertEquals(WritableUtils.val(new String[] {"aaa:1.0"}),
            udf.evaluate(WritableUtils.val(new String[] {"aaa:1.0"})));

        float[] normalized = MathUtils.l1normalize(new float[] {1.0f, 0.5f});
        assertEquals(
            WritableUtils.val(new String[] {"aaa:" + normalized[0], "bbb:" + normalized[1]}),
            udf.evaluate(WritableUtils.val(new String[] {"aaa:1.0", "bbb:0.5"})));

        normalized = MathUtils.l1normalize(new float[] {1.0f, -0.5f});
        assertEquals(
            WritableUtils.val(new String[] {"aaa:" + normalized[0], "bbb:" + normalized[1]}),
            udf.evaluate(WritableUtils.val(new String[] {"aaa:1.0", "bbb:-0.5"})));

        normalized = MathUtils.l1normalize(new float[] {1.0f, 2.0f, 3.0f});
        assertEquals(
            WritableUtils.val(new String[] {"1:123:" + normalized[0], "2:456:" + normalized[1],
                    "3:789:" + normalized[2]}),
            udf.evaluate(WritableUtils.val(new String[] {"1:123:1", "2:456:2", "3:789:3"})));

        List<Text> expected = udf.evaluate(WritableUtils.val(new String[] {"bbb:-0.5", "aaa:1.0"}));
        Collections.sort(expected);
        List<Text> actual = udf.evaluate(WritableUtils.val(new String[] {"aaa:1.0", "bbb:-0.5"}));
        Collections.sort(actual);
        assertEquals(expected, actual);
    }

}
