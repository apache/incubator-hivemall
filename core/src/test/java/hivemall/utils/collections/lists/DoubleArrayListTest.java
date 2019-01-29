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
package hivemall.utils.collections.lists;

import org.junit.Assert;
import org.junit.Test;

public class DoubleArrayListTest {

    @Test
    public void testExpandZeroSizedList() {
        DoubleArrayList list = new DoubleArrayList(0);
        list.add(100.d);
        Assert.assertEquals(1, list.size());
        Assert.assertEquals(100.d, list.get(0), 1e-8);
    }

}
