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
package hivemall.mf;

import hivemall.fm.Feature;
import hivemall.fm.StringFeature;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class CofactorizationUDTFTest {

    CofactorizationUDTF udtf;

    @Before
    public void setUp() throws HiveException {
        this.udtf = new CofactorizationUDTF();
    }


    @Test
    public void testCreateNnzFeatureArray() throws HiveException {
        Feature item1 = new StringFeature("item1", 0);
        Feature item2 = new StringFeature("item2", 1);
        Feature item3 = new StringFeature("item3", 0);

        Feature[] expected = new Feature[]{item2};

        Feature[] actual = udtf.createNnzFeatureArray(new Feature[]{
                item1, item2, item3
        });

        Assert.assertArrayEquals(actual, expected);
    }
}
