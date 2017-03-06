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
package hivemall.utils.stream;

import java.io.IOException;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public class StreamUtilsTest {

    @Test
    public void testToArrayIntStream() throws IOException {
        Random rand = new Random(43L);
        int[] src = new int[9999];
        for (int i = 0; i < src.length; i++) {
            src[i] = rand.nextInt();
        }

        IntStream stream = StreamUtils.toArrayIntStream(src);
        IntIterator itor = stream.iterator();
        int i = 0;
        while (itor.hasNext()) {
            Assert.assertEquals(src[i], itor.next());
            i++;
        }
        Assert.assertFalse(itor.hasNext());
        Assert.assertEquals(src.length, i);

        itor = stream.iterator();
        i = 0;
        while (itor.hasNext()) {
            Assert.assertEquals(src[i], itor.next());
            i++;
        }
        Assert.assertFalse(itor.hasNext());
        Assert.assertEquals(src.length, i);
    }


    @Test
    public void testToCompressedIntStreamIntArray() throws IOException {
        Random rand = new Random(43L);
        int[] src = new int[9999];
        for (int i = 0; i < src.length; i++) {
            src[i] = rand.nextInt();
        }

        IntStream stream = StreamUtils.toCompressedIntStream(src);
        IntIterator itor = stream.iterator();
        int i = 0;
        while (itor.hasNext()) {
            Assert.assertEquals(src[i], itor.next());
            i++;
        }
        Assert.assertFalse(itor.hasNext());
        Assert.assertEquals(src.length, i);

        itor = stream.iterator();
        i = 0;
        while (itor.hasNext()) {
            Assert.assertEquals(src[i], itor.next());
            i++;
        }
        Assert.assertFalse(itor.hasNext());
        Assert.assertEquals(src.length, i);
    }

}
