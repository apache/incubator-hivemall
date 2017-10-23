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
package hivemall.utils.lambda;

import static hivemall.utils.lambda.Throwing.rethrow;

import java.io.IOException;
import java.util.Arrays;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.ExpectedException;

public class ThrowingTest {

    @Rule
    public ExpectedException thrown = ExpectedException.none();

    @Test
    public void testRethrow() {
        thrown.expect(IOException.class);
        thrown.expectMessage("i=3");

        Arrays.asList(1, 2, 3).forEach(rethrow(e -> {
            int i = e.intValue();
            if (i == 3) {
                throw new IOException("i=" + i);
            }
        }));
    }

    @Test(expected = IOException.class)
    public void testSneakyThrow() {
        Throwing.sneakyThrow(new IOException());
    }

    @Test
    public void testThrowingConsumer() {
        thrown.expect(IOException.class);
        thrown.expectMessage("i=3");

        Arrays.asList(1, 2, 3).forEach((ThrowingConsumer<Integer>) e -> {
            int i = e.intValue();
            if (i == 3) {
                throw new IOException("i=" + i);
            }
        });
    }

}
