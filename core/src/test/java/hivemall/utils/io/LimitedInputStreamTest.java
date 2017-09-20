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
package hivemall.utils.io;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;

import org.junit.Assert;
import org.junit.Test;

public class LimitedInputStreamTest {

    @Test
    public void testExactSize() throws IOException {
        String expected = "abcdef";
        int len = expected.length();

        InputStream is = new FastByteArrayInputStream(expected.getBytes());
        LimitedInputStream isLimited = new LimitedInputStream(is, len);

        Reader reader = new InputStreamReader(isLimited);
        BufferedReader br = new BufferedReader(reader);

        char[] buf = new char[len];
        br.read(buf);

        Assert.assertTrue(expected.equals(new String(buf)));

        br.close();
    }

    @Test
    public void testLooseSize() throws IOException {
        String expected = "abcdef";
        int len = expected.length();

        InputStream is = new FastByteArrayInputStream(expected.getBytes());
        LimitedInputStream isLimited = new LimitedInputStream(is, len + 100); // large enough

        Reader reader = new InputStreamReader(isLimited);
        BufferedReader br = new BufferedReader(reader);

        char[] buf = new char[len];
        br.read(buf);

        Assert.assertTrue(expected.equals(new String(buf)));

        br.close();
    }

    @Test(expected = IOException.class)
    public void testExceed() throws IOException {
        String expected = "abcdef";
        int len = expected.length();

        InputStream is = new FastByteArrayInputStream(expected.getBytes());
        LimitedInputStream isLimited = new LimitedInputStream(is, len - 1); // not enough

        Reader reader = new InputStreamReader(isLimited);
        BufferedReader br = new BufferedReader(reader);

        char[] buf = new char[len];
        br.read(buf);

        br.close();
    }

    @Test(expected = NullPointerException.class)
    public void testNullInputStream() throws NullPointerException, IOException {
        new LimitedInputStream(null, 100).close();
    }

}
