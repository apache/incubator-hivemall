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

package hivemall.opennlp.tools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.StringReader;

import org.apache.hadoop.io.Text;

import opennlp.model.DataReader;

public class SepDelimitedTextDataReader implements DataReader {
    private BufferedReader input;
    String separator = "@";
    StringReader sr;

    public SepDelimitedTextDataReader(Text in) {
        String i = in.toString().replaceAll(separator, "\n");
        sr = new StringReader(i);
        input = new BufferedReader(sr);
        try {
            // read model name, should be GIS in our case
            input.readLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double readDouble() throws IOException {
        return Double.parseDouble(input.readLine());
    }

    public int readInt() throws IOException {
        String line = input.readLine();
        return Integer.parseInt(line);
    }

    public String readUTF() throws IOException {
        return input.readLine();
    }
}
