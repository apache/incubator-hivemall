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

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.StringWriter;
import java.util.zip.GZIPOutputStream;

import opennlp.maxent.io.GISModelWriter;
import opennlp.model.AbstractModel;

public class SepDelimitedTextGISModelWriter extends GISModelWriter {
    BufferedWriter output;
    String separator;
    StringWriter sw;

    /**
     * Constructor which takes a GISModel and a File and prepares itself to write the model to that file. Detects whether the file is gzipped or not
     * based on whether the suffix contains ".gz".
     *
     * @param model The GISModel which is to be persisted.
     * @param f The File in which the model is to be persisted.
     */
    public SepDelimitedTextGISModelWriter(AbstractModel model, String separator)
            throws IOException, FileNotFoundException {
        super(model);
        sw = new StringWriter();
        output = new BufferedWriter(sw);
        this.separator = separator;
    }

    /**
     * Constructor which takes a GISModel and a BufferedWriter and prepares itself to write the model to that writer.
     *
     * @param model The GISModel which is to be persisted.
     * @param bw The BufferedWriter which will be used to persist the model.
     */
    public SepDelimitedTextGISModelWriter(AbstractModel model, BufferedWriter bw) {
        super(model);
        output = bw;
    }

    public void writeUTF(String s) throws java.io.IOException {
        output.write(s);
        output.write(this.separator);
    }

    public void writeInt(int i) throws java.io.IOException {
        output.write(Integer.toString(i));
        output.write(this.separator);
    }

    public void writeDouble(double d) throws java.io.IOException {
        output.write(Double.toString(d));
        output.write(this.separator);
    }

    public void close() throws java.io.IOException {
        output.flush();
        output.close();
    }

    public String toString() {
        StringBuffer sb = sw.getBuffer();
        return sb.toString();
    }
}
