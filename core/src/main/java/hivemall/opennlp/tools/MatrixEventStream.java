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

import java.util.Iterator;
import java.util.List;

import hivemall.math.matrix.Matrix;
import hivemall.smile.data.Attribute;
import hivemall.smile.data.Attribute.AttributeType;
import opennlp.model.AbstractEventStream;
import opennlp.model.Event;

public class MatrixEventStream extends AbstractEventStream {
    Matrix ds;
    int[] y;
    Attribute[] attrs;
    int rowNum;
    Event next;

    public MatrixEventStream(Matrix ds, int[] y, Attribute[] attrs) {
        this.ds = ds;
        this.y = y;
        this.attrs = attrs;
        rowNum = 0;

        if (rowNum < ds.numRows()) {
            next = createEvent(ds.getRow(rowNum), y[rowNum]);
        }
    }

    /**
     * Returns the next Event object held in this EventStream. Each call to nextEvent advances the EventStream.
     *
     * @return the Event object which is next in this EventStream
     */
    public Event next() {
        while (next == null && (rowNum < ds.numRows())) {
            next = createEvent(ds.getRow(rowNum), y[rowNum]);
        }

        Event current = next;
        if (rowNum < ds.numRows()) {
            next = createEvent(ds.getRow(rowNum), y[rowNum]);
        } else {
            next = null;
        }
        return current;
    }

    /**
     * Test whether there are any Events remaining in this EventStream.
     *
     * @return true if this EventStream has more Events
     */
    public boolean hasNext() {
        while (next == null && rowNum < ds.numRows()) {
            next = createEvent(ds.getRow(rowNum), y[rowNum]);
        }
        return next != null;
    }

    private Event createEvent(double[] obs, int y) {
        rowNum++;
        if (obs == null) {
            return null;
        } else {
            String[] names = new String[obs.length];
            float[] values = new float[obs.length];
            for (int i = 0; i < obs.length; i++) {
                if (attrs[i].type == AttributeType.NOMINAL) {
                    names[i] = i + "_" + String.valueOf(obs[i]).toString();
                    values[i] = 1.f;
                } else {
                    names[i] = String.valueOf(i);
                    values[i] = Double.valueOf(obs[i]).floatValue();
                }
            }
            return new Event(String.valueOf(y), names, values);
        }
    }
}
