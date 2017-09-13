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
package hivemall.unsupervised;

import hivemall.utils.collections.maps.Int2DoubleOpenHashTable;
import hivemall.utils.collections.maps.Int2IntOpenHashTable;
import hivemall.utils.hadoop.HiveUtils;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.*;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

import java.util.*;

public final class AliasTableBuilderUDTF extends GenericUDTF {
    private MapObjectInspector negativeTableOI;
    private PrimitiveObjectInspector negativeTableKeyOI;
    private PrimitiveObjectInspector negativeTableValueOI;
    private PrimitiveObjectInspector numSplitOI;

    private int numSplit;
    private int numVocab;
    private List<String> index2word;
    private Int2IntOpenHashTable A;
    private Int2DoubleOpenHashTable S;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (!(argOIs.length >= 2)) {
            throw new UDFArgumentException(
                "_FUNC_(map<string, double>, int) takes at least two arguments");
        }

        this.negativeTableOI = HiveUtils.asMapOI(argOIs[0]);
        this.negativeTableKeyOI = HiveUtils.asStringOI(negativeTableOI.getMapKeyObjectInspector());
        this.negativeTableValueOI = HiveUtils.asDoubleOI(negativeTableOI.getMapValueObjectInspector());

        this.numSplitOI = HiveUtils.asIntCompatibleOI(argOIs[1]);

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("k");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableIntObjectInspector);

        fieldNames.add("word");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);

        fieldNames.add("p");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableDoubleObjectInspector);

        fieldNames.add("other");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {

        this.numSplit = PrimitiveObjectInspectorUtils.getInt(args[1], this.numSplitOI);
        if (!(numSplit >= 1)) {
            throw new UDFArgumentException("Illegal numSplit value: " + numSplit);
        }

        List<String> index2word = new ArrayList<>();
        List<Double> unnormalizedProb = new ArrayList<>();

        double denom = 0;
        for (Map.Entry<?, ?> entry : this.negativeTableOI.getMap(args[0]).entrySet()) {
            String word = PrimitiveObjectInspectorUtils.getString(entry.getKey(),
                this.negativeTableKeyOI);
            double v = PrimitiveObjectInspectorUtils.getDouble(entry.getValue(),
                this.negativeTableValueOI);
            unnormalizedProb.add(v);
            index2word.add(word);
            denom += v;
        }
        int V = index2word.size();

        Int2DoubleOpenHashTable S = new Int2DoubleOpenHashTable(V);
        Int2IntOpenHashTable A = new Int2IntOpenHashTable(V);

        Queue<Integer> higherBin = new ArrayDeque<>();
        Queue<Integer> lowerBin = new ArrayDeque<>();

        for (int i = 0; i < V; i++) {
            double v = V * unnormalizedProb.get(i) / denom;
            S.put(i, v);
            if (v > 1.) {
                higherBin.add(i);
            } else {
                lowerBin.add(i);
            }
        }

        while (lowerBin.size() > 0 && higherBin.size() > 0) {
            int low = lowerBin.remove();
            int high = higherBin.remove();
            A.put(low, high);
            S.put(high, S.get(high) - 1. + S.get(low));
            if (S.get(high) < 1.) {
                lowerBin.add(high);
            } else {
                higherBin.add(high);
            }
        }

        this.numVocab = V;
        this.A = A;
        this.S = S;
        this.index2word = index2word;
    }

    @Override
    public void close() throws HiveException {
        for (int i = 0; i < this.numVocab; i++) {
            Object[] res = new Object[4];
            res[0] = new IntWritable(i % this.numSplit);
            res[1] = new Text(this.index2word.get(i));
            res[2] = new DoubleWritable(this.S.get(i));
            if (this.A.get(i) == -1) {
                res[3] = new Text();
            } else {
                res[3] = new Text(this.index2word.get(this.A.get(i)));
            }
            forward(res);
        }
    }
}
