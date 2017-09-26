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
package hivemall.embedding;

import hivemall.utils.collections.maps.Int2FloatOpenHashTable;
import hivemall.utils.collections.maps.Int2IntOpenHashTable;
import hivemall.utils.hadoop.HiveUtils;

import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Queue;
import java.util.ArrayDeque;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.MapObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;

public final class AliasTableBuilderUDTF extends GenericUDTF {
    private MapObjectInspector negativeTableOI;
    private PrimitiveObjectInspector negativeTableKeyOI;
    private PrimitiveObjectInspector negativeTableValueOI;

    private int numVocab;
    private List<String> index2word;
    private Int2IntOpenHashTable A;
    private Int2FloatOpenHashTable S;

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (!(argOIs.length >= 1)) {
            throw new UDFArgumentException(
                "_FUNC_(map<string, double>) takes at least one argument");
        }

        this.negativeTableOI = HiveUtils.asMapOI(argOIs[0]);
        this.negativeTableKeyOI = HiveUtils.asStringOI(negativeTableOI.getMapKeyObjectInspector());
        this.negativeTableValueOI = HiveUtils.asFloatingPointOI(negativeTableOI.getMapValueObjectInspector());

        List<String> fieldNames = new ArrayList<>();
        List<ObjectInspector> fieldOIs = new ArrayList<>();

        fieldNames.add("word");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);

        fieldNames.add("p");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableFloatObjectInspector);

        fieldNames.add("other");
        fieldOIs.add(PrimitiveObjectInspectorFactory.writableStringObjectInspector);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(Object[] args) throws HiveException {

        final List<String> index2word = new ArrayList<>();
        final List<Float> unnormalizedProb = new ArrayList<>();

        float denom = 0;
        for (Map.Entry<?, ?> entry : negativeTableOI.getMap(args[0]).entrySet()) {
            String word = PrimitiveObjectInspectorUtils.getString(entry.getKey(),
                negativeTableKeyOI);
            index2word.add(word);

            float v = PrimitiveObjectInspectorUtils.getFloat(entry.getValue(), negativeTableValueOI);
            unnormalizedProb.add(v);
            denom += v;
        }

        this.numVocab = index2word.size();
        this.index2word = index2word;

        createAliasTable(numVocab, denom, unnormalizedProb);
    }

    private void createAliasTable(final int V, final float denom, final List<Float> unnormalizedProb) {
        Int2FloatOpenHashTable S = new Int2FloatOpenHashTable(V);
        Int2IntOpenHashTable A = new Int2IntOpenHashTable(V);

        final Queue<Integer> higherBin = new ArrayDeque<>();
        final Queue<Integer> lowerBin = new ArrayDeque<>();

        for (int i = 0; i < V; i++) {
            float v = V * unnormalizedProb.get(i) / denom;
            S.put(i, v);
            if (v > 1.f) {
                higherBin.add(i);
            } else {
                lowerBin.add(i);
            }
        }

        while (lowerBin.size() > 0 && higherBin.size() > 0) {
            int low = lowerBin.remove();
            int high = higherBin.remove();
            A.put(low, high);
            S.put(high, S.get(high) - 1.f + S.get(low));
            if (S.get(high) < 1.f) {
                lowerBin.add(high);
            } else {
                higherBin.add(high);
            }
        }
        this.A = A;
        this.S = S;
    }

    @Override
    public void close() throws HiveException {
        Text word = new Text();
        FloatWritable pro = new FloatWritable();
        Text otherWord = new Text();

        Object[] res = new Object[3];
        res[0] = word;
        res[1] = pro;
        res[2] = otherWord;


        for (int i = 0; i < numVocab; i++) {
            word.set(index2word.get(i));
            pro.set(S.get(i));
            if (A.get(i) == -1) {
                otherWord.set("");
            } else {
                otherWord.set(index2word.get(A.get(i)));
            }
            forward(res);
        }
    }
}
