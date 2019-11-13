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
package hivemall;

import java.io.ByteArrayOutputStream;
import java.io.IOException;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Utilities;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.Collector;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hive.com.esotericsoftware.kryo.Kryo;
import org.apache.hive.com.esotericsoftware.kryo.io.Input;
import org.apache.hive.com.esotericsoftware.kryo.io.Output;

public final class TestUtils {

    public static <T extends GenericUDF> void testGenericUDFSerialization(@Nonnull Class<T> clazz,
            @Nonnull ObjectInspector[] ois, @Nonnull Object[] row)
            throws HiveException, IOException {
        final T udf;
        try {
            udf = clazz.newInstance();
        } catch (InstantiationException | IllegalAccessException e) {
            throw new HiveException(e);
        }

        udf.initialize(ois);

        // serialization after initialization
        byte[] serialized = serializeObjectByKryo(udf);
        deserializeObjectByKryo(serialized, clazz);

        int size = row.length;
        GenericUDF.DeferredObject[] rowDeferred = new GenericUDF.DeferredObject[size];
        for (int i = 0; i < size; i++) {
            rowDeferred[i] = new GenericUDF.DeferredJavaObject(row[i]);
        }

        udf.evaluate(rowDeferred);

        // serialization after evaluating row
        serialized = serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, clazz);

        udf.close();
    }

    @SuppressWarnings("deprecation")
    public static <T extends GenericUDTF> void testGenericUDTFSerialization(@Nonnull Class<T> clazz,
            @Nonnull ObjectInspector[] ois, @Nonnull Object[][] rows) throws HiveException {
        final T udtf;
        try {
            udtf = clazz.newInstance();
        } catch (InstantiationException | IllegalAccessException e) {
            throw new HiveException(e);
        }

        udtf.initialize(ois);

        // serialization after initialization
        byte[] serialized = serializeObjectByKryo(udtf);
        deserializeObjectByKryo(serialized, clazz);

        udtf.setCollector(new Collector() {
            public void collect(Object input) throws HiveException {
                // noop
            }
        });

        for (Object[] row : rows) {
            udtf.process(row);
        }

        // serialization after processing row
        serialized = serializeObjectByKryo(udtf);
        TestUtils.deserializeObjectByKryo(serialized, clazz);

        udtf.close();
    }

    @Nonnull
    public static byte[] serializeObjectByKryo(@Nonnull Object obj) {
        Kryo kryo = getKryo();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Output output = new Output(bos);
        kryo.writeObject(output, obj);
        output.close();
        return bos.toByteArray();
    }

    @Nonnull
    public static <T> T deserializeObjectByKryo(@Nonnull byte[] in, @Nonnull Class<T> clazz) {
        Kryo kryo = getKryo();
        Input inp = new Input(in);
        T t = kryo.readObject(inp, clazz);
        inp.close();
        return t;
    }

    @Nonnull
    private static Kryo getKryo() {
        return Utilities.runtimeSerializationKryo.get();
    }

}
