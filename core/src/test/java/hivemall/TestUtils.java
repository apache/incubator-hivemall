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
import org.objenesis.strategy.StdInstantiatorStrategy;

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
        byte[] serialized1 = serializeObjectByKryo(udf);
        deserializeObjectByKryo(serialized1, clazz);

        byte[] serialized2 = serializeObjectByOriginalKryo(udf);
        deserializeObjectByOriginalKryo(serialized2, clazz);

        int size = row.length;
        GenericUDF.DeferredObject[] rowDeferred = new GenericUDF.DeferredObject[size];
        for (int i = 0; i < size; i++) {
            rowDeferred[i] = new GenericUDF.DeferredJavaObject(row[i]);
        }

        udf.evaluate(rowDeferred);

        // serialization after evaluating row
        serialized1 = serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized1, clazz);

        serialized2 = serializeObjectByOriginalKryo(udf);
        TestUtils.deserializeObjectByOriginalKryo(serialized2, clazz);

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

        byte[] serialized2 = serializeObjectByOriginalKryo(udtf);
        deserializeObjectByOriginalKryo(serialized2, clazz);

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

    // -------------------------------
    // Hive version of Kryo

    @Nonnull
    public static byte[] serializeObjectByKryo(@Nonnull Object obj) {
        Kryo kryo = getHiveKryo();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        Output output = new Output(bos);
        kryo.writeObject(output, obj);
        output.close();
        return bos.toByteArray();
    }

    @Nonnull
    public static <T> T deserializeObjectByKryo(@Nonnull byte[] in, @Nonnull Class<T> clazz) {
        Kryo kryo = getHiveKryo();
        Input inp = new Input(in);
        T t = kryo.readObject(inp, clazz);
        inp.close();
        return t;
    }

    @Nonnull
    private static Kryo getHiveKryo() {
        return Utilities.runtimeSerializationKryo.get();
    }

    // -------------------------------
    // esotericsoftware's original version of Kryo

    @Nonnull
    public static byte[] serializeObjectByOriginalKryo(@Nonnull Object obj) {
        com.esotericsoftware.kryo.Kryo kryo = getOriginalKryo();
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        com.esotericsoftware.kryo.io.Output output = new com.esotericsoftware.kryo.io.Output(bos);
        kryo.writeObject(output, obj);
        output.close();
        return bos.toByteArray();
    }

    @Nonnull
    public static <T> T deserializeObjectByOriginalKryo(@Nonnull byte[] in,
            @Nonnull Class<T> clazz) {
        com.esotericsoftware.kryo.Kryo kryo = getOriginalKryo();
        com.esotericsoftware.kryo.io.Input inp = new com.esotericsoftware.kryo.io.Input(in);
        T t = kryo.readObject(inp, clazz);
        inp.close();
        return t;
    }

    @Nonnull
    private static com.esotericsoftware.kryo.Kryo getOriginalKryo() {
        com.esotericsoftware.kryo.Kryo kryo = new com.esotericsoftware.kryo.Kryo();

        // kryo.setReferences(true);
        // kryo.setRegistrationRequired(false);

        // see https://stackoverflow.com/a/23962797/5332768
        ((com.esotericsoftware.kryo.Kryo.DefaultInstantiatorStrategy) kryo.getInstantiatorStrategy()).setFallbackInstantiatorStrategy(
            new StdInstantiatorStrategy());

        return kryo;
    }

}
