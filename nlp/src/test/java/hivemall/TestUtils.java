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

import org.apache.hadoop.hive.ql.exec.Utilities;
import org.apache.hive.com.esotericsoftware.kryo.Kryo;
import org.apache.hive.com.esotericsoftware.kryo.io.Input;
import org.apache.hive.com.esotericsoftware.kryo.io.Output;

import javax.annotation.Nonnull;
import java.io.ByteArrayOutputStream;

public final class TestUtils {

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
