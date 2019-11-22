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
package hivemall.utils.lang;

import java.util.Map;
import java.util.NoSuchElementException;

import javax.annotation.Nonnull;

public final class OptionUtils {

    private OptionUtils() {}

    public static boolean getBoolean(@Nonnull final Map<String, ?> options,
            @Nonnull final String optionName) {
        Object value = options.get(optionName);
        if (value == null) {
            throw new NoSuchElementException("Could not find an option: " + optionName);
        }
        if (!(value instanceof Boolean)) {
            throw new IllegalStateException(
                "Non-boolean unexpected value type: " + value.getClass().getSimpleName());
        }
        return ((Boolean) value).booleanValue();
    }

    public static int getInt(@Nonnull final Map<String, ?> options,
            @Nonnull final String optionName) {
        Object value = options.get(optionName);
        if (value == null) {
            throw new NoSuchElementException("Could not find an option: " + optionName);
        }
        if (!(value instanceof Integer)) {
            throw new IllegalStateException(
                "Non-integer unexpected value type: " + value.getClass().getSimpleName());
        }
        return ((Integer) value).intValue();
    }

    public static long getLong(@Nonnull final Map<String, ?> options,
            @Nonnull final String optionName) {
        Object value = options.get(optionName);
        if (value == null) {
            throw new NoSuchElementException("Could not find an option: " + optionName);
        }
        if (!(value instanceof Long)) {
            throw new IllegalStateException(
                "Non-long unexpected value type: " + value.getClass().getSimpleName());
        }
        return ((Long) value).longValue();
    }

    public static double getDouble(@Nonnull final Map<String, ?> options,
            @Nonnull final String optionName) {
        Object value = options.get(optionName);
        if (value == null) {
            throw new NoSuchElementException("Could not find an option: " + optionName);
        }
        if (!(value instanceof Double)) {
            throw new IllegalStateException(
                "Non-double unexpected value type: " + value.getClass().getSimpleName());
        }
        return ((Double) value).doubleValue();
    }


}
