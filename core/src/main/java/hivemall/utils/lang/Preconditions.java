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

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public final class Preconditions {

    private Preconditions() {}

    public static <T> T checkNotNull(T reference) {
        if (reference == null) {
            throw new NullPointerException();
        }
        return reference;
    }

    public static <T, E extends Throwable> T checkNotNull(@Nullable T reference,
            @Nonnull Class<E> clazz) throws E {
        if (reference == null) {
            final E throwable;
            try {
                throwable = clazz.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                throw new IllegalStateException("Failed to instantiate a class: " + clazz.getName(),
                    e);
            }
            throw throwable;
        }
        return reference;
    }

    public static <T, E extends Throwable> T checkNotNull(@Nullable T reference,
            @Nonnull String errorMessage, @Nonnull Class<E> clazz) throws E {
        if (reference == null) {
            final E throwable;
            try {
                Constructor<E> constructor = clazz.getConstructor(String.class);
                throwable = constructor.newInstance(errorMessage);
            } catch (NoSuchMethodException | SecurityException e1) {
                throw new IllegalStateException(
                    "Failed to get a Constructor(String): " + clazz.getName(), e1);
            } catch (InstantiationException | IllegalAccessException | IllegalArgumentException
                    | InvocationTargetException e2) {
                throw new IllegalStateException("Failed to instantiate a class: " + clazz.getName(),
                    e2);
            }
            throw throwable;
        }
        return reference;
    }

    public static <T> T checkNotNull(T reference, @Nullable Object errorMessage) {
        if (reference == null) {
            throw new NullPointerException(String.valueOf(errorMessage));
        }
        return reference;
    }

    public static void checkArgument(boolean expression) {
        if (!expression) {
            throw new IllegalArgumentException();
        }
    }

    public static <E extends Throwable> void checkArgument(boolean expression,
            @Nonnull Class<E> clazz) throws E {
        if (!expression) {
            final E throwable;
            try {
                throwable = clazz.newInstance();
            } catch (InstantiationException | IllegalAccessException e) {
                throw new IllegalStateException("Failed to instantiate a class: " + clazz.getName(),
                    e);
            }
            throw throwable;
        }
    }

    public static <E extends Throwable> void checkArgument(boolean expression,
            @Nonnull Class<E> clazz, @Nullable Object errorMessage) throws E {
        if (!expression) {
            Constructor<E> constructor;
            try {
                constructor = clazz.getConstructor(String.class);
            } catch (NoSuchMethodException | SecurityException e) {
                throw new IllegalStateException("Failed to get a constructor of " + clazz.getName(),
                    e);
            }
            final E throwable;
            try {
                throwable = constructor.newInstance(errorMessage);
            } catch (ReflectiveOperationException | IllegalArgumentException e) {
                throw new IllegalStateException("Failed to instantiate a class: " + clazz.getName(),
                    e);
            }
            throw throwable;
        }
    }

    public static void checkArgument(boolean expression, @Nullable Object errorMessage) {
        if (!expression) {
            throw new IllegalArgumentException(String.valueOf(errorMessage));
        }
    }

    public static void checkArgument(boolean expression, @Nullable String errorMessageTemplate,
            @Nullable Object... errorMessageArgs) {
        if (!expression) {
            throw new IllegalArgumentException(
                StringUtils.format(errorMessageTemplate, errorMessageArgs));
        }
    }

    @Nonnull
    public static int checkElementIndex(final int index, final int size) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("index (" + index + ") must not be negative");
        } else if (size < 0) {
            throw new IndexOutOfBoundsException("negative size: " + size);
        } else if (index >= size) {
            throw new IndexOutOfBoundsException(
                "index (" + index + ") must be less than size (" + size + ")");
        }
        return index;
    }

}
