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
package hivemall.utils.lang.mutable;

import javax.annotation.Nullable;

public final class MutableObject<T> {

    @Nullable
    private T _value;

    public MutableObject() {}

    public MutableObject(@Nullable T obj) {
        this._value = obj;
    }

    public boolean isSet() {
        return _value != null;
    }

    @Nullable
    public T get() {
        return _value;
    }

    public void set(@Nullable T obj) {
        this._value = obj;
    }

    public void setIfAbsent(@Nullable T obj) {
        if (_value == null) {
            this._value = obj;
        }
    }

    @Override
    public boolean equals(@Nullable Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        MutableObject<?> other = (MutableObject<?>) obj;
        if (_value == null) {
            if (other._value != null) {
                return false;
            }
        }
        return _value.equals(other._value);
    }

    @Override
    public int hashCode() {
        return _value == null ? 0 : _value.hashCode();
    }

    @Override
    public String toString() {
        return _value == null ? "null" : _value.toString();
    }

}
