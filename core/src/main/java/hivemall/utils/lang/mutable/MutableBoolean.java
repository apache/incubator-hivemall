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

import java.io.Serializable;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public final class MutableBoolean implements Comparable<MutableBoolean>, Serializable {
    private static final long serialVersionUID = -8946436031470563775L;

    private boolean value;

    public MutableBoolean() {
        this(false);
    }

    public MutableBoolean(boolean value) {
        this.value = value;
    }

    public boolean get() {
        return value;
    }

    public boolean booleanValue() {
        return value;
    }

    public void setValue(boolean value) {
        this.value = value;
    }

    public void setValue(@Nonnull Boolean value) {
        this.value = value.booleanValue();
    }

    public void setFalse() {
        this.value = false;
    }

    public void setTrue() {
        this.value = true;
    }

    @Override
    public int hashCode() {
        return value ? Boolean.TRUE.hashCode() : Boolean.FALSE.hashCode();
    }

    @Override
    public boolean equals(@Nullable Object other) {
        if (this == other) {
            return true;
        }
        if (other == null) {
            return false;
        }
        if (other instanceof MutableBoolean) {
            return value == ((MutableBoolean) other).booleanValue();
        }
        return false;
    }

    @Override
    public int compareTo(@Nonnull MutableBoolean o) {
        return Boolean.compare(value, o.value);
    }

}
