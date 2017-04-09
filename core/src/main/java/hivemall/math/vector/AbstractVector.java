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
package hivemall.math.vector;

import javax.annotation.Nonnegative;

public abstract class AbstractVector implements Vector {

    public AbstractVector() {}

    @Override
    public double get(@Nonnegative final int index) {
        return get(index, 0.d);
    }

    protected static final void checkIndex(final int index) {
        if (index < 0) {
            throw new IndexOutOfBoundsException("Invalid index " + index);
        }
    }

    protected static final void checkIndex(final int index, final int size) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index " + index + " out of bounds " + size);
        }
    }

}
