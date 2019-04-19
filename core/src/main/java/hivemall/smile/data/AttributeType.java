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
package hivemall.smile.data;

import hivemall.annotations.BackwardCompatibility;

public enum AttributeType {
    NUMERIC((byte) 1), NOMINAL((byte) 2);

    private final byte id;

    private AttributeType(byte id) {
        this.id = id;
    }

    public byte getTypeId() {
        return id;
    }

    public static AttributeType resolve(final byte id) {
        final AttributeType type;
        switch (id) {
            case 1:
                type = NUMERIC;
                break;
            case 2:
                type = NOMINAL;
                break;
            default:
                throw new IllegalStateException("Unexpected type: " + id);
        }
        return type;
    }

    @BackwardCompatibility
    public static AttributeType resolve(final int id) {
        final AttributeType type;
        switch (id) {
            case 1:
                type = NUMERIC;
                break;
            case 2:
                type = NOMINAL;
                break;
            default:
                throw new IllegalStateException("Unexpected type: " + id);
        }
        return type;
    }

}
