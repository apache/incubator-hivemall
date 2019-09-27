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
package hivemall.smile.classification;

import javax.annotation.Nonnull;

public abstract class PredictionHandler {

    public enum Operator {
        /* = */ EQ, /* != */ NE, /* <= */ LE, /* > */ GT;

        @Override
        public String toString() {
            switch (this) {
                case EQ:
                    return "=";
                case NE:
                    return "!=";
                case LE:
                    return "<=";
                case GT:
                    return ">";
                default:
                    throw new IllegalStateException("Unexpected operator: " + this);
            }
        }
    }

    public void init() {};

    public void visitBranch(@Nonnull Operator op, int splitFeatureIndex, double splitFeature,
            double splitValue) {}

    public void visitLeaf(double output) {}

    public void visitLeaf(int output, @Nonnull double[] posteriori) {}

    public <T> T getResult() {
        throw new UnsupportedOperationException();
    }

}
