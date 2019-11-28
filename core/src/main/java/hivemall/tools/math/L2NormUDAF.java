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
package hivemall.tools.math;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDAF;
import org.apache.hadoop.hive.ql.exec.UDAFEvaluator;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;

// @formatter:off
@SuppressWarnings("deprecation")
@Description(name = "l2_norm",
        value = "_FUNC_(double x) - Return a L2 norm of the given input x.",
        extended = "WITH input as (\n" + 
                "  select generate_series(1,3) as v\n" + 
                ")\n" + 
                "select l2_norm(v) as l2norm\n" + 
                "from input;\n" + 
                "3.7416573867739413 = sqrt(1^2+2^2+3^2))")
// @formatter:on
public final class L2NormUDAF extends UDAF {

    public static class Evaluator implements UDAFEvaluator {

        private PartialResult partial;

        public Evaluator() {}

        @Override
        public void init() {
            this.partial = null;
        }

        public boolean iterate(DoubleWritable xi) throws HiveException {
            if (xi == null) {// skip
                return true;
            }
            if (partial == null) {
                this.partial = new PartialResult();
            }
            partial.iterate(xi.get());
            return true;
        }

        public PartialResult terminatePartial() {
            return partial;
        }

        public boolean merge(PartialResult other) throws HiveException {
            if (other == null) {
                return true;
            }
            if (partial == null) {
                this.partial = new PartialResult();
            }
            partial.merge(other);
            return true;
        }

        public double terminate() {
            if (partial == null) {
                return 0.d;
            }
            return partial.get();
        }
    }

    public static class PartialResult {

        double squaredSum;

        PartialResult() {
            this.squaredSum = 0.d;
        }

        void iterate(double xi) {
            squaredSum += xi * xi;
        }

        void merge(PartialResult other) {
            squaredSum += other.squaredSum;
        }

        double get() {
            return Math.sqrt(squaredSum);
        }

    }

}
