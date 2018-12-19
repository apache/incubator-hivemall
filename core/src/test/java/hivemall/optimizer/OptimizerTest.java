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
package hivemall.optimizer;

import hivemall.optimizer.Optimizer.OptimizerBase;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.junit.Assert;
import org.junit.Test;

public final class OptimizerTest {

    @Test
    public void testIllegalOptimizer() {
        try {
            final Map<String, String> emptyOptions = new HashMap<String, String>();
            DenseOptimizerFactory.create(1024, emptyOptions);
            Assert.fail();
        } catch (IllegalArgumentException e) {
            // tests passed
        }
        try {
            final Map<String, String> options = new HashMap<String, String>();
            options.put("optimizer", "illegal");
            DenseOptimizerFactory.create(1024, options);
            Assert.fail();
        } catch (IllegalArgumentException e) {
            // tests passed
        }
        try {
            final Map<String, String> emptyOptions = new HashMap<String, String>();
            SparseOptimizerFactory.create(1024, emptyOptions);
            Assert.fail();
        } catch (IllegalArgumentException e) {
            // tests passed
        }
        try {
            final Map<String, String> options = new HashMap<String, String>();
            options.put("optimizer", "illegal");
            SparseOptimizerFactory.create(1024, options);
            Assert.fail();
        } catch (IllegalArgumentException e) {
            // tests passed
        }
    }

    @Test
    public void testOptimizerFactory() {
        final Map<String, String> options = new HashMap<String, String>();
        final String[] regTypes = new String[] {"NO", "L1", "L2", "ElasticNet"};
        options.put("optimizer", "SGD");
        for (final String regType : regTypes) {
            options.put("regularization", regType);
            Assert.assertTrue(DenseOptimizerFactory.create(8, options) instanceof Optimizer.SGD);
            Assert.assertTrue(SparseOptimizerFactory.create(8, options) instanceof Optimizer.SGD);
        }
        options.put("optimizer", "AdaDelta");
        for (final String regType : regTypes) {
            options.put("regularization", regType);
            Assert.assertTrue(
                DenseOptimizerFactory.create(8, options) instanceof DenseOptimizerFactory.AdaDelta);
            Assert.assertTrue(SparseOptimizerFactory.create(8,
                options) instanceof SparseOptimizerFactory.AdaDelta);
        }
        options.put("optimizer", "AdaGrad");
        for (final String regType : regTypes) {
            options.put("regularization", regType);
            Assert.assertTrue(
                DenseOptimizerFactory.create(8, options) instanceof DenseOptimizerFactory.AdaGrad);
            Assert.assertTrue(SparseOptimizerFactory.create(8,
                options) instanceof SparseOptimizerFactory.AdaGrad);
        }
        options.put("optimizer", "Adam");
        for (final String regType : regTypes) {
            options.put("regularization", regType);
            Assert.assertTrue(
                DenseOptimizerFactory.create(8, options) instanceof DenseOptimizerFactory.Adam);
            Assert.assertTrue(
                SparseOptimizerFactory.create(8, options) instanceof SparseOptimizerFactory.Adam);
        }

        // We need special handling for `Optimizer#RDA`
        options.put("optimizer", "AdaGrad");
        options.put("regularization", "RDA");
        Assert.assertTrue(
            DenseOptimizerFactory.create(8, options) instanceof DenseOptimizerFactory.AdagradRDA);
        Assert.assertTrue(
            SparseOptimizerFactory.create(8, options) instanceof SparseOptimizerFactory.AdagradRDA);

        // `SGD`, `AdaDelta`, and `Adam` currently does not support `RDA`
        for (final String optimizerType : new String[] {"SGD", "AdaDelta", "Adam"}) {
            options.put("optimizer", optimizerType);
            try {
                DenseOptimizerFactory.create(8, options);
                Assert.fail();
            } catch (IllegalArgumentException e) {
                // tests passed
            }
            try {
                SparseOptimizerFactory.create(8, options);
                Assert.fail();
            } catch (IllegalArgumentException e) {
                // tests passed
            }
        }
    }

    private void testUpdateWeights(OptimizerBase optimizer, int numUpdates, int initSize) {
        final float[] weights = new float[initSize * 2];
        final Random rnd = new Random();
        try {
            for (int i = 0; i < numUpdates; i++) {
                int index = rnd.nextInt(initSize);
                weights[index] = optimizer.update(index, weights[index], 0.1f);
            }
            for (int i = 0; i < numUpdates; i++) {
                int index = rnd.nextInt(initSize * 2);
                weights[index] = optimizer.update(index, weights[index], 0.1f);
            }
        } catch (Exception e) {
            Assert.fail("failed to update weights: " + e.getMessage());
        }
    }

    private void testOptimizer(final Map<String, String> options, int numUpdates, int initSize) {
        final Map<String, String> testOptions = new HashMap<String, String>(options);
        final String[] regTypes = new String[] {"NO", "L1", "L2", "RDA", "ElasticNet"};
        for (final String regType : regTypes) {
            options.put("regularization", regType);
            testUpdateWeights((OptimizerBase) DenseOptimizerFactory.create(1024, testOptions),
                65536, 1024);
            testUpdateWeights((OptimizerBase) SparseOptimizerFactory.create(1024, testOptions),
                65536, 1024);
        }
    }

    @Test
    public void testSGDOptimizer() {
        final Map<String, String> options = new HashMap<String, String>();
        options.put("optimizer", "SGD");
        testOptimizer(options, 65536, 1024);
    }

    @Test
    public void testAdaDeltaOptimizer() {
        final Map<String, String> options = new HashMap<String, String>();
        options.put("optimizer", "AdaDelta");
        testOptimizer(options, 65536, 1024);
    }

    @Test
    public void testAdaGradOptimizer() {
        final Map<String, String> options = new HashMap<String, String>();
        options.put("optimizer", "AdaGrad");
        testOptimizer(options, 65536, 1024);
    }

    @Test
    public void testAdamOptimizer() {
        final Map<String, String> options = new HashMap<String, String>();
        options.put("optimizer", "Adam");
        testOptimizer(options, 65536, 1024);
    }

}
