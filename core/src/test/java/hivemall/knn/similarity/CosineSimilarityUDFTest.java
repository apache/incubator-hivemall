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
package hivemall.knn.similarity;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import hivemall.TestUtils;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

public class CosineSimilarityUDFTest {

    @Test
    public void testEvaluate() throws IOException {
        {
            List<String> ftvec1 = Arrays.asList("bbb:1.4", "aaa:0.9", "ccc");
            Assert.assertEquals(1.f, CosineSimilarityUDF.cosineSimilarity(ftvec1, ftvec1), 0.0);
        }

        Assert.assertEquals(
            0.f,
            CosineSimilarityUDF.cosineSimilarity(Arrays.asList("a", "b", "c"),
                Arrays.asList("d", "e")), 0.0);
        Assert.assertEquals(
            0.f,
            CosineSimilarityUDF.cosineSimilarity(Arrays.asList("a", "b", "c"),
                Arrays.asList("d", "e")), 0.0);

        Assert.assertEquals(1.f,
            CosineSimilarityUDF.cosineSimilarity(Arrays.asList("a", "b"), Arrays.asList("a", "b")),
            0.0);

        Assert.assertEquals(0.5f,
            CosineSimilarityUDF.cosineSimilarity(Arrays.asList("a", "b"), Arrays.asList("a", "c")),
            0.0);

        Assert.assertEquals(-1.f,
            CosineSimilarityUDF.cosineSimilarity(Arrays.asList("a:1.0"), Arrays.asList("a:-1.0")),
            0.0);

        Assert.assertTrue(CosineSimilarityUDF.cosineSimilarity(Arrays.asList("apple", "orange"),
            Arrays.asList("banana", "apple")) > 0.f);
        Assert.assertTrue(CosineSimilarityUDF.cosineSimilarity(Arrays.asList("apple", "orange"),
            Arrays.asList("banana", "apple")) > 0.f);

        Assert.assertTrue((CosineSimilarityUDF.cosineSimilarity(Arrays.asList("apple", "orange"),
            Arrays.asList("banana", "orange", "apple"))) > (CosineSimilarityUDF.cosineSimilarity(
            Arrays.asList("apple", "orange"), Arrays.asList("banana", "orange"))));

        Assert.assertEquals(
            1.0f,
            CosineSimilarityUDF.cosineSimilarity(
                Arrays.asList("This is a sentence with seven tokens".split(" ")),
                Arrays.<String>asList("This is a sentence with seven tokens".split(" "))), 0.0);
        Assert.assertEquals(
            1.0f,
            CosineSimilarityUDF.cosineSimilarity(
                Arrays.asList("This is a sentence with seven tokens".split(" ")),
                Arrays.<String>asList("This is a sentence with seven tokens".split(" "))), 0.0);

        {
            List<String> tokens1 = Arrays.asList("1:1,2:1,3:1,4:1,5:0,6:1,7:1,8:1,9:0,10:1,11:1".split(","));
            List<String> tokens2 = Arrays.asList("1:1,2:1,3:0,4:1,5:1,6:1,7:1,8:0,9:1,10:1,11:1".split(","));
            Assert.assertEquals(0.77777f, CosineSimilarityUDF.cosineSimilarity(tokens1, tokens2),
                0.00001f);
        }

        {
            List<String> tokens1 = Arrays.asList("1 2 3 4   6 7 8   10 11".split("\\s+"));
            List<String> tokens2 = Arrays.asList("1 2   4 5 6 7   9 10 11".split("\\s+"));
            double dotp = 1 + 1 + 0 + 1 + 0 + 1 + 1 + 0 + 0 + 1 + 1;
            double norm = Math.sqrt(tokens1.size()) * Math.sqrt(tokens2.size());
            Assert.assertEquals(dotp / norm,
                CosineSimilarityUDF.cosineSimilarity(tokens1, tokens2), 0.00001f);
            Assert.assertEquals(dotp / norm,
                CosineSimilarityUDF.cosineSimilarity(tokens1, tokens2), 0.00001f);

            Assert.assertEquals(
                dotp / norm,
                CosineSimilarityUDF.cosineSimilarity(
                    Arrays.asList("1", "2", "3", "4", "6", "7", "8", "10", "11"),
                    Arrays.asList("1", "2", "4", "5", "6", "7", "9", "10", "11")), 0.00001f);
        }

        Assert.assertEquals(
            0.f,
            CosineSimilarityUDF.cosineSimilarity(Arrays.asList("1", "2", "3"),
                Arrays.asList("4", "5")), 0.0);
        Assert.assertEquals(1.f,
            CosineSimilarityUDF.cosineSimilarity(Arrays.asList("1", "2"), Arrays.asList("1", "2")),
            0.0);
    }

    @Test
    public void testSerialization() throws HiveException, IOException {
        TestUtils.testGenericUDFSerialization(
            CosineSimilarityUDF.class,
            new ObjectInspector[] {
                    ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector),
                    ObjectInspectorFactory.getStandardListObjectInspector(PrimitiveObjectInspectorFactory.javaStringObjectInspector)},
            new Object[] {Arrays.asList("1:1.0", "2:3.0", "3:3.0"), Arrays.asList("1:2.0", "3:6.0")});
    }
}
