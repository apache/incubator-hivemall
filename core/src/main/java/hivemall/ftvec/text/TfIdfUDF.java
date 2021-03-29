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
package hivemall.ftvec.text;

import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.StringUtils;

import javax.annotation.Nonnegative;
import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.exec.UDFArgumentLengthException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.io.DoubleWritable;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorUtils;

@Description(name = "tfidf",
        value = "_FUNC_(double termFrequency, long numDocs, const long totalNumDocs) "
                + "- Return a smoothed TFIDF score in double.")
@UDFType(deterministic = true, stateful = false)
public final class TfIdfUDF extends GenericUDF {

    private PrimitiveObjectInspector tfOI;
    private PrimitiveObjectInspector numDocsOI;
    private PrimitiveObjectInspector totalNumDocsOI;

    @Nonnull
    private final DoubleWritable result = new DoubleWritable();

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 3) {
            throw new UDFArgumentLengthException(
                "tfidf takes exactly three arguments but got " + argOIs.length);
        }

        this.tfOI = HiveUtils.asDoubleCompatibleOI(argOIs, 0);
        this.numDocsOI = HiveUtils.asIntegerOI(argOIs, 1);
        this.totalNumDocsOI = HiveUtils.asIntegerOI(argOIs, 2);

        return PrimitiveObjectInspectorFactory.writableDoubleObjectInspector;
    }


    @Override
    public Object evaluate(DeferredObject[] arguments) throws HiveException {
        Object arg0 = getObject(arguments, 0);
        Object arg1 = getObject(arguments, 1);
        Object arg2 = getObject(arguments, 2);

        double tf = PrimitiveObjectInspectorUtils.getDouble(arg0, tfOI);
        // Note: not long but double to avoid long by long division
        double numDocs = PrimitiveObjectInspectorUtils.getLong(arg1, numDocsOI);
        double totalNumDocs = PrimitiveObjectInspectorUtils.getLong(arg2, totalNumDocsOI);

        // basic IDF
        //    idf = log(N/n_t)
        // IDF with smoothing
        //    idf = log(N/(1+n_t))+1
        //    idf = log(N/max(1,n_t))+1 -- avoid zero division by max(1,n_t) and +1 for smoothing
        double idf = Math.log10(totalNumDocs / Math.max(1.d, numDocs)) + 1.0d;
        double tfidf = tf * idf;
        result.set(tfidf);
        return result;
    }

    @Nonnull
    private static Object getObject(@Nonnull final DeferredObject[] arguments,
            @Nonnegative final int index) throws HiveException {
        Object obj = arguments[index].get();
        if (obj == null) {
            throw new UDFArgumentException(String.format("%d-th argument MUST not be null", index));
        }
        return obj;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "tfidf(" + StringUtils.join(children, ',') + ")";
    }

}
