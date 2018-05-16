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
package hivemall.tools.array;

import hivemall.utils.hadoop.HiveUtils;

import java.util.Arrays;
import java.util.List;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDTF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.PrimitiveObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.StructObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.BooleanObjectInspector;

/**
 * ConditionalEmit takes an array of booleans and strings, and emits records if the boolean is true.
 *
 * <p/>
 * This allows you to emit multiple rows on one pass of the data, rather than doing a union of
 * multiple views with different where clauses.
 * <p/>
 *
 * <pre>
 * select
 *    conditional_emit( 
 *       array( maxwell_score > 80, abs( maxwell_score - other.maxwell_score ) < 5, city = "New York" ),
 *       array( "CELEB", "PEER", "NEW_YORKER" )
 *    )
 * from
 *    table_to_scan_once
 * </pre>
 */
// @formatter:off
@Description(name = "conditional_emit",
        value = "_FUNC_(array<boolean> conditions, array<primitive> features)"
                + " - Emit features of a row according to various conditions",
        extended = "WITH input as (\n" + 
                "   select array(true, false, true) as conditions, array(\"one\", \"two\", \"three\") as features\n" + 
                "   UNION ALL\n" + 
                "   select array(true, true, false), array(\"four\", \"five\", \"six\")\n" + 
                ")\n" + 
                "SELECT\n" + 
                "  conditional_emit(\n" + 
                "     conditions, features\n" + 
                "  )\n" + 
                "FROM \n" + 
                "  input;\n" +
                " one\n" + 
                " three\n" + 
                " four\n" + 
                " five")
// @formatter:on
@UDFType(deterministic = true, stateful = false)
public final class ConditionalEmitUDTF extends GenericUDTF {

    private ListObjectInspector conditionsOI;
    private BooleanObjectInspector condElemOI;
    private ListObjectInspector featuresOI;
    private PrimitiveObjectInspector featureElemOI;

    private final Object[] forwardObj = new Object[1];

    @Override
    public StructObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 2) {
            throw new UDFArgumentException(
                "conditional_emit takes 2 arguments: array<boolean>, array<primitive>");
        }

        this.conditionsOI = HiveUtils.asListOI(argOIs[0]);
        this.condElemOI = HiveUtils.asBooleanOI(conditionsOI.getListElementObjectInspector());

        this.featuresOI = HiveUtils.asListOI(argOIs[1]);
        this.featureElemOI =
                HiveUtils.asPrimitiveObjectInspector(featuresOI.getListElementObjectInspector());

        List<String> fieldNames = Arrays.asList("feature");
        List<ObjectInspector> fieldOIs = Arrays.<ObjectInspector>asList(featureElemOI);

        return ObjectInspectorFactory.getStandardStructObjectInspector(fieldNames, fieldOIs);
    }

    @Override
    public void process(@Nonnull Object[] args) throws HiveException {
        Object arg0 = args[0], arg1 = args[1];
        if (arg0 == null || arg1 == null) {
            return;
        }

        final int conditionSize = conditionsOI.getListLength(arg0);
        final int featureSize = featuresOI.getListLength(arg1);
        if (conditionSize != featureSize) {
            throw new HiveException(
                "Arrays must be of same length in condition_emit(array<boolean> conditions, array<string> features).\n"
                        + "#conditions=" + conditionSize + ", #features=" + featureSize);
        }

        for (int i = 0; i < conditionSize; i++) {
            Object condObj = conditionsOI.getListElement(arg0, i);
            if (condObj == null) {
                continue;
            }
            if (condElemOI.get(condObj) == false) {
                continue;
            }
            Object featureObj = featuresOI.getListElement(arg1, i);
            if (featureObj == null) {
                continue;
            }

            forwardObj[0] = featureObj;
            forward(forwardObj);
        }

    }

    @Override
    public void close() throws HiveException {}

}
