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
package hivemall.utils.hadoop;

import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.common.type.HiveChar;
import org.apache.hadoop.hive.common.type.HiveDecimal;
import org.apache.hadoop.hive.common.type.HiveVarchar;
import org.apache.hadoop.hive.serde2.SerDeException;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.StructTypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfo;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoFactory;
import org.apache.hadoop.hive.serde2.typeinfo.TypeInfoUtils;
import org.apache.hadoop.io.Text;
import org.apache.hive.hcatalog.data.DataType;
import org.apache.hive.hcatalog.data.DefaultHCatRecord;
import org.apache.hive.hcatalog.data.HCatRecordObjectInspector;
import org.apache.hive.hcatalog.data.HCatRecordObjectInspectorFactory;
import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.ImmutableMap;

public class JsonSerdeUtilsTest {

    /**
     * This test tests that our json deserialization is not too strict, as per HIVE-6166
     *
     * i.e, if our schema is "s:struct<a:int,b:string>,k:int", and we pass in data that looks like :
     * 
     * <pre>
     *                        {
     *                            "x" : "abc" ,
     *                            "t" : {
     *                                "a" : "1",
     *                                "b" : "2",
     *                                "c" : [
     *                                    { "x" : 2 , "y" : 3 } ,
     *                                    { "x" : 3 , "y" : 2 }
     *                                ]
     *                            } ,
     *                            "s" : {
     *                                "a" : 2 ,
     *                                "b" : "blah",
     *                                "c": "woo"
     *                            }
     *                        }
     * </pre>
     *
     * Then it should still work, and ignore the "x" and "t" field and "c" subfield of "s", and it
     * should read k as null.
     */
    @Test
    public void testLooseJsonReadability() throws Exception {
        List<String> columnNames = Arrays.asList("s,k".split(","));
        List<TypeInfo> columnTypes =
                TypeInfoUtils.getTypeInfosFromTypeString("struct<a:int,b:string>,int");

        Text jsonText1 = new Text("{ \"x\" : \"abc\" , "
                + " \"t\" : { \"a\":\"1\", \"b\":\"2\", \"c\":[ { \"x\":2 , \"y\":3 } , { \"x\":3 , \"y\":2 }] } ,"
                + "\"s\" : { \"a\" : 2 , \"b\" : \"blah\", \"c\": \"woo\" } }");

        Text jsonText2 = new Text("{ \"x\" : \"abc\" , "
                + " \"t\" : { \"a\":\"1\", \"b\":\"2\", \"c\":[ { \"x\":2 , \"y\":3 } , { \"x\":3 , \"y\":2 }] } ,"
                + "\"s\" : { \"a\" : 2 , \"b\" : \"blah\", \"c\": \"woo\" } , " + "\"k\" : 113 "
                + "}");

        List<Object> expected1 = Arrays.<Object>asList(Arrays.asList(2, "blah"), null);
        List<Object> expected2 = Arrays.<Object>asList(Arrays.asList(2, "blah"), 113);
        List<Object> result1 = JsonSerdeUtils.deserialize(jsonText1, columnNames, columnTypes);
        List<Object> result2 = JsonSerdeUtils.deserialize(jsonText2, columnNames, columnTypes);

        Assert.assertEquals(expected1, result1);
        Assert.assertEquals(expected2, result2);
    }

    @Test
    public void testMapValues() throws SerDeException {
        List<String> columnNames = Arrays.asList("a,b".split(","));
        List<TypeInfo> columnTypes =
                TypeInfoUtils.getTypeInfosFromTypeString("array<string>,map<string,int>");

        Text text1 = new Text("{ \"a\":[\"aaa\"],\"b\":{\"bbb\":1}} ");
        Text text2 = new Text("{\"a\":[\"yyy\"],\"b\":{\"zzz\":123}}");
        Text text3 = new Text("{\"a\":[\"a\"],\"b\":{\"x\":11, \"y\": 22, \"z\": null}}");

        List<Object> expected1 = Arrays.<Object>asList(Arrays.<String>asList("aaa"),
            createHashMapStringInteger("bbb", 1));
        List<Object> expected2 = Arrays.<Object>asList(Arrays.<String>asList("yyy"),
            createHashMapStringInteger("zzz", 123));
        List<Object> expected3 = Arrays.<Object>asList(Arrays.<String>asList("a"),
            createHashMapStringInteger("x", 11, "y", 22, "z", null));

        List<Object> result1 = JsonSerdeUtils.deserialize(text1, columnNames, columnTypes);
        List<Object> result2 = JsonSerdeUtils.deserialize(text2, columnNames, columnTypes);
        List<Object> result3 = JsonSerdeUtils.deserialize(text3, columnNames, columnTypes);

        Assert.assertEquals(expected1, result1);
        Assert.assertEquals(expected2, result2);
        Assert.assertEquals(expected3, result3);
    }

    private static HashMap<String, Integer> createHashMapStringInteger(Object... vals) {
        Assert.assertTrue(vals.length % 2 == 0);
        HashMap<String, Integer> retval = new HashMap<String, Integer>();
        for (int idx = 0; idx < vals.length; idx += 2) {
            retval.put((String) vals[idx], (Integer) vals[idx + 1]);
        }
        return retval;
    }

    @Test
    public void testRW() throws Exception {
        List<Object> rlist = new ArrayList<Object>(13);
        {
            rlist.add(new Byte("123"));
            rlist.add(new Short("456"));
            rlist.add(new Integer(789));
            rlist.add(new Long(1000L));
            rlist.add(new Double(5.3D));
            rlist.add(new Float(2.39F));
            rlist.add(new String("hcat\nand\nhadoop"));
            rlist.add(null);

            List<Object> innerStruct = new ArrayList<Object>(2);
            innerStruct.add(new String("abc"));
            innerStruct.add(new String("def"));
            rlist.add(innerStruct);

            List<Integer> innerList = new ArrayList<Integer>();
            innerList.add(314);
            innerList.add(007);
            rlist.add(innerList);

            Map<Short, String> map = new HashMap<Short, String>(3);
            map.put(new Short("2"), "hcat is cool");
            map.put(new Short("3"), "is it?");
            map.put(new Short("4"), "or is it not?");
            rlist.add(map);

            rlist.add(new Boolean(true));

            List<Object> c1 = new ArrayList<Object>();
            List<Object> c1_1 = new ArrayList<Object>();
            c1_1.add(new Integer(12));
            List<Object> i2 = new ArrayList<Object>();
            List<Integer> ii1 = new ArrayList<Integer>();
            ii1.add(new Integer(13));
            ii1.add(new Integer(14));
            i2.add(ii1);
            Map<String, List<?>> ii2 = new HashMap<String, List<?>>();
            List<Integer> iii1 = new ArrayList<Integer>();
            iii1.add(new Integer(15));
            ii2.put("phew", iii1);
            i2.add(ii2);
            c1_1.add(i2);
            c1.add(c1_1);
            rlist.add(c1);
            rlist.add(HiveDecimal.create(new BigDecimal("123.45")));//prec 5, scale 2
            rlist.add(new HiveChar("hive\nchar", 10));
            rlist.add(new HiveVarchar("hive\nvarchar", 20));
            rlist.add(Date.valueOf("2014-01-07"));
            rlist.add(new Timestamp(System.currentTimeMillis()));
            rlist.add("hive\nbinary".getBytes("UTF-8"));
        }

        DefaultHCatRecord r = new DefaultHCatRecord(rlist);

        List<String> columnNames =
                Arrays.asList("ti,si,i,bi,d,f,s,n,r,l,m,b,c1,bd,hc,hvc,dt,ts,bin".split(","));
        List<TypeInfo> columnTypes = TypeInfoUtils.getTypeInfosFromTypeString(
            "tinyint,smallint,int,bigint,double,float,string,string,"
                    + "struct<a:string,b:string>,array<int>,map<smallint,string>,boolean,"
                    + "array<struct<i1:int,i2:struct<ii1:array<int>,ii2:map<string,struct<iii1:int>>>>>,"
                    + "decimal(5,2),char(10),varchar(20),date,timestamp,binary");

        StructTypeInfo rowTypeInfo =
                (StructTypeInfo) TypeInfoFactory.getStructTypeInfo(columnNames, columnTypes);
        HCatRecordObjectInspector objInspector =
                HCatRecordObjectInspectorFactory.getHCatRecordObjectInspector(rowTypeInfo);

        Text serialized = JsonSerdeUtils.serialize(r, objInspector, columnNames);
        List<Object> deserialized =
                JsonSerdeUtils.deserialize(serialized, columnNames, columnTypes);

        assertRecordEquals(rlist, deserialized);
    }

    @Test
    public void testRWNull() throws Exception {
        List<Object> nlist = new ArrayList<Object>(13);
        {
            nlist.add(null); // tinyint
            nlist.add(null); // smallint
            nlist.add(null); // int
            nlist.add(null); // bigint
            nlist.add(null); // double
            nlist.add(null); // float
            nlist.add(null); // string
            nlist.add(null); // string
            nlist.add(null); // struct
            nlist.add(null); // array
            nlist.add(null); // map
            nlist.add(null); // bool
            nlist.add(null); // complex
            nlist.add(null); //decimal(5,2)
            nlist.add(null); //char(10)
            nlist.add(null); //varchar(20)
            nlist.add(null); //date
            nlist.add(null); //timestamp
            nlist.add(null); //binary
        }

        DefaultHCatRecord r = new DefaultHCatRecord(nlist);

        List<String> columnNames =
                Arrays.asList("ti,si,i,bi,d,f,s,n,r,l,m,b,c1,bd,hc,hvc,dt,ts,bin".split(","));
        List<TypeInfo> columnTypes = TypeInfoUtils.getTypeInfosFromTypeString(
            "tinyint,smallint,int,bigint,double,float,string,string,"
                    + "struct<a:string,b:string>,array<int>,map<smallint,string>,boolean,"
                    + "array<struct<i1:int,i2:struct<ii1:array<int>,ii2:map<string,struct<iii1:int>>>>>,"
                    + "decimal(5,2),char(10),varchar(20),date,timestamp,binary");

        StructTypeInfo rowTypeInfo =
                (StructTypeInfo) TypeInfoFactory.getStructTypeInfo(columnNames, columnTypes);
        HCatRecordObjectInspector objInspector =
                HCatRecordObjectInspectorFactory.getHCatRecordObjectInspector(rowTypeInfo);

        Text serialized = JsonSerdeUtils.serialize(r, objInspector, columnNames);
        List<Object> deserialized =
                JsonSerdeUtils.deserialize(serialized, columnNames, columnTypes);

        assertRecordEquals(nlist, deserialized);
    }

    @Test
    public void testStructWithoutColumnNames() throws Exception {
        Text json1 = new Text("{ \"person\" : { \"name\" : \"makoto\" , \"age\" : 37 } }");
        TypeInfo type1 = TypeInfoUtils.getTypeInfoFromTypeString("struct<name:string,age:int>");
        List<Object> expected1 = Arrays.<Object>asList("makoto", 37);

        List<Object> deserialized1 =
                JsonSerdeUtils.deserialize(json1, Arrays.asList("person"), Arrays.asList(type1));

        assertRecordEquals(expected1, deserialized1);
    }

    @Test
    public void testTopLevelArray() throws Exception {
        List<String> expected1 = Arrays.asList("Taro", "Tanaka");
        Text json1 = new Text("[\"Taro\",\"Tanaka\"]");
        TypeInfo type1 = TypeInfoUtils.getTypeInfoFromTypeString("array<string>");

        List<Object> deserialized1 = JsonSerdeUtils.deserialize(json1, type1);
        assertRecordEquals(expected1, deserialized1);
        Text serialized1 = JsonSerdeUtils.serialize(deserialized1,
            HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(type1));
        Assert.assertEquals(json1, serialized1);

        List<Double> expected2 = Arrays.asList(1.1d, 2.2d, 3.3d);
        Text json2 = new Text("[1.1,2.2,3.3]");
        TypeInfo type2 = TypeInfoUtils.getTypeInfoFromTypeString("array<double>");

        List<Object> deserialized2 = JsonSerdeUtils.deserialize(json2, type2);
        assertRecordEquals(expected2, deserialized2);
        Text serialized2 = JsonSerdeUtils.serialize(deserialized2,
            HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(type2));
        Assert.assertEquals(json2, serialized2);
    }

    @Test
    public void testTopLevelNestedArray() throws Exception {
        List<Map<String, Integer>> expected1 = Arrays.<Map<String, Integer>>asList(
            ImmutableMap.of("one", 1), ImmutableMap.of("two", 2));
        Text json1 = new Text("[{\"one\":1},{\"two\":2}]");
        TypeInfo type1 = TypeInfoUtils.getTypeInfoFromTypeString("array<map<string,int>>");

        List<Object> deserialized1 = JsonSerdeUtils.deserialize(json1, type1);
        assertRecordEquals(expected1, deserialized1);
        Text serialized1 = JsonSerdeUtils.serialize(deserialized1,
            HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(type1));
        Assert.assertEquals(json1, serialized1);

        List<Map<String, Integer>> expected2 = Arrays.<Map<String, Integer>>asList(
            ImmutableMap.of("one", 1, "two", 2), ImmutableMap.of("three", 3));
        Text json2 = new Text("[{\"one\":1,\"two\":2},{\"three\":3}]");
        TypeInfo type2 = TypeInfoUtils.getTypeInfoFromTypeString("array<map<string,int>>");

        List<Object> deserialized2 = JsonSerdeUtils.deserialize(json2, type2);
        assertRecordEquals(expected2, deserialized2);
        Text serialized2 = JsonSerdeUtils.serialize(deserialized2,
            HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(type2));
        Assert.assertEquals(json2, serialized2);
    }

    @Test
    public void testTopLevelMap() throws Exception {
        Map<String, Integer> expected1 = ImmutableMap.of("one", 1, "two", 2);
        Text json1 = new Text("{\"one\":1,\"two\":2}");
        TypeInfo type1 = TypeInfoUtils.getTypeInfoFromTypeString("map<string,int>");

        Map<String, Integer> deserialized1 = JsonSerdeUtils.deserialize(json1, type1);
        Assert.assertEquals(expected1, deserialized1);
        Text serialized1 = JsonSerdeUtils.serialize(deserialized1,
            HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(type1));
        Assert.assertEquals(json1, serialized1);
    }

    @Test
    public void testTopLevelPrimitive() throws Exception {
        Double expected1 = Double.valueOf(3.3);
        Text json1 = new Text("3.3");
        TypeInfo type1 = TypeInfoUtils.getTypeInfoFromTypeString("double");

        Object deserialized1 = JsonSerdeUtils.deserialize(json1, type1);
        Assert.assertEquals(expected1, deserialized1);
        Text serialized1 = JsonSerdeUtils.serialize(deserialized1,
            HCatRecordObjectInspectorFactory.getStandardObjectInspectorFromTypeInfo(type1));
        Assert.assertEquals(json1, serialized1);

        Boolean expected2 = Boolean.FALSE;
        Text json2 = new Text("false");

        Boolean deserialized2 = JsonSerdeUtils.deserialize(json2);
        Assert.assertEquals(expected2, deserialized2);
        Text serialized2 = JsonSerdeUtils.serialize(deserialized2,
            PrimitiveObjectInspectorFactory.javaBooleanObjectInspector);
        Assert.assertEquals(json2, serialized2);
    }

    private static void assertRecordEquals(@Nonnull final List<?> first,
            @Nonnull final List<?> second) {
        int mySz = first.size();
        int urSz = second.size();
        if (mySz != urSz) {
            throw new RuntimeException(
                "#expected != #actual. #expected=" + mySz + ", #actual=" + urSz);
        } else {
            for (int i = 0; i < first.size(); i++) {
                int c = DataType.compare(first.get(i), second.get(i));
                if (c != 0) {
                    String msg = "first.get(" + i + "}='" + first.get(i) + "' second.get(" + i
                            + ")='" + second.get(i) + "' compared as " + c + "\n" + "Types 1st/2nd="
                            + DataType.findType(first.get(i)) + "/"
                            + DataType.findType(second.get(i)) + '\n' + "first='" + first.get(i)
                            + "' second='" + second.get(i) + "'";
                    if (first.get(i) instanceof Date) {
                        msg += "\n((Date)first.get(i)).getTime()="
                                + ((Date) first.get(i)).getTime();
                    }
                    if (second.get(i) instanceof Date) {
                        msg += "\n((Date)second.get(i)).getTime()="
                                + ((Date) second.get(i)).getTime();
                    }
                    throw new RuntimeException(msg);
                }
            }
        }
    }

}
