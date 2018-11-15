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

package org.apache.spark.sql.hive

import org.apache.spark.sql.{AnalysisException, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.hive.HivemallGroupedDataset._
import org.apache.spark.sql.hive.HivemallOps._
import org.apache.spark.sql.hive.HivemallUtils._
import org.apache.spark.sql.hive.test.HivemallFeatureQueryTest
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.test.VectorQueryTest
import org.apache.spark.sql.types._
import org.apache.spark.test.TestFPWrapper._
import org.apache.spark.test.TestUtils


class HivemallOpsWithFeatureSuite extends HivemallFeatureQueryTest {

  test("anomaly") {
    import hiveContext.implicits._
    val df = spark.range(1000).selectExpr("id AS time", "rand() AS x")
    // TODO: Test results more strictly
    assert(df.sort($"time".asc).select(changefinder($"x")).count === 1000)
    assert(df.sort($"time".asc).select(sst($"x", lit("-th 0.005"))).count === 1000)
  }

  test("knn.similarity") {
    import hiveContext.implicits._

    val df1 = DummyInputData.select(
      cosine_similarity(typedLit(Seq(1, 2, 3, 4)), typedLit(Seq(3, 4, 5, 6))))
    val rows1 = df1.collect
    assert(rows1.length == 1)
    assert(rows1(0).getFloat(0) ~== 0.500f)

    val df2 = DummyInputData.select(jaccard_similarity(lit(5), lit(6)))
    val rows2 = df2.collect
    assert(rows2.length == 1)
    assert(rows2(0).getFloat(0) ~== 0.96875f)

    val df3 = DummyInputData.select(
      angular_similarity(typedLit(Seq(1, 2, 3)), typedLit(Seq(4, 5, 6))))
    val rows3 = df3.collect
    assert(rows3.length == 1)
    assert(rows3(0).getFloat(0) ~== 0.500f)

    val df4 = DummyInputData.select(
      euclid_similarity(typedLit(Seq(5, 3, 1)), typedLit(Seq(2, 8, 3))))
    val rows4 = df4.collect
    assert(rows4.length == 1)
    assert(rows4(0).getFloat(0) ~== 0.33333334f)

    val df5 = DummyInputData.select(distance2similarity(lit(1.0)))
    val rows5 = df5.collect
    assert(rows5.length == 1)
    assert(rows5(0).getFloat(0) ~== 0.5f)

    val df6 = Seq((Seq("1:0.3", "4:0.1"), Map(0 -> 0.5))).toDF("a", "b")
    // TODO: Currently, just check if no exception thrown
    assert(df6.dimsum_mapper(df6("a"), df6("b")).collect.isEmpty)
  }

  test("knn.distance") {
    val df1 = DummyInputData.select(hamming_distance(lit(1), lit(3)))
    checkAnswer(df1, Row(1))

    val df2 = DummyInputData.select(popcnt(lit(1)))
    checkAnswer(df2, Row(1))

    val rows3 = DummyInputData.select(kld(lit(0.1), lit(0.5), lit(0.2), lit(0.5))).collect
    assert(rows3.length === 1)
    assert(rows3(0).getDouble(0) ~== 0.01)

    val rows4 = DummyInputData.select(
      euclid_distance(typedLit(Seq("0.1", "0.5")), typedLit(Seq("0.2", "0.5")))).collect
    assert(rows4.length === 1)
    assert(rows4(0).getFloat(0) ~== 1.4142135f)

    val rows5 = DummyInputData.select(
      cosine_distance(typedLit(Seq("0.8", "0.3")), typedLit(Seq("0.4", "0.6")))).collect
    assert(rows5.length === 1)
    assert(rows5(0).getFloat(0) ~== 1.0f)

    val rows6 = DummyInputData.select(
      angular_distance(typedLit(Seq("0.1", "0.1")), typedLit(Seq("0.3", "0.8")))).collect
    assert(rows6.length === 1)
    assert(rows6(0).getFloat(0) ~== 0.50f)

    val rows7 = DummyInputData.select(
      manhattan_distance(typedLit(Seq("0.7", "0.8")), typedLit(Seq("0.5", "0.6")))).collect
    assert(rows7.length === 1)
    assert(rows7(0).getFloat(0) ~== 4.0f)

    val rows8 = DummyInputData.select(
      minkowski_distance(typedLit(Seq("0.1", "0.2")), typedLit(Seq("0.2", "0.2")), typedLit(1.0))
    ).collect
    assert(rows8.length === 1)
    assert(rows8(0).getFloat(0) ~== 2.0f)

    val rows9 = DummyInputData.select(
      jaccard_distance(typedLit(Seq("0.3", "0.8")), typedLit(Seq("0.1", "0.2")))).collect
    assert(rows9.length === 1)
    assert(rows9(0).getFloat(0) ~== 1.0f)
  }

  test("knn.lsh") {
    import hiveContext.implicits._
    checkAnswer(
      IntList2Data.minhash(lit(1), $"target").select($"clusterid", $"item"),
      Row(1016022700, 1) ::
      Row(1264890450, 1) ::
      Row(1304330069, 1) ::
      Row(1321870696, 1) ::
      Row(1492709716, 1) ::
      Row(1511363108, 1) ::
      Row(1601347428, 1) ::
      Row(1974434012, 1) ::
      Row(2022223284, 1) ::
      Row(326269457, 1) ::
      Row(50559334, 1) ::
      Row(716040854, 1) ::
      Row(759249519, 1) ::
      Row(809187771, 1) ::
      Row(900899651, 1) ::
      Nil
    )
    checkAnswer(
      DummyInputData.select(bbit_minhash(typedLit(Seq("1:0.1", "2:0.5")), lit(false))),
      Row("31175986876675838064867796245644543067")
    )
    checkAnswer(
      DummyInputData.select(minhashes(typedLit(Seq("1:0.1", "2:0.5")), lit(false))),
      Row(Seq(1571683640, 987207869, 370931990, 988455638, 846963275))
    )
  }

  test("ftvec - add_bias") {
    import hiveContext.implicits._
    checkAnswer(TinyTrainData.select(add_bias($"features")),
        Row(Seq("1:0.8", "2:0.2", "0:1.0")) ::
        Row(Seq("2:0.7", "0:1.0")) ::
        Row(Seq("1:0.9", "0:1.0")) ::
        Nil
      )
  }

  test("ftvec - extract_feature") {
    val df = DummyInputData.select(extract_feature(lit("1:0.8")))
    checkAnswer(df, Row("1"))
  }

  test("ftvec - extract_weight") {
    val rows = DummyInputData.select(extract_weight(lit("3:0.1"))).collect
    assert(rows.length === 1)
    assert(rows(0).getDouble(0) ~== 0.1)
  }

  test("ftvec - explode_array") {
    import hiveContext.implicits._
    val df = TinyTrainData.explode_array($"features").select($"feature")
    checkAnswer(df, Row("1:0.8") :: Row("2:0.2") :: Row("2:0.7") :: Row("1:0.9") :: Nil)
  }

  test("ftvec - add_feature_index") {
    import hiveContext.implicits._
    val doubleListData = Seq(Array(0.8, 0.5), Array(0.3, 0.1), Array(0.2)).toDF("data")
    checkAnswer(
        doubleListData.select(add_feature_index($"data")),
        Row(Seq("1:0.8", "2:0.5")) ::
        Row(Seq("1:0.3", "2:0.1")) ::
        Row(Seq("1:0.2")) ::
        Nil
      )
  }

  test("ftvec - sort_by_feature") {
    // import hiveContext.implicits._
    val intFloatMapData = {
      // TODO: Use `toDF`
      val rowRdd = hiveContext.sparkContext.parallelize(
          Row(Map(1 -> 0.3f, 2 -> 0.1f, 3 -> 0.5f)) ::
          Row(Map(2 -> 0.4f, 1 -> 0.2f)) ::
          Row(Map(2 -> 0.4f, 3 -> 0.2f, 1 -> 0.1f, 4 -> 0.6f)) ::
          Nil
        )
      hiveContext.createDataFrame(
        rowRdd,
        StructType(
          StructField("data", MapType(IntegerType, FloatType), true) ::
          Nil)
        )
    }
    val sortedKeys = intFloatMapData.select(sort_by_feature(intFloatMapData.col("data")))
      .collect.map {
        case Row(m: Map[Int, Float]) => m.keysIterator.toSeq
    }
    assert(sortedKeys.toSet === Set(Seq(1, 2, 3), Seq(1, 2), Seq(1, 2, 3, 4)))
  }

  test("ftvec.hash") {
    checkAnswer(DummyInputData.select(mhash(lit("test"))), Row(4948445))
    checkAnswer(DummyInputData.select(HivemallOps.sha1(lit("test"))), Row(12184508))
    checkAnswer(DummyInputData.select(feature_hashing(typedLit(Seq("1:0.1", "3:0.5")))),
      Row(Seq("11293631:0.1", "4331412:0.5")))
    checkAnswer(DummyInputData.select(array_hash_values(typedLit(Seq("aaa", "bbb")))),
      Row(Seq(4063537, 8459207)))
    checkAnswer(DummyInputData.select(
        prefixed_hash_values(typedLit(Seq("ccc", "ddd")), lit("prefix"))),
      Row(Seq("prefix7873825", "prefix8965544")))
  }

  test("ftvec.parting") {
    checkAnswer(DummyInputData.select(polynomial_features(typedLit(Seq("2:0.4", "6:0.1")), lit(2))),
      Row(Seq("2:0.4", "2^2:0.16000001", "2^6:0.040000003", "6:0.1", "6^6:0.010000001")))
    checkAnswer(DummyInputData.select(powered_features(typedLit(Seq("4:0.8", "5:0.2")), lit(2))),
      Row(Seq("4:0.8", "4^2:0.64000005", "5:0.2", "5^2:0.040000003")))
  }

  test("ftvec.scaling") {
    val rows1 = TinyTrainData.select(rescale(lit(2.0f), lit(1.0), lit(5.0))).collect
    assert(rows1.length === 3)
    assert(rows1(0).getFloat(0) ~== 0.25f)
    assert(rows1(1).getFloat(0) ~== 0.25f)
    assert(rows1(2).getFloat(0) ~== 0.25f)
    val rows2 = TinyTrainData.select(zscore(lit(1.0f), lit(0.5), lit(0.5))).collect
    assert(rows2.length === 3)
    assert(rows2(0).getFloat(0) ~== 1.0f)
    assert(rows2(1).getFloat(0) ~== 1.0f)
    assert(rows2(2).getFloat(0) ~== 1.0f)
    val df3 = TinyTrainData.select(l2_normalize(TinyTrainData.col("features")))
    checkAnswer(
      df3,
      Row(Seq("1:0.9701425", "2:0.24253562")) ::
      Row(Seq("2:1.0")) ::
      Row(Seq("1:1.0")) ::
      Nil)
  }

  test("ftvec.selection - chi2") {
    import hiveContext.implicits._

    // See also hivemall.ftvec.selection.ChiSquareUDFTest
    val df = Seq(
      Seq(
        Seq(250.29999999999998, 170.90000000000003, 73.2, 12.199999999999996),
        Seq(296.8, 138.50000000000003, 212.99999999999997, 66.3),
        Seq(329.3999999999999, 148.7, 277.59999999999997, 101.29999999999998)
      ) -> Seq(
        Seq(292.1666753739119, 152.70000455081467, 187.93333893418327, 59.93333511948589),
        Seq(292.1666753739119, 152.70000455081467, 187.93333893418327, 59.93333511948589),
        Seq(292.1666753739119, 152.70000455081467, 187.93333893418327, 59.93333511948589)))
      .toDF("arg0", "arg1")

    val rows = df.select(chi2(df("arg0"), df("arg1"))).collect
    assert(rows.length == 1)
    val chi2Val = rows.head.getAs[Row](0).getAs[Seq[Double]](0)
    val pVal = rows.head.getAs[Row](0).getAs[Seq[Double]](1)

    (chi2Val, Seq(10.81782088, 3.59449902, 116.16984746, 67.24482759))
      .zipped
      .foreach((actual, expected) => assert(actual ~== expected))

    (pVal, Seq(4.47651499e-03, 1.65754167e-01, 5.94344354e-26, 2.50017968e-15))
      .zipped
      .foreach((actual, expected) => assert(actual ~== expected))
  }

  test("ftvec.conv - quantify") {
    import hiveContext.implicits._
    val testDf = Seq((1, "aaa", true), (2, "bbb", false), (3, "aaa", false)).toDF
    // This test is done in a single partition because `HivemallOps#quantify` assigns identifiers
    // for non-numerical values in each partition.
    checkAnswer(
      testDf.coalesce(1).quantify(lit(true) +: testDf.cols: _*).select($"c0", $"c1", $"c2"),
      Row(1, 0, 0) :: Row(2, 1, 1) :: Row(3, 0, 1) :: Nil)
  }

  test("ftvec.amplify") {
    import hiveContext.implicits._
    assert(TinyTrainData.amplify(lit(3), $"label", $"features").count() == 9)
    assert(TinyTrainData.part_amplify(lit(3)).count() == 9)
    // TODO: The test below failed because:
    //   java.lang.RuntimeException: Unsupported literal type class scala.Tuple3
    //     (-buf 128,label,features)
    //
    // assert(TinyTrainData.rand_amplify(lit(3), lit("-buf 8", $"label", $"features")).count() == 9)
  }

  test("ftvec.conv") {
    import hiveContext.implicits._

    checkAnswer(
      DummyInputData.select(to_dense_features(typedLit(Seq("0:0.1", "1:0.3")), lit(1))),
      Row(Array(0.1f, 0.3f))
    )
    checkAnswer(
      DummyInputData.select(to_sparse_features(typedLit(Seq(0.1f, 0.2f, 0.3f)))),
      Row(Seq("0:0.1", "1:0.2", "2:0.3"))
    )
    checkAnswer(
      DummyInputData.select(feature_binning(typedLit(Seq("1")), typedLit(Map("1" -> Seq(0, 3))))),
      Row(Seq("1"))
    )
  }

  test("ftvec.trans") {
    import hiveContext.implicits._

    checkAnswer(
      DummyInputData.select(vectorize_features(typedLit(Seq("a", "b")), lit(0.1f), lit(0.2f))),
      Row(Seq("a:0.1", "b:0.2"))
    )
    checkAnswer(
      DummyInputData.select(categorical_features(typedLit(Seq("a", "b")), lit("c11"), lit("c12"))),
      Row(Seq("a#c11", "b#c12"))
    )
    checkAnswer(
      DummyInputData.select(indexed_features(lit(0.1), lit(0.2), lit(0.3))),
      Row(Seq("1:0.1", "2:0.2", "3:0.3"))
    )
    checkAnswer(
      DummyInputData.select(quantitative_features(typedLit(Seq("a", "b")), lit(0.1), lit(0.2))),
      Row(Seq("a:0.1", "b:0.2"))
    )
    checkAnswer(
      DummyInputData.select(ffm_features(typedLit(Seq("1", "2")), lit(0.5), lit(0.2))),
      Row(Seq("190:140405:1", "111:1058718:1"))
    )
    checkAnswer(
      DummyInputData.select(add_field_indices(typedLit(Seq("0.5", "0.1")))),
      Row(Seq("1:0.5", "2:0.1"))
    )

    val df1 = Seq((1, -3, 1), (2, -2, 1)).toDF("a", "b", "c")
    checkAnswer(
      df1.binarize_label($"a", $"b", $"c").select($"c0", $"c1"),
      Row(1, 1) :: Row(1, 1) :: Row(1, 1) :: Nil
    )
    val df2 = Seq(("xxx", "yyy", 0), ("zzz", "yyy", 1)).toDF("a", "b", "c").coalesce(1)
    checkAnswer(
      df2.quantified_features(lit(true), df2("a"), df2("b"), df2("c")).select($"features"),
      Row(Seq(0.0, 0.0, 0.0)) :: Row(Seq(1.0, 0.0, 1.0)) :: Nil
    )
  }

  test("ftvec.ranking") {
    import hiveContext.implicits._

    val df1 = Seq((1, 0 :: 3 :: 4 :: Nil), (2, 8 :: 9 :: Nil)).toDF("a", "b").coalesce(1)
    checkAnswer(
      df1.bpr_sampling($"a", $"b").select($"user", $"pos_item", $"neg_item"),
      Row(1, 0, 7) ::
      Row(1, 3, 6) ::
      Row(2, 8, 0) ::
      Row(2, 8, 4) ::
      Row(2, 9, 7) ::
      Nil
    )
    val df2 = Seq(1 :: 8 :: 9 :: Nil, 0 :: 3 :: Nil).toDF("a").coalesce(1)
    checkAnswer(
      df2.item_pairs_sampling($"a", lit(3)).select($"pos_item_id", $"neg_item_id"),
      Row(0, 1) ::
      Row(1, 0) ::
      Row(3, 2) ::
      Nil
    )
    val df3 = Seq(3 :: 5 :: Nil, 0 :: Nil).toDF("a").coalesce(1)
    checkAnswer(
      df3.populate_not_in($"a", lit(1)).select($"item"),
      Row(0) ::
      Row(1) ::
      Row(1) ::
      Nil
    )
  }

  test("tools") {
    // checkAnswer(
    //   DummyInputData.select(convert_label(lit(5))),
    //   Nil
    // )
    checkAnswer(
      DummyInputData.select(x_rank(lit("abc"))),
      Row(1)
    )
  }

  test("tools.array") {
    checkAnswer(
      DummyInputData.select(float_array(lit(3))),
      Row(Seq())
    )
    checkAnswer(
      DummyInputData.select(array_remove(typedLit(Seq(1, 2, 3)), lit(2))),
      Row(Seq(1, 3))
    )
    checkAnswer(
      DummyInputData.select(sort_and_uniq_array(typedLit(Seq(2, 1, 3, 1)))),
      Row(Seq(1, 2, 3))
    )
    checkAnswer(
      DummyInputData.select(subarray_endwith(typedLit(Seq(1, 2, 3, 4, 5)), lit(4))),
      Row(Seq(1, 2, 3, 4))
    )
    checkAnswer(
      DummyInputData.select(
        array_concat(typedLit(Seq(1, 2)), typedLit(Seq(3)), typedLit(Seq(4, 5)))),
      Row(Seq(1, 2, 3, 4, 5))
    )
    checkAnswer(
      DummyInputData.select(subarray(typedLit(Seq(1, 2, 3, 4, 5)), lit(2), lit(4))),
      Row(Seq(3, 4, 5))
    )
    checkAnswer(
      DummyInputData.select(to_string_array(typedLit(Seq(1, 2, 3, 4, 5)))),
      Row(Seq("1", "2", "3", "4", "5"))
    )
    checkAnswer(
      DummyInputData.select(array_intersect(typedLit(Seq(1, 2, 3)), typedLit(Seq(2, 3, 4)))),
      Row(Seq(2, 3))
    )
  }

  test("tools.array - select_k_best") {
    import hiveContext.implicits._

    val data = Seq(Seq(0, 1, 3), Seq(2, 4, 1), Seq(5, 4, 9))
    val df = data.map(d => (d, Seq(3, 1, 2))).toDF("features", "importance_list")
    val k = 2

    checkAnswer(
      df.select(select_k_best(df("features"), df("importance_list"), lit(k))),
      Row(Seq(0.0, 3.0)) :: Row(Seq(2.0, 1.0)) :: Row(Seq(5.0, 9.0)) :: Nil
    )
  }

  test("tools.bits") {
    checkAnswer(
      DummyInputData.select(to_bits(typedLit(Seq(1, 3, 9)))),
      Row(Seq(522L))
    )
    checkAnswer(
      DummyInputData.select(unbits(typedLit(Seq(1L, 3L)))),
      Row(Seq(0L, 64L, 65L))
    )
    checkAnswer(
      DummyInputData.select(bits_or(typedLit(Seq(1L, 3L)), typedLit(Seq(8L, 23L)))),
      Row(Seq(9L, 23L))
    )
  }

  test("tools.compress") {
    checkAnswer(
      DummyInputData.select(inflate(deflate(lit("input text")))),
      Row("input text")
    )
  }

  test("tools.map") {
    val rows = DummyInputData.select(
      map_get_sum(typedLit(Map(1 -> 0.2f, 2 -> 0.5f, 4 -> 0.8f)), typedLit(Seq(1, 4)))
    ).collect
    assert(rows.length === 1)
    assert(rows(0).getDouble(0) ~== 1.0f)

    checkAnswer(
      DummyInputData.select(map_tail_n(typedLit(Map(1 -> 2, 2 -> 5)), lit(1))),
      Row(Map(2 -> 5))
    )
  }

  test("tools.text") {
    checkAnswer(
      DummyInputData.select(tokenize(lit("This is a pen"))),
      Row("This" :: "is" :: "a" :: "pen" :: Nil)
    )
    checkAnswer(
      DummyInputData.select(is_stopword(lit("because"))),
      Row(true)
    )
    checkAnswer(
      DummyInputData.select(singularize(lit("between"))),
      Row("between")
    )
    checkAnswer(
      DummyInputData.select(split_words(lit("Hello, world"))),
      Row("Hello," :: "world" :: Nil)
    )
    checkAnswer(
      DummyInputData.select(normalize_unicode(lit("abcdefg"))),
      Row("abcdefg")
    )
    checkAnswer(
      DummyInputData.select(base91(typedLit("input text".getBytes))),
      Row("xojg[@TX;R..B")
    )
    checkAnswer(
      DummyInputData.select(unbase91(lit("XXXX"))),
      Row(68 :: -120 :: 8 :: Nil)
    )
    checkAnswer(
      DummyInputData.select(word_ngrams(typedLit("abcd" :: "efg" :: "hij" :: Nil), lit(2), lit(2))),
      Row("abcd efg" :: "efg hij" :: Nil)
    )
  }

  test("tools - generated_series") {
    import hiveContext.implicits._
    checkAnswer(
      DummyInputData.generate_series(lit(0), lit(3)).select($"generate_series"),
      Row(0) :: Row(1) :: Row(2) :: Row(3) :: Nil
    )
  }

  test("geospatial") {
    val rows1 = DummyInputData.select(tilex2lon(lit(1), lit(6))).collect
    assert(rows1.length === 1)
    assert(rows1(0).getDouble(0) ~== -174.375)

    val rows2 = DummyInputData.select(tiley2lat(lit(1), lit(3))).collect
    assert(rows2.length === 1)
    assert(rows2(0).getDouble(0) ~== 79.17133464081945)

    val rows3 = DummyInputData.select(
      haversine_distance(lit(0.3), lit(0.1), lit(0.4), lit(0.1))).collect
    assert(rows3.length === 1)
    assert(rows3(0).getDouble(0) ~== 11.119492664455878)

    checkAnswer(
      DummyInputData.select(tile(lit(0.1), lit(0.8), lit(3))),
      Row(28)
    )
    checkAnswer(
      DummyInputData.select(map_url(lit(0.1), lit(0.8), lit(3))),
      Row("http://tile.openstreetmap.org/3/4/3.png")
    )
    checkAnswer(
      DummyInputData.select(lat2tiley(lit(0.3), lit(3))),
      Row(3)
    )
    checkAnswer(
      DummyInputData.select(lon2tilex(lit(0.4), lit(2))),
      Row(2)
    )
  }

  test("misc - hivemall_version") {
    checkAnswer(DummyInputData.select(hivemall_version()), Row("0.5.2-incubating"))
  }

  test("misc - rowid") {
    assert(DummyInputData.select(rowid()).distinct.count == DummyInputData.count)
  }

  test("misc - each_top_k") {
    import hiveContext.implicits._
    val inputDf = Seq(
      ("a", "1", 0.5, 0.1, Array(0, 1, 2)),
      ("b", "5", 0.1, 0.2, Array(3)),
      ("a", "3", 0.8, 0.8, Array(2, 5)),
      ("c", "6", 0.3, 0.3, Array(1, 3)),
      ("b", "4", 0.3, 0.4, Array(2)),
      ("a", "2", 0.6, 0.5, Array(1))
    ).toDF("key", "value", "x", "y", "data")

    // Compute top-1 rows for each group
    val distance = sqrt(inputDf("x") * inputDf("x") + inputDf("y") * inputDf("y")).as("score")
    val top1Df = inputDf.each_top_k(lit(1), distance, $"key".as("group"))
    assert(top1Df.schema.toSet === Set(
      StructField("rank", IntegerType, nullable = true),
      StructField("score", DoubleType, nullable = true),
      StructField("key", StringType, nullable = true),
      StructField("value", StringType, nullable = true),
      StructField("x", DoubleType, nullable = true),
      StructField("y", DoubleType, nullable = true),
      StructField("data", ArrayType(IntegerType, containsNull = false), nullable = true)
    ))
    checkAnswer(
      top1Df.select($"rank", $"key", $"value", $"data"),
      Row(1, "a", "3", Array(2, 5)) ::
      Row(1, "b", "4", Array(2)) ::
      Row(1, "c", "6", Array(1, 3)) ::
      Nil
    )

    // Compute reverse top-1 rows for each group
    val bottom1Df = inputDf.each_top_k(lit(-1), distance, $"key".as("group"))
    checkAnswer(
      bottom1Df.select($"rank", $"key", $"value", $"data"),
      Row(1, "a", "1", Array(0, 1, 2)) ::
      Row(1, "b", "5", Array(3)) ::
      Row(1, "c", "6", Array(1, 3)) ::
      Nil
    )

    // Check if some exceptions thrown in case of some conditions
    assert(intercept[AnalysisException] { inputDf.each_top_k(lit(0.1), $"score", $"key") }
      .getMessage contains "`k` must be integer, however")
    assert(intercept[AnalysisException] { inputDf.each_top_k(lit(1), $"data", $"key") }
      .getMessage contains "must have a comparable type")
  }

  test("misc - join_top_k") {
    Seq("true", "false").map { flag =>
      withSQLConf(SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> flag) {
        import hiveContext.implicits._
        val inputDf = Seq(
          ("user1", 1, 0.3, 0.5),
          ("user2", 2, 0.1, 0.1),
          ("user3", 3, 0.8, 0.0),
          ("user4", 1, 0.9, 0.9),
          ("user5", 3, 0.7, 0.2),
          ("user6", 1, 0.5, 0.4),
          ("user7", 2, 0.6, 0.8)
        ).toDF("userId", "group", "x", "y")

        val masterDf = Seq(
          (1, "pos1-1", 0.5, 0.1),
          (1, "pos1-2", 0.0, 0.0),
          (1, "pos1-3", 0.3, 0.3),
          (2, "pos2-3", 0.1, 0.3),
          (2, "pos2-3", 0.8, 0.8),
          (3, "pos3-1", 0.1, 0.7),
          (3, "pos3-1", 0.7, 0.1),
          (3, "pos3-1", 0.9, 0.0),
          (3, "pos3-1", 0.1, 0.3)
        ).toDF("group", "position", "x", "y")

        // Compute top-1 rows for each group
        val distance = sqrt(
          pow(inputDf("x") - masterDf("x"), lit(2.0)) +
            pow(inputDf("y") - masterDf("y"), lit(2.0))
        ).as("score")
        val top1Df = inputDf.top_k_join(
          lit(1), masterDf, inputDf("group") === masterDf("group"), distance)
        assert(top1Df.schema.toSet === Set(
          StructField("rank", IntegerType, nullable = true),
          StructField("score", DoubleType, nullable = true),
          StructField("group", IntegerType, nullable = false),
          StructField("userId", StringType, nullable = true),
          StructField("position", StringType, nullable = true),
          StructField("x", DoubleType, nullable = false),
          StructField("y", DoubleType, nullable = false)
        ))
        checkAnswer(
          top1Df.select($"rank", inputDf("group"), $"userId", $"position"),
          Row(1, 1, "user1", "pos1-2") ::
          Row(1, 2, "user2", "pos2-3") ::
          Row(1, 3, "user3", "pos3-1") ::
          Row(1, 1, "user4", "pos1-2") ::
          Row(1, 3, "user5", "pos3-1") ::
          Row(1, 1, "user6", "pos1-2") ::
          Row(1, 2, "user7", "pos2-3") ::
          Nil
        )
      }
    }
  }

  test("HIVEMALL-76 top-K funcs must assign the same rank with the rows having the same scores") {
    import hiveContext.implicits._
    val inputDf = Seq(
      ("a", "1", 0.1),
      ("b", "5", 0.1),
      ("a", "3", 0.1),
      ("b", "4", 0.1),
      ("a", "2", 0.0)
    ).toDF("key", "value", "x")

    // Compute top-2 rows for each group
    val top2Df = inputDf.each_top_k(lit(2), $"x".as("score"), $"key".as("group"))
    checkAnswer(
      top2Df.select($"rank", $"score", $"key", $"value"),
      Row(1, 0.1, "a", "3") ::
      Row(1, 0.1, "a", "1") ::
      Row(1, 0.1, "b", "4") ::
      Row(1, 0.1, "b", "5") ::
      Nil
    )
    Seq("true", "false").map { flag =>
      withSQLConf(SQLConf.WHOLESTAGE_CODEGEN_ENABLED.key -> flag) {
        val inputDf = Seq(
          ("user1", 1, 0.3, 0.5),
          ("user2", 2, 0.1, 0.1)
        ).toDF("userId", "group", "x", "y")

        val masterDf = Seq(
          (1, "pos1-1", 0.5, 0.1),
          (1, "pos1-2", 0.5, 0.1),
          (1, "pos1-3", 0.3, 0.4),
          (2, "pos2-1", 0.8, 0.2),
          (2, "pos2-2", 0.8, 0.2)
        ).toDF("group", "position", "x", "y")

        // Compute top-2 rows for each group
        val distance = sqrt(
          pow(inputDf("x") - masterDf("x"), lit(2.0)) +
            pow(inputDf("y") - masterDf("y"), lit(2.0))
        ).as("score")
        val top2Df = inputDf.top_k_join(
          lit(2), masterDf, inputDf("group") === masterDf("group"), distance)
        checkAnswer(
          top2Df.select($"rank", inputDf("group"), $"userId", $"position"),
          Row(1, 1, "user1", "pos1-1") ::
          Row(1, 1, "user1", "pos1-2") ::
          Row(1, 2, "user2", "pos2-1") ::
          Row(1, 2, "user2", "pos2-2") ::
          Nil
        )
      }
    }
  }

  test("misc - flatten") {
    import hiveContext.implicits._
    val df = Seq((0, (1, "a", (3.0, "b")), (5, 0.9, "c", "d"), 9)).toDF()
    assert(df.flatten().schema === StructType(
      StructField("_1", IntegerType, nullable = false) ::
      StructField("_2$_1", IntegerType, nullable = true) ::
      StructField("_2$_2", StringType, nullable = true) ::
      StructField("_2$_3$_1", DoubleType, nullable = true) ::
      StructField("_2$_3$_2", StringType, nullable = true) ::
      StructField("_3$_1", IntegerType, nullable = true) ::
      StructField("_3$_2", DoubleType, nullable = true) ::
      StructField("_3$_3", StringType, nullable = true) ::
      StructField("_3$_4", StringType, nullable = true) ::
      StructField("_4", IntegerType, nullable = false) ::
      Nil
    ))
    checkAnswer(df.flatten("$").select("_2$_1"), Row(1))
    checkAnswer(df.flatten("_").select("_2__1"), Row(1))
    checkAnswer(df.flatten(".").select("`_2._1`"), Row(1))

    val errMsg1 = intercept[IllegalArgumentException] { df.flatten("\t") }
    assert(errMsg1.getMessage.startsWith("Must use '$', '_', or '.' for separator, but got"))
    val errMsg2 = intercept[IllegalArgumentException] { df.flatten("12") }
    assert(errMsg2.getMessage.startsWith("Separator cannot be more than one character:"))
  }

  test("misc - from_csv") {
    import hiveContext.implicits._
    val df = Seq("""1,abc""").toDF()
    val schema = new StructType().add("a", IntegerType).add("b", StringType)
    checkAnswer(
      df.select(from_csv($"value", schema)),
      Row(Row(1, "abc")))
  }

  test("misc - to_csv") {
    import hiveContext.implicits._
    val df = Seq((1, "a", (0, 3.9, "abc")), (8, "c", (2, 0.4, "def"))).toDF()
    checkAnswer(
      df.select(to_csv($"_3")),
      Row("0,3.9,abc") ::
      Row("2,0.4,def") ::
      Nil)
  }

  /**
   * This test fails because;
   *
   * Cause: java.lang.OutOfMemoryError: Java heap space
   *  at hivemall.smile.tools.RandomForestEnsembleUDAF$Result.<init>
   *    (RandomForestEnsembleUDAF.java:128)
   *  at hivemall.smile.tools.RandomForestEnsembleUDAF$RandomForestPredictUDAFEvaluator
   *    .terminate(RandomForestEnsembleUDAF.java:91)
   *  at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
   */
  ignore("misc - tree_predict") {
    import hiveContext.implicits._

    val model = Seq((0.0, 0.1 :: 0.1 :: Nil), (1.0, 0.2 :: 0.3 :: 0.2 :: Nil))
      .toDF("label", "features")
      .train_randomforest_regressor($"features", $"label")

    val testData = Seq((0.0, 0.1 :: 0.0 :: Nil), (1.0, 0.3 :: 0.5 :: 0.4 :: Nil))
      .toDF("label", "features")
      .select(rowid(), $"label", $"features")

    val predicted = model
      .join(testData).coalesce(1)
      .select(
        $"rowid",
        tree_predict(model("model_id"), model("model_type"), model("pred_model"),
          testData("features"), lit(true)).as("predicted")
      )
      .groupBy($"rowid")
      .rf_ensemble("predicted").toDF("rowid", "predicted")
      .select($"predicted.label")

    checkAnswer(predicted, Seq(Row(0), Row(1)))
  }

  test("misc - sigmoid") {
    import hiveContext.implicits._
    val rows = DummyInputData.select(sigmoid($"c0")).collect
    assert(rows.length === 1)
    assert(rows(0).getDouble(0) ~== 0.500)
  }

  test("misc - lr_datagen") {
    assert(TinyTrainData.lr_datagen(lit("-n_examples 100 -n_features 10 -seed 100")).count >= 100)
  }

  test("invoke regression functions") {
    import hiveContext.implicits._
    Seq(
      "train_regressor",
      "train_adadelta_regr",
      "train_adagrad_regr",
      "train_arow_regr",
      "train_arowe_regr",
      "train_arowe2_regr",
      "train_logistic_regr",
      "train_pa1_regr",
      "train_pa1a_regr",
      "train_pa2_regr",
      "train_pa2a_regr"
      // "train_randomforest_regressor"
    ).map { func =>
      TestUtils.invokeFunc(new HivemallOps(TinyTrainData), func, Seq($"features", $"label"))
        .foreach(_ => {}) // Just call it
    }
  }

  test("invoke classifier functions") {
    import hiveContext.implicits._
    Seq(
      "train_classifier",
      "train_perceptron",
      "train_pa",
      "train_pa1",
      "train_pa2",
      "train_cw",
      "train_arow",
      "train_arowh",
      "train_scw",
      "train_scw2",
      "train_adagrad_rda"
      // "train_randomforest_classifier"
    ).map { func =>
      TestUtils.invokeFunc(new HivemallOps(TinyTrainData), func, Seq($"features", $"label"))
        .foreach(_ => {}) // Just call it
    }
  }

  test("invoke multiclass classifier functions") {
    import hiveContext.implicits._
    Seq(
      "train_multiclass_perceptron",
      "train_multiclass_pa",
      "train_multiclass_pa1",
      "train_multiclass_pa2",
      "train_multiclass_cw",
      "train_multiclass_arow",
      "train_multiclass_arowh",
      "train_multiclass_scw",
      "train_multiclass_scw2"
    ).map { func =>
      // TODO: Why is a label type [Int|Text] only in multiclass classifiers?
      TestUtils.invokeFunc(
          new HivemallOps(TinyTrainData), func, Seq($"features", $"label".cast(IntegerType)))
        .foreach(_ => {}) // Just call it
    }
  }

  test("invoke random forest functions") {
    import hiveContext.implicits._
    val testDf = Seq(
      (Array(0.3, 0.1, 0.2), 1),
      (Array(0.3, 0.1, 0.2), 0),
      (Array(0.3, 0.1, 0.2), 0)).toDF("features", "label")
    Seq(
      "train_randomforest_regressor",
      "train_randomforest_classifier"
    ).map { func =>
      TestUtils.invokeFunc(new HivemallOps(testDf.coalesce(1)), func, Seq($"features", $"label"))
        .foreach(_ => {}) // Just call it
    }
  }

  test("invoke recommend functions") {
    import hiveContext.implicits._
    val df = Seq((1, Map(1 -> 0.3), Map(2 -> Map(4 -> 0.1)), 0, Map(3 -> 0.5)))
      .toDF("i", "r_i", "topKRatesOfI", "j", "r_j")
    // Just call it
    df.train_slim($"i", $"r_i", $"topKRatesOfI", $"j", $"r_j").collect

  }

  ignore("invoke topicmodel functions") {
    import hiveContext.implicits._
    val testDf = Seq(Seq("abcd", "'efghij", "klmn")).toDF("words")
    Seq(
      "train_lda",
      "train_plsa"
    ).map { func =>
      TestUtils.invokeFunc(new HivemallOps(testDf.coalesce(1)), func, Seq($"words"))
        .foreach(_ => {}) // Just call it
    }
  }

  protected def checkRegrPrecision(func: String): Unit = {
    import hiveContext.implicits._

    // Build a model
    val model = {
      val res = TestUtils.invokeFunc(new HivemallOps(LargeRegrTrainData),
        func, Seq(add_bias($"features"), $"label"))
      if (!res.columns.contains("conv")) {
        res.groupBy("feature").agg("weight" -> "avg")
      } else {
        res.groupBy("feature").argmin_kld("weight", "conv")
      }
    }.toDF("feature", "weight")

    // Data preparation
    val testDf = LargeRegrTrainData
      .select(rowid(), $"label".as("target"), $"features")
      .cache

    val testDf_exploded = testDf
      .explode_array($"features")
      .select($"rowid", extract_feature($"feature"), extract_weight($"feature"))

    // Do prediction
    val predict = testDf_exploded
      .join(model, testDf_exploded("feature") === model("feature"), "LEFT_OUTER")
      .select($"rowid", ($"weight" * $"value").as("value"))
      .groupBy("rowid").sum("value")
      .toDF("rowid", "predicted")

    // Evaluation
    val eval = predict
      .join(testDf, predict("rowid") === testDf("rowid"))
      .groupBy()
      .agg(Map("target" -> "avg", "predicted" -> "avg"))
      .toDF("target", "predicted")

    val diff = eval.map {
      case Row(target: Double, predicted: Double) =>
        Math.abs(target - predicted)
    }.first

    TestUtils.expectResult(diff > 0.10, s"Low precision -> func:${func} diff:${diff}")
  }

  protected def checkClassifierPrecision(func: String): Unit = {
    import hiveContext.implicits._

    // Build a model
    val model = {
      val res = TestUtils.invokeFunc(new HivemallOps(LargeClassifierTrainData),
        func, Seq(add_bias($"features"), $"label"))
      if (!res.columns.contains("conv")) {
        res.groupBy("feature").agg("weight" -> "avg")
      } else {
        res.groupBy("feature").argmin_kld("weight", "conv")
      }
    }.toDF("feature", "weight")

    // Data preparation
    val testDf = LargeClassifierTestData
      .select(rowid(), $"label".as("target"), $"features")
      .cache

    val testDf_exploded = testDf
      .explode_array($"features")
      .select($"rowid", extract_feature($"feature"), extract_weight($"feature"))

    // Do prediction
    val predict = testDf_exploded
      .join(model, testDf_exploded("feature") === model("feature"), "LEFT_OUTER")
      .select($"rowid", ($"weight" * $"value").as("value"))
      .groupBy("rowid").sum("value")
      /**
       * TODO: This sentence throws an exception below:
       *
       * WARN Column: Constructing trivially true equals predicate, 'rowid#1323 = rowid#1323'.
       * Perhaps you need to use aliases.
       */
      .select($"rowid", when(sigmoid($"sum(value)") > 0.50, 1.0).otherwise(0.0))
      .toDF("rowid", "predicted")

    // Evaluation
    val eval = predict
      .join(testDf, predict("rowid") === testDf("rowid"))
      .where($"target" === $"predicted")

    val precision = (eval.count + 0.0) / predict.count

    TestUtils.expectResult(precision < 0.70, s"Low precision -> func:${func} value:${precision}")
  }

  ignore("check regression precision") {
    Seq(
      "train_adadelta_regr",
      "train_adagrad_regr",
      "train_arow_regr",
      "train_arowe_regr",
      "train_arowe2_regr",
      "train_logistic_regr",
      "train_pa1_regr",
      "train_pa1a_regr",
      "train_pa2_regr",
      "train_pa2a_regr"
    ).map { func =>
      checkRegrPrecision(func)
    }
  }

  ignore("check classifier precision") {
    Seq(
      "train_perceptron",
      "train_pa",
      "train_pa1",
      "train_pa2",
      "train_cw",
      "train_arow",
      "train_arowh",
      "train_scw",
      "train_scw2",
      "train_adagrad_rda"
    ).map { func =>
      checkClassifierPrecision(func)
    }
  }

  test("aggregations for classifiers") {
    import hiveContext.implicits._
    val df1 = Seq((1, 0.1, 0.1, 0.2f, 0.2f, 0.2f, 0.2f))
      .toDF("key", "xh", "xk", "w0", "w1", "w2", "w3")
    val row1 = df1.groupBy($"key").kpa_predict("xh", "xk", "w0", "w1", "w2", "w3").collect
    assert(row1.length === 1)
    assert(row1(0).getDouble(1) ~== 0.002000000029802)
  }

  test("aggregations for ensembles") {
    import hiveContext.implicits._

    val df1 = Seq((1, 0.1), (1, 0.2), (2, 0.1)).toDF("c0", "c1")
    val rows1 = df1.groupBy($"c0").voted_avg("c1").collect
    assert(rows1.length === 2)
    assert(rows1(0).getDouble(1) ~== 0.15)
    assert(rows1(1).getDouble(1) ~== 0.10)

    val df3 = Seq((1, 0.2), (1, 0.8), (2, 0.3)).toDF("c0", "c1")
    val rows3 = df3.groupBy($"c0").weight_voted_avg("c1").collect
    assert(rows3.length === 2)
    assert(rows3(0).getDouble(1) ~== 0.50)
    assert(rows3(1).getDouble(1) ~== 0.30)

    val df5 = Seq((1, 0.2f, 0.1f), (1, 0.4f, 0.2f), (2, 0.8f, 0.9f)).toDF("c0", "c1", "c2")
    val rows5 = df5.groupBy($"c0").argmin_kld("c1", "c2").collect
    assert(rows5.length === 2)
    assert(rows5(0).getFloat(1) ~== 0.266666666)
    assert(rows5(1).getFloat(1) ~== 0.80)

    val df6 = Seq((1, "id-0", 0.2), (1, "id-1", 0.4), (1, "id-2", 0.1)).toDF("c0", "c1", "c2")
    val rows6 = df6.groupBy($"c0").max_label("c2", "c1").collect
    assert(rows6.length === 1)
    assert(rows6(0).getString(1) == "id-1")

    val df7 = Seq((1, "id-0", 0.5), (1, "id-1", 0.1), (1, "id-2", 0.2)).toDF("c0", "c1", "c2")
    val rows7 = df7.groupBy($"c0").maxrow("c2", "c1").toDF("c0", "c1").select($"c1.col1").collect
    assert(rows7.length === 1)
    assert(rows7(0).getString(0) == "id-0")

    val df8 = Seq((1, 1), (1, 2), (2, 1), (1, 5)).toDF("c0", "c1")
    val rows8 = df8.groupBy($"c0").rf_ensemble("c1").toDF("c0", "c1")
      .select("c1.probability").collect
    assert(rows8.length === 2)
    assert(rows8(0).getDouble(0) ~== 0.3333333333)
    assert(rows8(1).getDouble(0) ~== 1.0)
  }

  test("aggregations for evaluation") {
    import hiveContext.implicits._

    val testDf1 = Seq((1, 1.0, 0.5), (1, 0.3, 0.5), (1, 0.1, 0.2)).toDF("c0", "c1", "c2")
    val rows1 = testDf1.groupBy($"c0").mae("c1", "c2").collect
    assert(rows1.length === 1)
    assert(rows1(0).getDouble(1) ~== 0.26666666)
    val rows2 = testDf1.groupBy($"c0").mse("c1", "c2").collect
    assert(rows2.length === 1)
    assert(rows2(0).getDouble(1) ~== 0.1)
    val rows3 = testDf1.groupBy($"c0").rmse("c1", "c2").collect
    assert(rows3.length === 1)
    assert(rows3(0).getDouble(1) ~== 0.31622776601683794)
    val rows4 = testDf1.groupBy($"c0").r2("c1", "c2").collect
    assert(rows4.length === 1)
    assert(rows4(0).getDouble(1) ~== -4.0)
    val rows5 = testDf1.groupBy($"c0").logloss("c1", "c2").collect
    assert(rows5.length === 1)
    assert(rows5(0).getDouble(1) ~== 6.198305767142615)

    val testDf2 = Seq((1, Array(1, 2), Array(2, 3)), (1, Array(3, 8), Array(5, 4)))
      .toDF("c0", "c1", "c2")
    val rows6 = testDf2.groupBy($"c0").ndcg("c1", "c2").collect
    assert(rows6.length === 1)
    assert(rows6(0).getDouble(1) ~== 0.19342640361727081)
    val rows7 = testDf2.groupBy($"c0").precision_at("c1", "c2").collect
    assert(rows7.length === 1)
    assert(rows7(0).getDouble(1) ~== 0.25)
    val rows8 = testDf2.groupBy($"c0").recall_at("c1", "c2").collect
    assert(rows8.length === 1)
    assert(rows8(0).getDouble(1) ~== 0.25)
    val rows9 = testDf2.groupBy($"c0").hitrate("c1", "c2").collect
    assert(rows9.length === 1)
    assert(rows9(0).getDouble(1) ~== 0.50)
    val rows10 = testDf2.groupBy($"c0").mrr("c1", "c2").collect
    assert(rows10.length === 1)
    assert(rows10(0).getDouble(1) ~== 0.25)
    val rows11 = testDf2.groupBy($"c0").average_precision("c1", "c2").collect
    assert(rows11.length === 1)
    assert(rows11(0).getDouble(1) ~== 0.25)
    val rows12 = testDf2.groupBy($"c0").auc("c1", "c2").collect
    assert(rows12.length === 1)
    assert(rows12(0).getDouble(1) ~== 0.25)
  }

  test("aggregations for topicmodel") {
    import hiveContext.implicits._

    val testDf = Seq((1, "abcd", 0.1, 0, 0.1), (1, "efgh", 0.2, 0, 0.1))
      .toDF("key", "word", "value", "label", "lambda")
    val rows1 = testDf.groupBy($"key").lda_predict("word", "value", "label", "lambda").collect
    assert(rows1.length === 1)
    val result1 = rows1(0).getSeq[Row](1).map { case Row(label: Int, prob: Float) => label -> prob }
      .toMap[Int, Float]
    assert(result1.size === 10)
    assert(result1(0) ~== 0.07692449)
    assert(result1(1) ~== 0.07701121)
    assert(result1(2) ~== 0.07701129)
    assert(result1(3) ~== 0.07705542)
    assert(result1(4) ~== 0.07701511)
    assert(result1(5) ~== 0.07701234)
    assert(result1(6) ~== 0.07701384)
    assert(result1(7) ~== 0.30693996)
    assert(result1(8) ~== 0.07700701)
    assert(result1(9) ~== 0.07700934)

    val rows2 = testDf.groupBy($"key").plsa_predict("word", "value", "label", "lambda").collect
    assert(rows2.length === 1)
    val result2 = rows2(0).getSeq[Row](1).map { case Row(label: Int, prob: Float) => label -> prob }
      .toMap[Int, Float]
    assert(result2.size === 10)
    assert(result2(0) ~== 0.062156882)
    assert(result2(1) ~== 0.05088547)
    assert(result2(2) ~== 0.12434204)
    assert(result2(3) ~== 0.31869823)
    assert(result2(4) ~== 0.01584355)
    assert(result2(5) ~== 0.0057667173)
    assert(result2(6) ~== 0.10864779)
    assert(result2(7) ~== 0.09346106)
    assert(result2(8) ~== 0.13905199)
    assert(result2(9) ~== 0.081146255)
  }

  test("aggregations for ftvec.text") {
    import hiveContext.implicits._
    val testDf = Seq((1, "abc def hi jk l"), (1, "def jk")).toDF("key", "text")
    val rows = testDf.groupBy($"key").tf("text").collect
    assert(rows.length === 1)
    val result = rows(0).getAs[Map[String, Float]](1)
    assert(result.size === 2)
    assert(result("def jk") ~== 0.5f)
    assert(result("abc def hi jk l") ~== 0.5f)
  }

  test("aggregations for tools.array") {
    import hiveContext.implicits._

    val testDf = Seq((1, 1 :: 3 :: Nil), (1, 3 :: 5 :: Nil)).toDF("key", "ar")
    val rows1 = testDf.groupBy($"key").array_avg("ar").collect
    assert(rows1.length === 1)
    val result1 = rows1(0).getSeq[Float](1)
    assert(result1.length === 2)
    assert(result1(0) ~== 2.0f)
    assert(result1(1) ~== 4.0f)

    val rows2 = testDf.groupBy($"key").array_sum("ar").collect
    assert(rows2.length === 1)
    val result2 = rows2(0).getSeq[Double](1)
    assert(result2.length === 2)
    assert(result2(0) ~== 4.0)
    assert(result2(1) ~== 8.0)
  }

  test("aggregations for tools.bits") {
    import hiveContext.implicits._
    val testDf = Seq((1, 1), (1, 7)).toDF("key", "x")
    val rows = testDf.groupBy($"key").bits_collect("x").collect
    assert(rows.length === 1)
    val result = rows(0).getSeq[Int](1)
    assert(result === Seq(130))
  }

  test("aggregations for tools.list") {
    import hiveContext.implicits._
    val testDf = Seq((1, 3), (1, 1), (1, 2)).toDF("key", "x")
    val rows = testDf.groupBy($"key").to_ordered_list("x").collect
    assert(rows.length === 1)
    val result = rows(0).getSeq[Int](1)
    assert(result === Seq(1, 2, 3))
  }

  test("aggregations for tools.map") {
    import hiveContext.implicits._
    val testDf = Seq((1, 1, "a"), (1, 2, "b"), (1, 3, "c")).toDF("key", "k", "v")
    val rows = testDf.groupBy($"key").to_map("k", "v").collect
    assert(rows.length === 1)
    val result = rows(0).getMap[Int, String](1)
    assert(result === Map(1 -> "a", 2 -> "b", 3 -> "c"))
  }

  test("aggregations for tools.math") {
    import hiveContext.implicits._
    val testDf = Seq(
      (1, Seq(1, 2, 3, 4), Seq(5, 6, 7, 8)),
      (1, Seq(9, 10, 11, 12), Seq(13, 14, 15, 16))
    ).toDF("key", "mtx1", "mtx2")
    val rows = testDf.groupBy($"key").transpose_and_dot("mtx1", "mtx2").collect
    assert(rows.length === 1)
    val result = rows(0).getSeq[Int](1)
    assert(result === Seq(
      Seq(122.0, 132.0, 142.0, 152.0),
      Seq(140.0, 152.0, 164.0, 176.0),
      Seq(158.0, 172.0, 186.0, 200.0),
      Seq(176.0, 192.0, 208.0, 224.0))
    )
  }

  test("aggregations for ftvec.trans") {
    import hiveContext.implicits._

    val df0 = Seq((1, "cat", "mammal", 9), (1, "dog", "mammal", 10), (1, "human", "mammal", 10),
      (1, "seahawk", "bird", 101), (1, "wasp", "insect", 3), (1, "wasp", "insect", 9),
      (1, "cat", "mammal", 101), (1, "dog", "mammal", 1), (1, "human", "mammal", 9))
      .toDF("col0", "cat1", "cat2", "cat3")
    val row00 = df0.groupBy($"col0").onehot_encoding("cat1")
    val row01 = df0.groupBy($"col0").onehot_encoding("cat1", "cat2", "cat3")

    val result000 = row00.collect()(0).getAs[Row](1).getAs[Map[String, Int]](0)
    val result01 = row01.collect()(0).getAs[Row](1)
    val result010 = result01.getAs[Map[String, Int]](0)
    val result011 = result01.getAs[Map[String, Int]](1)
    val result012 = result01.getAs[Map[String, Int]](2)

    assert(result000.keySet === Set("seahawk", "cat", "human", "wasp", "dog"))
    assert(result000.values.toSet === Set(1, 2, 3, 4, 5))
    assert(result010.keySet === Set("seahawk", "cat", "human", "wasp", "dog"))
    assert(result010.values.toSet === Set(1, 2, 3, 4, 5))
    assert(result011.keySet === Set("bird", "insect", "mammal"))
    assert(result011.values.toSet === Set(6, 7, 8))
    assert(result012.keySet === Set(1, 3, 9, 10, 101))
    assert(result012.values.toSet === Set(9, 10, 11, 12, 13))
  }

  test("aggregations for ftvec.selection") {
    import hiveContext.implicits._

    // see also hivemall.ftvec.selection.SignalNoiseRatioUDAFTest
    // binary class
    // +-----------------+-------+
    // |     features    | class |
    // +-----------------+-------+
    // | 5.1,3.5,1.4,0.2 |     0 |
    // | 4.9,3.0,1.4,0.2 |     0 |
    // | 4.7,3.2,1.3,0.2 |     0 |
    // | 7.0,3.2,4.7,1.4 |     1 |
    // | 6.4,3.2,4.5,1.5 |     1 |
    // | 6.9,3.1,4.9,1.5 |     1 |
    // +-----------------+-------+
    val df0 = Seq(
      (1, Seq(5.1, 3.5, 1.4, 0.2), Seq(1, 0)), (1, Seq(4.9, 3.0, 1.4, 0.2), Seq(1, 0)),
      (1, Seq(4.7, 3.2, 1.3, 0.2), Seq(1, 0)), (1, Seq(7.0, 3.2, 4.7, 1.4), Seq(0, 1)),
      (1, Seq(6.4, 3.2, 4.5, 1.5), Seq(0, 1)), (1, Seq(6.9, 3.1, 4.9, 1.5), Seq(0, 1)))
      .toDF("c0", "arg0", "arg1")
    val row0 = df0.groupBy($"c0").snr("arg0", "arg1").collect
    (row0(0).getAs[Seq[Double]](1), Seq(4.38425236, 0.26390002, 15.83984511, 26.87005769))
      .zipped
      .foreach((actual, expected) => assert(actual ~== expected))

    // multiple class
    // +-----------------+-------+
    // |     features    | class |
    // +-----------------+-------+
    // | 5.1,3.5,1.4,0.2 |     0 |
    // | 4.9,3.0,1.4,0.2 |     0 |
    // | 7.0,3.2,4.7,1.4 |     1 |
    // | 6.4,3.2,4.5,1.5 |     1 |
    // | 6.3,3.3,6.0,2.5 |     2 |
    // | 5.8,2.7,5.1,1.9 |     2 |
    // +-----------------+-------+
    val df1 = Seq(
      (1, Seq(5.1, 3.5, 1.4, 0.2), Seq(1, 0, 0)), (1, Seq(4.9, 3.0, 1.4, 0.2), Seq(1, 0, 0)),
      (1, Seq(7.0, 3.2, 4.7, 1.4), Seq(0, 1, 0)), (1, Seq(6.4, 3.2, 4.5, 1.5), Seq(0, 1, 0)),
      (1, Seq(6.3, 3.3, 6.0, 2.5), Seq(0, 0, 1)), (1, Seq(5.8, 2.7, 5.1, 1.9), Seq(0, 0, 1)))
      .toDF("c0", "arg0", "arg1")
    val row1 = df1.groupBy($"c0").snr("arg0", "arg1").collect
    (row1(0).getAs[Seq[Double]](1), Seq(8.43181818, 1.32121212, 42.94949495, 33.80952381))
      .zipped
      .foreach((actual, expected) => assert(actual ~== expected))
  }

  test("aggregations for tools.matrix") {
    import hiveContext.implicits._

    // | 1  2  3 |T    | 5  6  7 |
    // | 3  4  5 |  *  | 7  8  9 |
    val df0 = Seq((1, Seq(1, 2, 3), Seq(5, 6, 7)), (1, Seq(3, 4, 5), Seq(7, 8, 9)))
      .toDF("c0", "arg0", "arg1")

    checkAnswer(df0.groupBy($"c0").transpose_and_dot("arg0", "arg1"),
      Seq(Row(1, Seq(Seq(26.0, 30.0, 34.0), Seq(38.0, 44.0, 50.0), Seq(50.0, 58.0, 66.0)))))
  }
}

final class HivemallOpsWithVectorSuite extends VectorQueryTest {
  import hiveContext.implicits._

  test("to_hivemall_features") {
    checkAnswer(
      mllibTrainDf.select(to_hivemall_features($"features")),
      Seq(
        Row(Seq("0:1.0", "2:2.0", "4:3.0")),
        Row(Seq("0:1.0", "3:1.5", "4:2.1", "6:1.2")),
        Row(Seq("0:1.1", "3:1.0", "4:2.3", "6:1.0")),
        Row(Seq("1:4.0", "3:5.0", "5:6.0"))
      )
    )
  }

  ignore("append_bias") {
    /**
     * TODO: This test throws an exception:
     * Failed to analyze query: org.apache.spark.sql.AnalysisException: cannot resolve
     *   'UDF(UDF(features))' due to data type mismatch: argument 1 requires vector type,
     *    however, 'UDF(features)' is of vector type.; line 2 pos 8
     */
    checkAnswer(
      mllibTrainDf.select(to_hivemall_features(append_bias($"features"))),
      Seq(
        Row(Seq("0:1.0", "0:1.0", "2:2.0", "4:3.0")),
        Row(Seq("0:1.0", "0:1.0", "3:1.5", "4:2.1", "6:1.2")),
        Row(Seq("0:1.0", "0:1.1", "3:1.0", "4:2.3", "6:1.0")),
        Row(Seq("0:1.0", "1:4.0", "3:5.0", "5:6.0"))
      )
    )
  }

  test("explode_vector") {
    checkAnswer(
      mllibTrainDf.explode_vector($"features").select($"feature", $"weight"),
      Seq(
        Row("0", 1.0), Row("0", 1.0), Row("0", 1.1),
        Row("1", 4.0),
        Row("2", 2.0),
        Row("3", 1.0), Row("3", 1.5), Row("3", 5.0),
        Row("4", 2.1), Row("4", 2.3), Row("4", 3.0),
        Row("5", 6.0),
        Row("6", 1.0), Row("6", 1.2)
      )
    )
  }

  test("train_logistic_regr") {
    checkAnswer(
      mllibTrainDf.train_logistic_regr($"features", $"label")
        .groupBy("feature").agg("weight" -> "avg")
        .select($"feature"),
      Seq(0, 1, 2, 3, 4, 5, 6).map(v => Row(s"$v"))
    )
  }
}
