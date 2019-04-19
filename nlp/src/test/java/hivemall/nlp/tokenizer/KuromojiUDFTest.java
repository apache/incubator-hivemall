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
package hivemall.nlp.tokenizer;

import hivemall.TestUtils;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.PrivilegedAccessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF.DeferredObject;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.typeinfo.PrimitiveTypeInfo;
import org.apache.hadoop.io.Text;
import org.apache.lucene.analysis.ja.JapaneseTokenizer.Mode;
import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

public class KuromojiUDFTest {

    @Test
    public void testOneArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testTwoArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        udf.initialize(argOIs);
        udf.close();
    }

    public void testExpectedMode() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("normal"));
        udf.initialize(argOIs);
        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testInvalidMode() throws IOException, HiveException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("unsupported mode"));
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。テスト。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        udf.evaluate(args);

        udf.close();
    }

    @Test
    public void testThreeArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[3];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testFourArgument() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[4];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testFiveArgumentArray() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // userDictUrl
        argOIs[4] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testFiveArgumenString() throws UDFArgumentException, IOException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector, null);
        // userDictUrl
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        udf.initialize(argOIs);
        udf.close();
    }

    @Test
    public void testEvaluateOneRow() throws IOException, HiveException {
        KuromojiUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。テスト。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        @SuppressWarnings("unchecked")
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(5, tokens.size());
        udf.close();
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testEvaluateTwoRows() throws IOException, HiveException {
        KuromojiUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。テスト。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(5, tokens.size());

        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(4, tokens.size());

        udf.close();
    }

    @Test
    public void testEvaluateLongRow() throws IOException, HiveException {
        KuromojiUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text(
                    "商品の購入・詳細(サイズ、画像)は商品名をクリックしてください！[L.B　CANDY　STOCK]フラワービジューベアドレス[L.B　DAILY　STOCK]ボーダーニットトップス［L.B　DAILY　STOCK］ボーダーロングニットOP［L.B　DAILY　STOCK］ロゴトートBAG［L.B　DAILY　STOCK］裏毛ロゴプリントプルオーバー【TVドラマ着用】アンゴラワッフルカーディガン【TVドラマ着用】グラフィティーバックリボンワンピース【TVドラマ着用】ボーダーハイネックトップス【TVドラマ着用】レオパードミッドカーフスカート【セットアップ対応商品】起毛ニットスカート【セットアップ対応商品】起毛ニットプルオーバー2wayサングラス33ナンバーリングニット3Dショルダーフレアードレス3周年スリッパ3周年ラグマット3周年ロックグラスキャンドルLily　Brown　2015年　福袋MIXニットプルオーバーPeckhamロゴニットアンゴラジャガードプルオーバーアンゴラタートルアンゴラチュニックアンゴラニットカーディガンアンゴラニットプルオーバーアンゴラフレアワンピースアンゴラロングカーディガンアンゴラワッフルカーディガンヴィンテージファー付コートヴィンテージボーダーニットヴィンテージレースハイネックトップスヴィンテージレースブラウスウエストシースルーボーダーワンピースオーガンジーラインフレアスカートオープンショルダーニットトップスオフショルシャーリングワンピースオフショルニットオフショルニットプルオーバーオフショルボーダーロンパースオフショルワイドコンビネゾンオルテガ柄ニットプルオーバーカシュクールオフショルワンピースカットアシンメトリードレスカットサテンプリーツフレアースカートカラースーパーハイウェストスキニーカラーブロックドレスカラーブロックニットチュニックギャザーフレアスカートキラキラストライプタイトスカートキラキラストライプドレスキルティングファーコートグラデーションベアドレスグラデーションラウンドサングラスグラフティーオフショルトップスグラフティーキュロットグリッターリボンヘアゴムクロップドブラウスケーブルハイウエストスカートコーデュロイ×スエードパネルスカートコーデュロイタイトスカートゴールドバックルベルト付スカートゴシックヒールショートブーツゴシック柄ニットワンピコンビスタジャンサイドステッチボーイズデニムパンツサスペつきショートパンツサスペンダー付プリーツロングスカートシャーリングタイトスカートジャガードタックワンピーススエードフリルフラワーパンツスエード裏毛肩空きトップススクエアショルダーBAGスクエアバックルショルダースクエアミニバッグストーンビーチサンダルストライプサスペ付きスキニーストライプバックスリットシャツスライバーシャギーコートタートル×レースタイトスカートタートルニットプルオーバータイトジャンパースカートダブルクロスチュールフレアスカートダブルストラップパンプスダブルハートリングダブルフェイスチェックストールチェーンコンビビジューネックレスチェーンコンビビジューピアスチェーンコンビビジューブレスチェーンツバ広HATチェーンビジューピアスチェックニットプルオーバーチェックネルミディアムスカートチェック柄スキニーパンツチュールコンビアシメトップスデニムフレアースカートドットオフショルフリルブラウスドットジャガードドレスドットニットプルオーバードットレーストップスニット×オーガンジースカートセットニットキャミソールワンピースニットスヌードパールコンビフープピアスハイウエストショートデニムハイウエストタイトスカートハイウエストデニムショートパンツハイウエストプリーツスカートハイウエストミッドカーフスカートハイゲージタートルニットハイゲージラインニットハイネック切り替えスウェットバタフライネックレスバタフライミニピアスバタフライリングバックタンクリブワンピースバックリボンスキニーデニムパンツバックリボン深Vワンピースビジューストラップサンダルビスチェコンビオフショルブラウスブークレジャガードニットフェイクムートンショートコートフェレットカーディガンフェレットビックタートルニットブラウジングクルーブラウスプリーツブラウスフリルニットプルオーバーフリンジニットプルオーバーフレアニットスカートブロウ型サングラスベーシックフェレットプルオーバーベルト付ガウチョパンツベルト付ショートパンツベルト付タックスカートベルト付タックパンツベルベットインヒールパンプスベロアウェッジパンプスベロアミッドカーフワンピースベロアワンピースベロア風ニットカーディガンボア付コートボーダーVネックTシャツボーダーオフショルカットソーボーダーカットソーワンピースボーダータイトカットソーボーダートップスボーダートップス×スカートセットボストンメガネマオカラーシャツニットセットミックスニットプルオーバーミッドカーフ丈ポンチスカートミリタリーギャザーショートパンツメッシュハイネックトップスメルトンPコートメルトンダッフルコートメルトンダブルコートモヘアニットカーディガンモヘアニットタートルユリ柄プリーツフレアースカートライダースデニムジャケットライナー付チェスターコートラッフルプリーツブラウスラメジャガードハイゲージニットリブニットワンピリボン×パールバレッタリボンバレッタリボンベルトハイウエストパンツリリー刺繍開襟ブラウスレースビスチェローファーサボロゴニットキャップロゴ刺繍ニットワッチロングニットガウンワッフルアンゴラプルオーバーワンショルダワーワンピース光沢ラメニットカーディガン刺繍シフォンブラウス台形ミニスカート配色ニットプルオーバー裏毛プルオーバー×オーガンジースカートセット");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        @SuppressWarnings("unchecked")
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);
        Assert.assertEquals(182, tokens.size());
        udf.close();
    }

    @Test
    public void testEvaluateUserDictArray() throws IOException, HiveException {
        KuromojiUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // userDictArray (from https://raw.githubusercontent.com/atilika/kuromoji/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt)
        List<String> userDict = new ArrayList<String>();
        userDict.add("日本経済新聞,日本 経済 新聞,ニホン ケイザイ シンブン,カスタム名詞");
        userDict.add("関西国際空港,関西 国際 空港,カンサイ コクサイ クウコウ,テスト名詞");
        argOIs[4] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, userDict);
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("日本経済新聞。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };

        @SuppressWarnings("unchecked")
        List<Text> tokens = (List<Text>) udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(3, tokens.size());

        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testEvaluateInvalidUserDictURL() throws IOException, HiveException {
        KuromojiUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // userDictUrl
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text("http://google.com/"));
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。テスト。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };

        @SuppressWarnings("unchecked")
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);

        udf.close();
    }

    @Test
    public void testEvaluateUserDictURL() throws IOException, HiveException {
        KuromojiUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[5];
        // line
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        // mode
        PrimitiveTypeInfo stringType = new PrimitiveTypeInfo();
        stringType.setTypeName("string");
        argOIs[1] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, null);
        // stopWords
        argOIs[2] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // stopTags
        argOIs[3] = ObjectInspectorFactory.getStandardConstantListObjectInspector(
            PrimitiveObjectInspectorFactory.writableStringObjectInspector, null);
        // userDictUrl (Kuromoji official sample user defined dict on GitHub)
        // e.g., "日本経済新聞" will be "日本", "経済", and "新聞"
        argOIs[4] = PrimitiveObjectInspectorFactory.getPrimitiveWritableConstantObjectInspector(
            stringType, new Text(
                "https://raw.githubusercontent.com/atilika/kuromoji/909fd6b32bf4e9dc86b7599de5c9b50ca8f004a1/kuromoji-core/src/test/resources/userdict.txt"));
        udf.initialize(argOIs);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。日本経済新聞。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };

        @SuppressWarnings("unchecked")
        List<Text> tokens = (List<Text>) udf.evaluate(args);

        Assert.assertNotNull(tokens);
        Assert.assertEquals(7, tokens.size());

        udf.close();
    }

    @Test
    public void testSerialization() throws IOException, HiveException {
        final KuromojiUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[1];
        argOIs[0] = PrimitiveObjectInspectorFactory.writableStringObjectInspector;
        udf.initialize(argOIs);

        // serialization after initialization
        byte[] serialized = TestUtils.serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, KuromojiUDF.class);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。テスト。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        @SuppressWarnings("unchecked")
        List<Text> tokens = (List<Text>) udf.evaluate(args);
        Assert.assertNotNull(tokens);

        // serialization after evaluation
        serialized = TestUtils.serializeObjectByKryo(udf);
        TestUtils.deserializeObjectByKryo(serialized, KuromojiUDF.class);

        udf.close();
    }

    @Test
    public void testNormalModeWithOption()
            throws IOException, HiveException, IllegalAccessException, NoSuchFieldException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];

        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector; // line
        argOIs[1] = HiveUtils.getConstStringObjectInspector("-mode normal"); // mode
        udf.initialize(argOIs);

        Object mode = PrivilegedAccessor.getValue(udf, "_mode");
        Assert.assertEquals(Mode.NORMAL, mode);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。テスト。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };
        Object result = udf.evaluate(args);
        Assert.assertThat(Arrays.asList(new Text("クロモジ"), new Text("japaneseanalyzer"),
            new Text("使う"), new Text("みる"), new Text("テスト")), CoreMatchers.is(result));

        udf.close();
    }

    @Test
    public void testNormalModeWithPosOptions()
            throws IOException, HiveException, IllegalAccessException, NoSuchFieldException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];

        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector; // line
        argOIs[1] = HiveUtils.getConstStringObjectInspector("-mode normal -pos"); // mode
        udf.initialize(argOIs);

        Object mode = PrivilegedAccessor.getValue(udf, "_mode");
        Assert.assertEquals(Mode.NORMAL, mode);

        DeferredObject[] args = new DeferredObject[1];
        args[0] = new DeferredObject() {
            public Text get() throws HiveException {
                return new Text("クロモジのJapaneseAnalyzerを使ってみる。テスト。");
            }

            @Override
            public void prepare(int arg) throws HiveException {}
        };

        Object[] result = (Object[]) udf.evaluate(args);
        Assert.assertEquals(2, result.length);

        Assert.assertEquals(Arrays.asList(new Text("クロモジ"), new Text("japaneseanalyzer"),
            new Text("使う"), new Text("みる"), new Text("テスト")), result[0]);
        Assert.assertEquals(Arrays.asList(new Text("名詞-一般"), new Text("名詞-一般"), new Text("動詞-自立"),
            new Text("動詞-非自立"), new Text("名詞-サ変接続")), result[1]);

        udf.close();
    }

    @Test(expected = UDFArgumentException.class)
    public void testUnsupportedOptionArgs()
            throws IOException, HiveException, IllegalAccessException, NoSuchFieldException {
        GenericUDF udf = new KuromojiUDF();
        ObjectInspector[] argOIs = new ObjectInspector[2];

        argOIs[0] = PrimitiveObjectInspectorFactory.javaStringObjectInspector; // line
        argOIs[1] = HiveUtils.getConstStringObjectInspector("-mode normal -unsupported_option"); // mode
        udf.initialize(argOIs);

        udf.close();
    }
}
