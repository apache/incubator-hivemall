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

import hivemall.annotations.VisibleForTesting;
import hivemall.utils.hadoop.HiveUtils;
import hivemall.utils.lang.ArrayUtils;
import hivemall.utils.lang.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDFArgumentException;
import org.apache.hadoop.hive.ql.metadata.HiveException;
import org.apache.hadoop.hive.ql.udf.UDFType;
import org.apache.hadoop.hive.ql.udf.generic.GenericUDF;
import org.apache.hadoop.hive.serde2.objectinspector.ListObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspector;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorFactory;
import org.apache.hadoop.hive.serde2.objectinspector.ObjectInspectorUtils;
import org.apache.hadoop.hive.serde2.objectinspector.primitive.PrimitiveObjectInspectorFactory;

@Description(name = "stoptags_exclude",
        value = "_FUNC_(array<string> excludeTags, [, const string lang='ja']) - Returns stoptags excluding given tags",
        extended = "SELECT stoptags_exclude(array('名詞-固有名詞', '形容詞'))")
@UDFType(deterministic = true, stateful = false)
public final class StoptagsExcludeUDF extends GenericUDF {

    static final String[] STOPTAGS_JA;
    static {
        STOPTAGS_JA = new String[] {"名詞", "名詞-一般", "名詞-固有名詞", "名詞-固有名詞-一般", "名詞-固有名詞-人名",
                "名詞-固有名詞-人名-一般", "名詞-固有名詞-人名-姓", "名詞-固有名詞-人名-名", "名詞-固有名詞-組織", "名詞-固有名詞-地域",
                "名詞-固有名詞-地域-一般", "名詞-固有名詞-地域-国", "名詞-代名詞", "名詞-代名詞-一般", "名詞-代名詞-縮約", "名詞-副詞可能",
                "名詞-サ変接続", "名詞-形容動詞語幹", "名詞-数", "名詞-非自立", "名詞-非自立-一般", "名詞-非自立-副詞可能",
                "名詞-非自立-助動詞語幹", "名詞-非自立-形容動詞語幹", "名詞-特殊", "名詞-特殊-助動詞語幹", "名詞-接尾", "名詞-接尾-一般",
                "名詞-接尾-人名", "名詞-接尾-地域", "名詞-接尾-サ変接続", "名詞-接尾-助動詞語幹", "名詞-接尾-形容動詞語幹", "名詞-接尾-副詞可能",
                "名詞-接尾-助数詞", "名詞-接尾-特殊", "名詞-接続詞的", "名詞-動詞非自立的", "名詞-引用文字列", "名詞-ナイ形容詞語幹", "接頭詞",
                "接頭詞-名詞接続", "接頭詞-動詞接続", "接頭詞-形容詞接続", "接頭詞-数接", "動詞", "動詞-自立", "動詞-非自立", "動詞-接尾",
                "形容詞", "形容詞-自立", "形容詞-非自立", "形容詞-接尾", "副詞", "副詞-一般", "副詞-助詞類接続", "連体詞", "接続詞", "助詞",
                "助詞-格助詞", "助詞-格助詞-一般", "助詞-格助詞-引用", "助詞-格助詞-連語", "助詞-接続助詞", "助詞-係助詞", "助詞-副助詞",
                "助詞-間投助詞", "助詞-並立助詞", "助詞-終助詞", "助詞-副助詞／並立助詞／終助詞", "助詞-連体化", "助詞-副詞化", "助詞-特殊",
                "助動詞", "感動詞", "記号", "記号-一般", "記号-読点", "記号-句点", "記号-空白", "記号-括弧開", "記号-括弧閉",
                "記号-アルファベット", "その他", "その他-間投", "フィラー", "非言語音", "語断片", "未知語"};
        Arrays.sort(STOPTAGS_JA);
    }

    private ListObjectInspector tagsOI;
    private String[] stopTags;

    @Nullable
    private List<String> result;

    @Override
    public ObjectInspector initialize(ObjectInspector[] argOIs) throws UDFArgumentException {
        if (argOIs.length != 1 && argOIs.length != 2) {
            throw new UDFArgumentException(
                "stoptags_exclude(array<string> tags, [, const string lang='ja']) takes one or two arguments: "
                        + argOIs.length);
        }

        if (!HiveUtils.isStringListOI(argOIs[0])) {
            throw new UDFArgumentException(
                "stoptags_exclude(array<string> tags, [, const string lang='ja']) expects array<string> for the first argument : "
                        + argOIs[0].getTypeName());
        }
        this.tagsOI = HiveUtils.asListOI(argOIs[0]);

        if (argOIs.length == 2) {
            if (!HiveUtils.isConstString(argOIs[1])) {
                throw new UDFArgumentException(
                    "stoptags_exclude(array<string> tags, [, const string lang='ja']) expects const string for the second argument: "
                            + argOIs[1].getTypeName());
            }
            String lang = HiveUtils.getConstString(argOIs[1]);
            if (!"ja".equalsIgnoreCase(lang)) {
                throw new UDFArgumentException("Unsupported lang: " + lang);
            }
        }
        this.stopTags = STOPTAGS_JA;

        if (ObjectInspectorUtils.isConstantObjectInspector(tagsOI)) {
            String[] excludeTags = HiveUtils.getConstStringArray(tagsOI);
            this.result = getStoptags(stopTags, excludeTags);
        }

        return ObjectInspectorFactory.getStandardListObjectInspector(
            PrimitiveObjectInspectorFactory.javaStringObjectInspector);
    }

    @Override
    public List<String> evaluate(DeferredObject[] arguments) throws HiveException {
        if (result != null) {
            return result;
        }
        Objects.requireNonNull(stopTags);

        final String[] excludeTags = HiveUtils.asStringArray(arguments[0], tagsOI);
        if (excludeTags == null) {
            return ArrayUtils.asKryoSerializableList(stopTags);
        } else {
            return getStoptags(stopTags, excludeTags);
        }
    }

    @Nonnull
    @VisibleForTesting
    static List<String> getStoptags(@Nonnull final String[] stopTags,
            @Nonnull final String[] excludeTags) {
        final String[] mutableStopTags = stopTags.clone();
        for (String tag : excludeTags) {
            final int index = Arrays.binarySearch(stopTags, tag);
            if (index < 0) {
                continue;
            }
            // found prefix of given tag
            for (int i = index; i < mutableStopTags.length; i++) {
                final String stopTag = mutableStopTags[i];
                if (stopTag == null) {
                    continue;
                }
                if (stopTag.startsWith(tag)) {
                    final int tagLen = tag.length();
                    if (stopTag.length() > tagLen) {
                        final char c = stopTag.charAt(tagLen);
                        if (c != '-') {
                            continue;
                        }
                    }
                    mutableStopTags[i] = null;
                } else {
                    break;
                }
            }
        }
        final List<String> result = new ArrayList<>(mutableStopTags.length);
        for (String tag : mutableStopTags) {
            if (tag != null) {
                result.add(tag);
            }
        }
        return result;
    }

    @Override
    public String getDisplayString(String[] children) {
        return "stoptags_exclude(" + StringUtils.join(children, ',') + ')';
    }

}
