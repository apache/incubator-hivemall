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
package hivemall.tools.text;

import hivemall.utils.lang.StringUtils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.annotation.Nullable;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.hadoop.hive.ql.exec.UDF;
import org.apache.hadoop.hive.ql.udf.UDFType;

// Inspired by
//  https://github.com/sundrio/sundrio/blob/95c2b11f7b842bdaa04f61e8e338aea60fb38f70/codegen/src/main/java/io/sundr/codegen/functions/Singularize.java
//  https://github.com/clips/pattern/blob/3eef00481a4555331cf9a099308910d977f6fc22/pattern/text/en/inflect.py#L445-L623
@Description(name = "singularize",
        value = "_FUNC_(string word) - Returns singular form of a given English word")
@UDFType(deterministic = true, stateful = false)
public final class SingularizeUDF extends UDF {

    // sorted by an ascending (i.e., alphabetical) order for binary search
    // plural preposition to detect compound words like "plural-preposition-something"
    private static final String[] prepositions = new String[] {"about", "above", "across", "after",
            "among", "around", "at", "athwart", "before", "behind", "below", "beneath", "beside",
            "besides", "between", "betwixt", "beyond", "but", "by", "during", "except", "for",
            "from", "in", "into", "near", "of", "off", "on", "onto", "out", "over", "since",
            "till", "to", "under", "until", "unto", "upon", "with"};
    // uninfected or uncountable words
    private static final String[] unchanged = new String[] {"advice", "bison", "bread", "bream",
            "breeches", "britches", "butter", "carp", "chassis", "cheese", "christmas", "clippers",
            "cod", "contretemps", "corps", "debris", "diabetes", "djinn", "eland", "electricity",
            "elk", "equipment", "flounder", "fruit", "furniture", "gallows", "garbage", "georgia",
            "graffiti", "gravel", "happiness", "headquarters", "herpes", "high-jinks", "homework",
            "information", "innings", "jackanapes", "ketchup", "knowledge", "love", "luggage",
            "mackerel", "mathematics", "mayonnaise", "measles", "meat", "mews", "mumps", "mustard",
            "news", "news", "pincers", "pliers", "proceedings", "progress", "rabies", "research",
            "rice", "salmon", "sand", "scissors", "series", "shears", "software", "species",
            "swine", "swiss", "trout", "tuna", "understanding", "water", "whiting", "wildebeest"};

    private static final Map<String, String> irregular = new HashMap<String, String>();
    static {
        irregular.put("atlantes", "atlas");
        irregular.put("atlases", "atlas");
        irregular.put("axes", "axe");
        irregular.put("beeves", "beef");
        irregular.put("brethren", "brother");
        irregular.put("children", "child");
        irregular.put("corpora", "corpus");
        irregular.put("corpuses", "corpus");
        irregular.put("ephemerides", "ephemeris");
        irregular.put("feet", "foot");
        irregular.put("ganglia", "ganglion");
        irregular.put("geese", "goose");
        irregular.put("genera", "genus");
        irregular.put("genii", "genie");
        irregular.put("graffiti", "graffito");
        irregular.put("helves", "helve");
        irregular.put("kine", "cow");
        irregular.put("leaves", "leaf");
        irregular.put("loaves", "loaf");
        irregular.put("men", "man");
        irregular.put("mongooses", "mongoose");
        irregular.put("monies", "money");
        irregular.put("moves", "move");
        irregular.put("mythoi", "mythos");
        irregular.put("numena", "numen");
        irregular.put("occipita", "occiput");
        irregular.put("octopodes", "octopus");
        irregular.put("opera", "opus");
        irregular.put("opuses", "opus");
        irregular.put("our", "my");
        irregular.put("oxen", "ox");
        irregular.put("penes", "penis");
        irregular.put("penises", "penis");
        irregular.put("people", "person");
        irregular.put("sexes", "sex");
        irregular.put("soliloquies", "soliloquy");
        irregular.put("teeth", "tooth");
        irregular.put("testes", "testis");
        irregular.put("trilbys", "trilby");
        irregular.put("turves", "turf");
        irregular.put("zoa", "zoon");
    }

    private static final List<String> rules = Arrays.asList(
        // regexp1, replacement1, regexp2, replacement2, ...
        "(quiz)zes$", "$1", "(matr)ices$", "$1ix", "(vert|ind)ices$", "$1ex", "^(ox)en", "$1",
        "(alias|status)$", "$1", "(alias|status)es$", "$1", "(octop|vir)us$", "$1us",
        "(octop|vir)i$", "$1us", "(cris|ax|test)es$", "$1is", "(cris|ax|test)is$", "$1is",
        "(shoe)s$", "$1", "(o)es$", "$1", "(bus)es$", "$1", "([m|l])ice$", "$1ouse",
        "(x|ch|ss|sh)es$", "$1", "(m)ovies$", "$1ovie", "(s)eries$", "$1eries",
        "([^aeiouy]|qu)ies$", "$1y", "([lr])ves$", "$1f", "(tive)s$", "$1", "(hive)s$", "$1",
        "([^f])ves$", "$1fe", "(^analy)sis$", "$1sis", "(^analy)ses$", "$1sis",
        "((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$", "$1$2sis", "([ti])a$",
        "$1um", "(n)ews$", "$1ews", "(s|si|u)s$", "$1s", "s$", "");

    @Nullable
    public String evaluate(@Nullable String word) {
        return singularize(word);
    }

    @Nullable
    private static String singularize(@Nullable final String word) {
        if (word == null) {
            return null;
        }

        if (word.isEmpty()) {
            return word;
        }

        if (Arrays.binarySearch(unchanged, word) >= 0) {
            return word;
        }

        if (word.contains("-")) { // compound words (e.g., mothers-in-law)
            final List<String> chunks = new ArrayList<>();
            Collections.addAll(chunks, word.split("-"));
            if ((chunks.size() > 1) && (Arrays.binarySearch(prepositions, chunks.get(1)) >= 0)) {
                String head = chunks.remove(0);
                return singularize(head) + "-" + StringUtils.join(chunks, "-");
            }
        }

        if (word.endsWith("'")) { // dogs' => dog's
            return singularize(word.substring(0, word.length() - 1)) + "'s";
        }

        if (irregular.containsKey(word)) {
            return irregular.get(word);
        }

        for (int i = 0, n = rules.size(); i < n; i += 2) {
            Pattern pattern = Pattern.compile(rules.get(i), Pattern.CASE_INSENSITIVE);
            Matcher matcher = pattern.matcher(word);
            if (matcher.find()) {
                return matcher.replaceAll(rules.get(i + 1));
            }
        }

        return word;
    }

}
