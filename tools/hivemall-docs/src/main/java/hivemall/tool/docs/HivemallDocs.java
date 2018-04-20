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
package hivemall.tool.docs;

import hivemall.tool.docs.utils.MarkdownUtils;
import hivemall.utils.lang.StringUtils;

import org.apache.hadoop.hive.ql.exec.Description;
import org.reflections.Reflections;

import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.commons.lang.StringEscapeUtils.escapeHtml;

public class HivemallDocs {

    public static void main(String... args) {
        Map<String, Set<String>> packages = getHivemallPerPackageDocumentSet();

        for (Map.Entry<String, Set<String>> e : packages.entrySet()) {
            System.out.println(MarkdownUtils.asHeader(e.getKey(), 3));
            for (String desc : e.getValue()) {
                System.out.println(desc);
            }
        }
    }

    private static Map<String, Set<String>> getHivemallPerPackageDocumentSet() {
        Reflections reflections = new Reflections("hivemall");
        Set<Class<?>> annotatedClasses = reflections.getTypesAnnotatedWith(Description.class);

        StringBuilder sb = new StringBuilder();
        Map<String, Set<String>> packages = new TreeMap<>();

        Pattern func = Pattern.compile("_FUNC_(\\(.*?\\))(.*)", Pattern.DOTALL);

        for (Class<?> annotatedClass : annotatedClasses) {
            Deprecated deprecated = annotatedClass.getAnnotation(Deprecated.class);
            if (deprecated != null) {
                continue;
            }

            Description description = annotatedClass.getAnnotation(Description.class);

            String[] values = description.value().split("\n", 2);

            String value = values[0];
            Matcher matcher = func.matcher(value);
            if (matcher.find()) {
                value = MarkdownUtils.asInlineCode(description.name() + matcher.group(1))
                        + escapeHtml(matcher.group(2));
            }
            sb.append(MarkdownUtils.asListElement(value));

            StringBuilder sbExtended = new StringBuilder();
            if (values.length == 2) {
                sbExtended.append(values[1]);
                sb.append("\n");
            }
            if (!description.extended().isEmpty()) {
                sbExtended.append(description.extended());
                sb.append("\n");
            }

            String extended = sbExtended.toString();
            if (!extended.isEmpty()) {
                sb.append(MarkdownUtils.indent(MarkdownUtils.asCodeBlock(extended)));
            } else {
                sb.append("\n");
            }

            String packageName = annotatedClass.getPackage().getName();
            if (!packages.containsKey(packageName)) {
                Set<String> set = new TreeSet<>();
                packages.put(packageName, set);
            }
            Set<String> List = packages.get(packageName);
            List.add(sb.toString());

            StringUtils.clear(sb);
        }

        return packages;
    }
}
