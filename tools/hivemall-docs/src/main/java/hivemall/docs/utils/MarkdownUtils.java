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
package hivemall.docs.utils;

public class MarkdownUtils {
    private static final String TAB = "  ";

    public static String indent(final String s) {
        if (s.isEmpty()) {
            return s;
        }
        return TAB + s.replaceAll("(\\r\\n|\\r|\\n)(.+)", "$1" + TAB + "$2");
    }

    public static String asBold(final String s) {
        return "**" + s + "**";
    }

    public static String asInlineCode(final String s) {
        return "`" + s + "`";
    }

    public static String asListElement(final String s) {
        return "- " + s;
    }

    public static String asCodeBlock(final String s) {
        return asCodeBlock(s, "");
    }

    public static String asCodeBlock(final String s, final String lang) {
        return "```" + lang + "\n" + s + "\n```\n";
    }

    public static String asHeader(final String s, int level) {
        char[] buf = new char[level];
        for (int i = 0; i < level; i++) {
            buf[i] = '#';
        }
        return new String(buf) + " " + s + "\n";
    }
}
