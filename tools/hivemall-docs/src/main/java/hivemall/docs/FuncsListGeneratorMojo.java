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
package hivemall.docs;

import static hivemall.docs.utils.MarkdownUtils.asCodeBlock;
import static hivemall.docs.utils.MarkdownUtils.asInlineCode;
import static hivemall.docs.utils.MarkdownUtils.asListElement;
import static hivemall.docs.utils.MarkdownUtils.indent;
import static org.apache.commons.lang.StringEscapeUtils.escapeHtml;

import hivemall.annotations.Cite;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javax.annotation.Nonnull;

import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.maven.execution.MavenSession;
import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugins.annotations.LifecyclePhase;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.maven.plugins.annotations.Parameter;
import org.apache.maven.plugins.annotations.ResolutionScope;
import org.reflections.Reflections;

/**
 * Generate a list of UDFs for documentation.
 *
 * @link https://hivemall.incubator.apache.org/userguide/misc/generic_funcs.html
 * @link https://hivemall.incubator.apache.org/userguide/misc/funcs.html
 */
@Mojo(name = "generate-funcs-list", defaultPhase = LifecyclePhase.PROCESS_CLASSES,
        requiresDependencyResolution = ResolutionScope.COMPILE_PLUS_RUNTIME,
        configurator = "include-project-dependencies")
public class FuncsListGeneratorMojo extends AbstractMojo {

    @Parameter(defaultValue = "${basedir}", readonly = true)
    private File basedir;

    @Parameter(defaultValue = "${session}", readonly = true)
    private MavenSession session;

    @Parameter(defaultValue = "docs/gitbook/misc/generic_funcs.md")
    private String pathToGenericFuncs;

    @Parameter(defaultValue = "docs/gitbook/misc/funcs.md")
    private String pathToFuncs;

    private static final Map<String, List<String>> genericFuncsHeaders = new LinkedHashMap<>();
    static {
        genericFuncsHeaders.put("# Aggregation", Arrays.asList("hivemall.tools.aggr"));
        genericFuncsHeaders.put("# Array",
            Arrays.asList("hivemall.tools.array", "hivemall.tools.list"));
        genericFuncsHeaders.put("# Bitset", Collections.singletonList("hivemall.tools.bits"));
        genericFuncsHeaders.put("# Compression",
            Collections.singletonList("hivemall.tools.compress"));
        genericFuncsHeaders.put("# Datetime", Collections.singletonList("hivemall.tools.datetime"));
        genericFuncsHeaders.put("# JSON", Collections.singletonList("hivemall.tools.json"));
        genericFuncsHeaders.put("# Map", Collections.singletonList("hivemall.tools.map"));
        genericFuncsHeaders.put("# MapReduce", Collections.singletonList("hivemall.tools.mapred"));
        genericFuncsHeaders.put("# Math", Collections.singletonList("hivemall.tools.math"));
        genericFuncsHeaders.put("# Vector/Matrix",
            Arrays.asList("hivemall.tools.matrix", "hivemall.tools.vector"));
        genericFuncsHeaders.put("# Sanity Checks",
            Collections.singletonList("hivemall.tools.sanity"));
        genericFuncsHeaders.put("# Text processing",
            Arrays.asList("hivemall.tools.text", "hivemall.tools.strings"));
        genericFuncsHeaders.put("# Timeseries",
            Collections.singletonList("hivemall.tools.timeseries"));
        genericFuncsHeaders.put("# Others", Collections.singletonList("hivemall.tools"));
    }

    private static final Map<String, List<String>> funcsHeaders = new LinkedHashMap<>();
    static {
        funcsHeaders.put("# Regression", Collections.singletonList("hivemall.regression"));
        funcsHeaders.put("# Classification", Collections.<String>emptyList());
        funcsHeaders.put("## Binary classification",
            Collections.singletonList("hivemall.classifier"));
        funcsHeaders.put("## Multiclass classification",
            Collections.singletonList("hivemall.classifier.multiclass"));
        funcsHeaders.put("# Matrix factorization",
            Collections.singletonList("hivemall.factorization.mf"));
        funcsHeaders.put("# Factorization machines",
            Collections.singletonList("hivemall.factorization.fm"));
        funcsHeaders.put("# Recommendation", Collections.singletonList("hivemall.recommend"));
        funcsHeaders.put("# Anomaly detection", Collections.singletonList("hivemall.anomaly"));
        funcsHeaders.put("# Topic modeling", Collections.singletonList("hivemall.topicmodel"));
        funcsHeaders.put("# Preprocessing", Collections.singletonList("hivemall.ftvec"));
        funcsHeaders.put("## Data amplification",
            Collections.singletonList("hivemall.ftvec.amplify"));
        funcsHeaders.put("## Feature binning", Collections.singletonList("hivemall.ftvec.binning"));
        funcsHeaders.put("## Feature format conversion",
            Collections.singletonList("hivemall.ftvec.conv"));
        funcsHeaders.put("## Feature hashing", Collections.singletonList("hivemall.ftvec.hashing"));
        funcsHeaders.put("## Feature paring", Collections.singletonList("hivemall.ftvec.pairing"));
        funcsHeaders.put("## Ranking", Collections.singletonList("hivemall.ftvec.ranking"));
        funcsHeaders.put("## Feature scaling", Collections.singletonList("hivemall.ftvec.scaling"));
        funcsHeaders.put("## Feature selection",
            Collections.singletonList("hivemall.ftvec.selection"));
        funcsHeaders.put("## Feature transformation and vectorization",
            Collections.singletonList("hivemall.ftvec.trans"));
        funcsHeaders.put("# Geospatial functions",
            Collections.singletonList("hivemall.geospatial"));
        funcsHeaders.put("# Distance measures", Collections.singletonList("hivemall.knn.distance"));
        funcsHeaders.put("# Locality-sensitive hashing",
            Collections.singletonList("hivemall.knn.lsh"));
        funcsHeaders.put("# Similarity measures",
            Collections.singletonList("hivemall.knn.similarity"));
        funcsHeaders.put("# Evaluation", Collections.singletonList("hivemall.evaluation"));
        funcsHeaders.put("# Sketching",
            Arrays.asList("hivemall.sketch.hll", "hivemall.sketch.bloom"));
        funcsHeaders.put("# Ensemble learning", Collections.singletonList("hivemall.ensemble"));
        funcsHeaders.put("## Bagging", Collections.singletonList("hivemall.ensemble.bagging"));
        funcsHeaders.put("# Decision trees and RandomForest", Arrays.asList(
            "hivemall.smile.classification", "hivemall.smile.regression", "hivemall.smile.tools"));
        funcsHeaders.put("# XGBoost", Arrays.asList("hivemall.xgboost"));
        funcsHeaders.put("# Term Vector Model", Collections.singletonList("hivemall.ftvec.text"));
        funcsHeaders.put("# Others",
            Arrays.asList("hivemall", "hivemall.dataset", "hivemall.ftvec.text"));
    }

    @Override
    public void execute() throws MojoExecutionException {
        if (!isReactorRootProject()) {
            // output only once across the projects
            return;
        }

        generate(new File(basedir, pathToGenericFuncs),
            "This page describes a list of useful Hivemall generic functions. See also a [list of machine-learning-related functions](./funcs.md).",
            genericFuncsHeaders);
        generate(new File(basedir, pathToFuncs),
            "This page describes a list of Hivemall functions. See also a [list of generic Hivemall functions](./generic_funcs.md) for more general-purpose functions such as array and map UDFs.",
            funcsHeaders);
    }

    private boolean isReactorRootProject() {
        return session.getExecutionRootDirectory().equalsIgnoreCase(basedir.toString());
    }

    private void generate(@Nonnull File outputFile, @Nonnull String preface,
            @Nonnull Map<String, List<String>> headers) throws MojoExecutionException {
        Reflections reflections = new Reflections("hivemall");
        Set<Class<?>> annotatedClasses = reflections.getTypesAnnotatedWith(Description.class);

        StringBuilder sb = new StringBuilder();
        Map<String, Set<String>> packages = new HashMap<>();

        Pattern func = Pattern.compile("_FUNC_(\\(.*?\\))(.*)", Pattern.DOTALL);

        for (Class<?> annotatedClass : annotatedClasses) {
            Deprecated deprecated = annotatedClass.getAnnotation(Deprecated.class);
            if (deprecated != null) {
                continue;
            }

            Description description = annotatedClass.getAnnotation(Description.class);

            String value = description.value().replaceAll("\n", " ");
            Matcher matcher = func.matcher(value);
            if (matcher.find()) {
                value = asInlineCode(description.name() + matcher.group(1))
                        + escapeHtml(matcher.group(2));
            }
            sb.append(asListElement(value));

            StringBuilder sbExtended = new StringBuilder();
            if (!description.extended().isEmpty()) {
                sbExtended.append(description.extended());
                sb.append("\n");
            }

            String extended = sbExtended.toString();
            if (extended.isEmpty()) {
                sb.append("\n");
            } else {
                if (extended.toLowerCase().contains("select")) { // extended description contains SQL statements
                    sb.append(indent(asCodeBlock(extended, "sql")));
                } else {
                    sb.append(indent(asCodeBlock(extended)));
                }
            }

            Cite cite = annotatedClass.getAnnotation(Cite.class);
            if (cite != null) {
                sb.append("Reference: ");
                String desc = cite.description();
                String url = cite.url();
                if (url == null) {
                    sb.append(desc);
                } else {
                    sb.append("<a href=\"").append(url).append("\" target=\"_blank\">");
                    sb.append(desc);
                    sb.append("</a>");
                }
                sb.append("<br/>");
            }

            String packageName = annotatedClass.getPackage().getName();
            if (!packages.containsKey(packageName)) {
                Set<String> set = new TreeSet<>();
                packages.put(packageName, set);
            }
            Set<String> List = packages.get(packageName);
            List.add(sb.toString());

            sb.setLength(0);
        }

        try (PrintWriter writer = new PrintWriter(outputFile)) {
            // license header
            writer.println("<!--");
            try {
                File licenseFile = new File(basedir, "resources/license-header.txt");
                FileReader fileReader = new FileReader(licenseFile);

                try (BufferedReader bufferedReader = new BufferedReader(fileReader)) {
                    String line;
                    while ((line = bufferedReader.readLine()) != null) {
                        writer.println(indent(line));
                    }
                }
            } catch (IOException e) {
                throw new MojoExecutionException("Failed to read license file");
            }
            writer.println("-->\n");

            writer.println(preface);

            writer.println("\n<!-- toc -->\n");

            for (Map.Entry<String, List<String>> e : headers.entrySet()) {
                writer.println(e.getKey() + "\n");
                List<String> packageNames = e.getValue();
                for (String packageName : packageNames) {
                    if (!packages.containsKey(packageName)) {
                        writer.close();
                        throw new MojoExecutionException(
                            "Failed to find package in the classpath: " + packageName);
                    }
                    for (String desc : packages.get(packageName)) {
                        writer.println(desc);
                    }
                }
            }

            writer.flush();
        } catch (FileNotFoundException e) {
            throw new MojoExecutionException("Output file is not found");
        }
    }
}
