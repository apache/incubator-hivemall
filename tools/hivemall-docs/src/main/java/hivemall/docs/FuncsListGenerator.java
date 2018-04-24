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

import hivemall.docs.utils.MarkdownUtils;
import hivemall.utils.lang.StringUtils;

import org.apache.maven.plugin.AbstractMojo;
import org.apache.maven.execution.MavenSession;
import org.apache.maven.plugin.MojoExecutionException;
import org.apache.maven.plugins.annotations.Mojo;
import org.apache.hadoop.hive.ql.exec.Description;
import org.apache.maven.plugins.annotations.Parameter;
import org.apache.maven.project.MavenProject;
import org.reflections.Reflections;

import javax.annotation.Nonnull;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.LinkedHashMap;
import java.util.TreeSet;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.apache.commons.lang.StringEscapeUtils.escapeHtml;

/**
 * Generate a list of UDFs for documentation.
 *
 * @link https://hivemall.incubator.apache.org/userguide/misc/generic_funcs.html
 * @link https://hivemall.incubator.apache.org/userguide/misc/funcs.html
 */
@Mojo(name = "generate-funcs-list")
public class FuncsListGenerator extends AbstractMojo {

    @Parameter(defaultValue = "${project}", readonly = true)
    private MavenProject project;

    @Parameter(defaultValue = "${session}", readonly = true)
    private MavenSession session;

    @Parameter(defaultValue = "generic_funcs.md")
    private String pathToGenericFuncs;

    @Parameter(defaultValue = "funcs.md")
    private String pathToFuncs;

    private static final Map<String, List<String>> genericFuncsHeaders = new LinkedHashMap<>();
    static {
        genericFuncsHeaders.put("# Generic functions", Arrays.asList("hivemall.tools"));
        genericFuncsHeaders.put("## Array",
            Arrays.asList("hivemall.tools.array", "hivemall.tools.list"));
        genericFuncsHeaders.put("## Map", Arrays.asList("hivemall.tools.map"));
        genericFuncsHeaders.put("## Bitset", Arrays.asList("hivemall.tools.bits"));
        genericFuncsHeaders.put("## Compression", Arrays.asList("hivemall.tools.compress"));
        genericFuncsHeaders.put("## MapReduce", Arrays.asList("hivemall.tools.mapred"));
        genericFuncsHeaders.put("## Math", Arrays.asList("hivemall.tools.math"));
        genericFuncsHeaders.put("## Matrix", Arrays.asList("hivemall.tools.matrix"));
        genericFuncsHeaders.put("## Text processing", Arrays.asList("hivemall.tools.text"));
    }

    private static final Map<String, List<String>> funcsHeaders = new LinkedHashMap<>();
    static {
        funcsHeaders.put("# Regression", Arrays.asList("hivemall.regression"));
        funcsHeaders.put("# Classification", null);
        funcsHeaders.put("## Binary classification", Arrays.asList("hivemall.classifier"));
        funcsHeaders.put("## Multiclass classification",
            Arrays.asList("hivemall.classifier.multiclass"));
        funcsHeaders.put("# Matrix factorization", Arrays.asList("hivemall.mf"));
        funcsHeaders.put("# Factorization machines", Arrays.asList("hivemall.fm"));
        funcsHeaders.put("# Recommendation", Arrays.asList("hivemall.recommend"));
        funcsHeaders.put("# Anomaly detection", Arrays.asList("hivemall.anomaly"));
        funcsHeaders.put("# Topic modeling", Arrays.asList("hivemall.topicmodel"));
        funcsHeaders.put("# Preprocessing", Arrays.asList("hivemall.ftvec"));
        funcsHeaders.put("## Data amplification", Arrays.asList("hivemall.ftvec.amplify"));
        funcsHeaders.put("## Feature binning", Arrays.asList("hivemall.ftvec.binning"));
        funcsHeaders.put("## Feature format conversion", Arrays.asList("hivemall.ftvec.conv"));
        funcsHeaders.put("## Feature hashing", Arrays.asList("hivemall.ftvec.hashing"));
        funcsHeaders.put("## Feature paring", Arrays.asList("hivemall.ftvec.pairing"));
        funcsHeaders.put("## Ranking", Arrays.asList("hivemall.ftvec.ranking"));
        funcsHeaders.put("## Feature scaling", Arrays.asList("hivemall.ftvec.scaling"));
        funcsHeaders.put("## Feature selection", Arrays.asList("hivemall.ftvec.selection"));
        funcsHeaders.put("## Feature transformation and vectorization",
            Arrays.asList("hivemall.ftvec.trans"));
        funcsHeaders.put("# Geospatial functions", Arrays.asList("hivemall.geospatial"));
        funcsHeaders.put("# Distance measures", Arrays.asList("hivemall.knn.distance"));
        funcsHeaders.put("# Locality-sensitive hashing", Arrays.asList("hivemall.knn.lsh"));
        funcsHeaders.put("# Similarity measures", Arrays.asList("hivemall.knn.similarity"));
        funcsHeaders.put("# Evaluation", Arrays.asList("hivemall.evaluation"));
        funcsHeaders.put("# Sketching", Arrays.asList("hivemall.sketch.hll"));
        funcsHeaders.put("# Ensemble learning", Arrays.asList("hivemall.ensemble"));
        funcsHeaders.put("## Bagging", Arrays.asList("hivemall.ensemble.bagging"));
        funcsHeaders.put("# Decision trees and RandomForest", Arrays.asList(
            "hivemall.smile.classification", "hivemall.smile.regression", "hivemall.smile.tools"));
        funcsHeaders.put("# XGBoost", Arrays.asList("hivemall.xgboost.classification",
            "hivemall.xgboost.regression", "hivemall.xgboost.tools"));
        funcsHeaders.put("# Others",
            Arrays.asList("hivemall", "hivemall.dataset", "hivemall.ftvec.text"));
    }

    @Override
    public void execute() throws MojoExecutionException {
        if (!isReactorRootProject()) {
            // output only once across the projects
            return;
        }

        String target = project.getBuild().getDirectory();
        generate(new File(target, pathToGenericFuncs), genericFuncsHeaders);
        generate(new File(target, pathToFuncs), funcsHeaders);
    }

    private boolean isReactorRootProject() {
        return session.getExecutionRootDirectory()
                      .equalsIgnoreCase(project.getBasedir().toString());
    }

    private void generate(@Nonnull File outputFile, @Nonnull Map<String, List<String>> headers)
            throws MojoExecutionException {
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
                value = MarkdownUtils.asInlineCode(description.name() + matcher.group(1))
                        + escapeHtml(matcher.group(2));
            }
            sb.append(MarkdownUtils.asListElement(value));

            StringBuilder sbExtended = new StringBuilder();
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

        PrintWriter writer;
        try {
            writer = new PrintWriter(outputFile);
        } catch (FileNotFoundException e) {
            throw new MojoExecutionException("Output file is not found");
        }

        for (Map.Entry<String, List<String>> e : headers.entrySet()) {
            writer.println(e.getKey() + "\n");
            List<String> packageNames = e.getValue();
            if (packageNames == null) {
                continue;
            }
            for (String packageName : packageNames) {
                for (String desc : packages.get(packageName)) {
                    writer.println(desc);
                }
            }
        }

        writer.close();
    }
}
