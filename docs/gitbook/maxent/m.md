<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

Maximum entropy modeling is a framework for integrating information from many heterogeneous information sources for classification.  The data for a  classification problem is described as a (potentially large) number of features.  These features can be quite complex and allow the experimenter to make use of prior knowledge about what types of informations are expected to be important for classification. Each feature corresponds to a constraint on the model.  We then compute the maximum entropy model, the model with the maximum entropy of all the models that satisfy the constraints.  This term may seem perverse, since we have spent most of the book trying to minimize the (cross) entropy of models, but the idea is that we do not want to go beyond the data.  If we chose a model with less entropy, we would add **information** constraints to the model that are not justified by the empirical evidence available to us. Choosing the maximum entropy model is motivated by the desire to preserve as much uncertainty as possible. 

Papers introduce the method are as follows:

- Adwait Ratnaparkhi [Dissertation](ftp://ftp.cis.upenn.edu/pub/ircs/tr/98-15/98-15.ps.gz).
- M. Mohri, et al. [Efficient Large Scale Distributed Training of Conditional Maximum Entropy Models](http://www.cs.nyu.edu/~mohri/pub/maxent.pdf). NIPS 2009.

Hivemall implements a large scale distributed Maximum Entropy Model. The model is implemented in exactly the same way as [OpenNLP MaxEnt Model](http://maxent.sourceforge.net/about.html). That allows to run OpenNLP model on smaller dataset on local machine without Hadoop.

<!-- toc -->

> #### Note
> This feature is supported from Hivemall v0.5-rc.1 or later.

