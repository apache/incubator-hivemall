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

<!-- toc -->

# What is "prediction problem"?

In a context of machine learning, numerous tasks can be seen as **prediction problem**. For example, this user guide provides solutions for:

- [spam detection](../binaryclass/webspam.md)
- [news article classification](../multiclass/news20.md)
- [click-through-rate estimation](../regression/kddcup12tr2.md)

For any kinds of prediction problems, we generally provide a set of input-output pairs as:

- **Input:** Set of features
	- e.g., `["1:0.001","4:0.23","35:0.0035",...]`
- **Output:** Target value
	- e.g., 1, 0, 0.54, 42.195, ...
	
Once a prediction model has been constructed based on the samples, the model can make prediction for unforeseen inputs. 

In order to train prediction models, an algorithm so-called ***stochastic gradient descent*** (SGD) is normally applied. You can learn more about this from the following external resources:

- [scikit-learn documentation](http://scikit-learn.org/stable/modules/sgd.html)
- [Spark MLlib documentation](http://spark.apache.org/docs/latest/mllib-optimization.html)

Importantly, depending on types of output value, prediction problem can be categorized into **regression** and **classification** problem.

# Regression

The goal of regression is to predict **real values** as shown below:

| features (input) | target real value (output) |
|:---|:---:|
|["1:0.001","4:0.23","35:0.0035",...] | 21.3 |
|["1:0.2","3:0.1","13:0.005",...] | 6.2 |
|["5:1.3","22:0.0.089","77:0.0001",...] | 17.1 |
| ... | ... |

In practice, target values could be any of small/large float/int negative/positive values. [Our CTR prediction tutorial](../regression/kddcup12tr2.md) solves regression problem with small floating point target values in a 0-1 range, for example.

While there are several ways to realize regression by using Hivemall, `train_regression()` is one of the most flexible functions. This feature is explained in: [Regression](../regression/general.md).

# Classification

In contrast to regression, output for classification problems should be (integer) **labels**:

| features (input) | label (output) |
|:---|:---:|
|["1:0.001","4:0.23","35:0.0035",...] | 0 |
|["1:0.2","3:0.1","13:0.005",...] | 1 |
|["5:1.3","22:0.0.089","77:0.0001",...] | 1 |
| ... | ... |

In case the number of possible labels is 2 (0/1 or -1/1), the problem is **binary classification**, and Hivemall's `train_classifier()` function enables you to build binary classifiers. [Binary Classification](../binaryclass/general.md) demonstrates how to use the function.

Another type of classification problems is **multi-class classification**. This task assumes that the number of possible labels is more than 2. We need to use different functions for the multi-class problems, and our [news20](../multiclass/news20.md) and [iris](../multiclass/iris.md) tutorials would be helpful.

# Mathematical formulation of generic prediction model

Here, we briefly explain about how prediction model is constructed.

First and foremost, we represent **input** and **output** for prediction models as follows:

- **Input:** a vector $$\mathbf{x}$$
- **Output:** a value $$y$$

For a set of samples $$(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \cdots, (\mathbf{x}_n, y_n)$$, the goal of prediction algorithms is to find a weight vector (i.e., parameters) $$\mathbf{w}$$ by minimizing the following error:

$$
E(\mathbf{w}) := \frac{1}{n} \sum_{i=1}^{n} L(\mathbf{w}; \mathbf{x}_i, y_i) + \lambda R(\mathbf{w})
$$

In the above formulation, there are two auxiliary functions we have to know: 

- $$L(\mathbf{w}; \mathbf{x}_i, y_i)$$
	- **Loss function** for a single sample $$(\mathbf{x}_i, y_i)$$ and given $$\mathbf{w}$$.
	- If this function produces small values, it means the parameter $$\mathbf{w}$$ is successfully learnt. 
- $$R(\mathbf{w})$$
	- **Regularization function** for the current parameter $$\mathbf{w}$$.
	- It prevents failing to a negative condition so-called **over-fitting**.
	
($$\lambda$$ is a small value which controls the effect of regularization function.)

Eventually, minimizing the function $$E(\mathbf{w})$$ can be implemented by the SGD technique as described before, and $$\mathbf{w}$$ itself is used as a "model" for future prediction.

Interestingly, depending on a choice of loss and regularization function, prediction model you obtained will behave differently; even if one combination could work as a classifier, another choice might be appropriate for regression.

Below we list possible options for `train_regression` and `train_classifier`, and this is the reason why these two functions are the most flexible in Hivemall:

- Loss function: `-loss`, `-loss_function`
	- For `train_regression`
		- SquaredLoss (synonym: squared)
		- QuantileLoss (synonym: quantile)
		- EpsilonInsensitiveLoss (synonym: epsilon_intensitive)
		- SquaredEpsilonInsensitiveLoss (synonym: squared_epsilon_intensitive)
		- HuberLoss (synonym: huber)
	- For `train_classifier`
		- HingeLoss (synonym: hinge)
		- LogLoss (synonym: log, logistic)
		- SquaredHingeLoss (synonym: squared_hinge)
		- ModifiedHuberLoss (synonym: modified_huber)
		- The following losses are mainly designed for regression but can sometimes be useful in classification as well:
		  - SquaredLoss (synonym: squared)
		  - QuantileLoss (synonym: quantile)
		  - EpsilonInsensitiveLoss (synonym: epsilon_intensitive)
		  - SquaredEpsilonInsensitiveLoss (synonym: squared_epsilon_intensitive)
		  - HuberLoss (synonym: huber)

- Regularization function: `-reg`, `-regularization`
	- L1
	- L2
	- ElasticNet
	- RDA
	
Additionally, there are several variants of the SGD technique, and it is also configureable as:

- Optimizer `-opt`, `-optimizer`
	- SGD
	- AdaGrad
	- AdaDelta
	- Adam

> #### Note
>
> Option values are case insensitive and you can use `sgd` or `rda`, or `huberloss`.

In practice, you can try different combinations of the options in order to achieve higher prediction accuracy.