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

- [scikit-learn documentation](https://scikit-learn.org/stable/modules/sgd.html)
- [Spark MLlib documentation](https://spark.apache.org/docs/latest/mllib-optimization.html)

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

While there are several ways to realize regression by using Hivemall, `train_regressor()` is one of the most flexible functions. This feature is explained in [this page](../regression/general.md).

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

Below we list possible options for `train_regressor` and `train_classifier`, and this is the reason why these two functions are the most flexible in Hivemall:

- Loss function: `-loss`, `-loss_function`
	- For `train_regressor`
		- SquaredLoss (synonym: squared)
		- QuantileLoss (synonym: quantile)
		- EpsilonInsensitiveLoss (synonym: epsilon_insensitive)
		- SquaredEpsilonInsensitiveLoss (synonym: squared_epsilon_insensitive)
		- HuberLoss (synonym: huber)
	- For `train_classifier`
		- HingeLoss (synonym: hinge)
		- LogLoss (synonym: log, logistic)
		- SquaredHingeLoss (synonym: squared_hinge)
		- ModifiedHuberLoss (synonym: modified_huber)
		- The following losses are mainly designed for regression but can sometimes be useful in classification as well:
		  - SquaredLoss (synonym: squared)
		  - QuantileLoss (synonym: quantile)
		  - EpsilonInsensitiveLoss (synonym: epsilon_insensitive)
		  - SquaredEpsilonInsensitiveLoss (synonym: squared\_epsilon_insensitive)
		  - HuberLoss (synonym: huber)

- Regularization function: `-reg`, `-regularization`
	- L1
	- L2
	- ElasticNet
	- RDA
	
Additionally, there are several variants of the SGD technique, and it is also configurable as:

- Optimizer: `-opt`, `-optimizer`
	- SGD
	- Momentum
		- Hyperparameters
			- `-alpha 1.0` Learning rate.
			- `-momentum 0.9` Exponential decay rate of the first order moment.
	- Nesterov
		- See: [https://arxiv.org/abs/1212.0901](https://arxiv.org/abs/1212.0901)
		- Hyperparameters
			- same as Momentum
	- AdaGrad (default)
		- See: [http://jmlr.org/papers/v12/duchi11a.html](http://jmlr.org/papers/v12/duchi11a.html)
		- Hyperparameters
			- `-eps 1.0` Constant for the numerical stability.
	- RMSprop
		- Description: RMSprop optimizer introducing weight decay to AdaGrad.
		- See: [http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
		- Hyperparameters
			- `-decay 0.95` Weight decay rate
			- `-eps 1.0` Constant for numerical stability
	- RMSpropGraves
		- Description: Alex Graves's RMSprop introducing weight decay and momentum.
		- See: [https://arxiv.org/abs/1308.0850](https://arxiv.org/abs/1308.0850)
		- Hyperparameters
			- `-alpha 1.0` Learning rate.
			- `-decay 0.95` Weight decay rate
			- `-momentum 0.9` Exponential decay rate of the first order moment.
			- `-eps 1.0` Constant for numerical stability
	- AdaDelta
		- See: [https://arxiv.org/abs/1212.5701](https://arxiv.org/abs/1212.5701)
		- Hyperparameters
			- `-decay 0.95` Weight decay rate
			- `-eps 1e-6f` Constant for numerical stability
	- Adam
		- See:
			- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
			- [Fixing Weight Decay Regularization in Adam](https://openreview.net/forum?id=rk6qdGgCZ)
			- [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
		- Hyperparameters
			- `-alpha 1.0` Learning rate.
			- `-beta1 0.9` Exponential decay rate of the first order moment.
			- `-beta2 0.999` Exponential decay rate of the second order moment.
			- `-eps 1e-8f` Constant for numerical stability
			- `-decay 0.0` Weight decay rate
	- Nadam
		- Description: Nadam is Adam optimizer with Nesterov momentum.
		- See:
			- [Incorporating Nesterov Momentum into Adam](https://openreview.net/pdf?id=OM0jvwB8jIp57ZJjtNEZ)
			- [Adam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
			- [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
		- Hyperparameters
			- same as Adam except ...
			- `-scheduleDecay 0.004` Scheduled decay rate (for each 250 steps by the default; 1/250=0.004)
	- Eve
		- See: [https://openreview.net/forum?id=r1WUqIceg](https://openreview.net/forum?id=r1WUqIceg)
		- Hyperparameters
			- same as Adam except ...
			- `-beta3 0.999` Decay rate for Eve coefficient.
			- `-c 10` Constant used for gradient clipping `clip(val, 1/c, c)`
	- AdamHD
		- Description: Adam optimizer with Hypergradient Descent. Learning rate `-alpha` is automatically tuned.
		- See:
			- [Online Learning Rate Adaptation with Hypergradient Descent](https://openreview.net/forum?id=BkrsAzWAb)
			- [Convergence Analysis of an Adaptive Method of Gradient Descent](https://damaru2.github.io/convergence_analysis_hypergradient_descent/dissertation_hypergradients.pdf)
		-  Hyperparameters
			- same as Adam except ...
			- `-alpha 0.02` Learning rate.
			- `-beta -1e-6` Constant used for tuning learning rate.

Default (Adagrad+RDA), AdaDelta, Adam, and AdamHD is worth trying in my experience.

> #### Note
>
> Option values are case insensitive and you can use `sgd` or `rda`, or `huberloss` in lower-case letters.

Furthermore, optimizer offers to set auxiliary options such as:

- Number of iterations: `-iter`, `-iterations` [default: 10]
	- Repeat optimizer's learning procedure more than once to diligently find better result.
- Convergence rate: `-cv_rate`, `-convergence_rate` [default: 0.005]
	- Define a stopping criterion for the iterative training.
	- If the criterion is too small or too large, you may encounter over-fitting or under-fitting depending on value of `-iter` option.
- Mini-batch size: `-mini_batch`, `-mini_batch_size` [default: 1]
	- Instead of learning samples one-by-one, this option enables optimizer to utilize multiple samples at once to minimize the error function.
	- Appropriate mini-batch size leads efficient training and effective prediction model.

For details of available options, following queries might be helpful to list all of them:

```sql
select train_regressor('-help');
select train_classifier('-help');
```

```
SELECT train_regressor('-help');

FAILED: UDFArgumentException
train_regressor takes two or three arguments: List<Int|BigInt|Text> features, float target [, constant string options]

usage: train_regressor(list<string|int|bigint> features, double label [,
       const string options]) - Returns a relation consists of
       <string|int|bigint feature, float weight> [-alpha <arg>] [-amsgrad]
       [-beta <arg>] [-beta1 <arg>] [-beta2 <arg>] [-beta3 <arg>] [-c
       <arg>] [-cv_rate <arg>] [-decay] [-dense] [-dims <arg>]
       [-disable_cv] [-disable_halffloat] [-eps <arg>] [-eta <arg>] [-eta0
       <arg>] [-inspect_opts] [-iter <arg>] [-iters <arg>] [-l1_ratio
       <arg>] [-lambda <arg>] [-loss <arg>] [-mini_batch <arg>] [-mix
       <arg>] [-mix_cancel] [-mix_session <arg>] [-mix_threshold <arg>]
       [-opt <arg>] [-power_t <arg>] [-reg <arg>] [-rho <arg>] [-scale
       <arg>] [-ssl] [-t <arg>]
 -alpha <arg>                            Coefficient of learning rate
                                         [default: 1.0
                                         (adam/RMSPropGraves), 0.02
                                         (AdamHD/Nesterov)]
 -amsgrad                                Whether to use AMSGrad variant of
                                         Adam
 -beta <arg>                             Hyperparameter for tuning alpha
                                         in Adam-HD [default: 1e-6f]
 -beta1,--momentum <arg>                 Exponential decay rate of the
                                         first order moment used in Adam
                                         [default: 0.9]
 -beta2 <arg>                            Exponential decay rate of the
                                         second order moment used in Adam
                                         [default: 0.999]
 -beta3 <arg>                            Exponential decay rate of alpha
                                         value  [default: 0.999]
 -c <arg>                                Clipping constant of alpha used
                                         in Eve optimizer so that clipped
[default: 10]
-cv_rate,--convergence_rate <arg>       Threshold to determine
                                         convergence [default: 0.005]
 -decay                                  Weight decay rate [default: 0.0]
 -dense,--densemodel                     Use dense model or not
 -dims,--feature_dimensions <arg>        The dimension of model [default:
                                         16777216 (2^24)]
 -disable_cv,--disable_cvtest            Whether to disable convergence
                                         check [default: OFF]
 -disable_halffloat                      Toggle this option to disable the
                                         use of SpaceEfficientDenseModel
 -eps <arg>                              Denominator value of
                                         AdaDelta/AdaGrad/Adam [default:
                                         1e-8 (AdaDelta/Adam), 1.0
                                         (Adagrad)]
 -eta <arg>                              Learning rate scheme [default:
                                         inverse/inv, fixed, simple]
 -eta0 <arg>                             The initial learning rate
                                         [default: 0.1]
 -inspect_opts                           Inspect Optimizer options
 -iter,--iterations <arg>                The maximum number of iterations
                                         [default: 10]
 -iters,--iterations <arg>               The maximum number of iterations
                                         [default: 10]
 -l1_ratio <arg>                         Ratio of L1 regularizer as a part
                                         of Elastic Net regularization
                                         [default: 0.5]
 -lambda <arg>                           Regularization term [default
                                         0.0001]
 -loss,--loss_function <arg>             Loss function [SquaredLoss
                                         (default), QuantileLoss,
                                         EpsilonInsensitiveLoss,
                                         SquaredEpsilonInsensitiveLoss,
                                         HuberLoss]
 -mini_batch,--mini_batch_size <arg>     Mini batch size [default: 1].
                                         Expecting the value in range
                                         [1,100] or so.
 -mix,--mix_servers <arg>                Comma separated list of MIX
                                         servers
 -mix_cancel,--enable_mix_canceling      Enable mix cancel requests
 -mix_session,--mix_session_name <arg>   Mix session name [default:
                                         ${mapred.job.id}]
 -mix_threshold <arg>                    Threshold to mix local updates in
                                         range (0,127] [default: 3]
 -opt,--optimizer <arg>                  Optimizer to update weights
                                         [default: adagrad, sgd, momentum,
                                         nesterov, rmsprop, rmspropgraves,
                                         adadelta, adam, eve, adam_hd]
 -power_t <arg>                          The exponent for inverse scaling
                                         learning rate [default: 0.1]
 -reg,--regularization <arg>             Regularization type [default:
                                         rda, l1, l2, elasticnet]
 -rho,--decay <arg>                       Exponential decay rate of the
                                         first and second order moments
                                         [default 0.95 (AdaDelta,
                                         rmsprop)]
 -scale <arg>                            Scaling factor for cumulative
                                         weights [100.0]
 -ssl                                    Use SSL for the communication
                                         with mix servers
 -t,--total_steps <arg>                  a total of n_samples * epochs
time steps
```

In practice, you can try different combinations of the options in order to achieve higher prediction accuracy.

You can also find the default optimizer hyperparameters by `-inspect_opts` option as follows:

```sql
select train_regressor(array(), 0, '-inspect_opts -optimizer adam -reg l1');

FAILED: UDFArgumentException Inspected Optimizer options ...
{disable_cvtest=false, regularization=L1, loss_function=SquaredLoss, eps=1.0E-8, decay=0.0, iterations=10, eta0=0.1, lambda=1.0E-4, eta=Invscaling, optimizer=adam, beta1=0.9, beta2=0.999, alpha=1.0, cv_rate=0.005, power_t=0.1}```
