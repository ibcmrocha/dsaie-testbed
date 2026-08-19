# Linear Basis Function Models

Here we come back to basis function models for regression but now through a Bayesian perspective. To make the discussion easier to follow, we again start at decision theory. We also draw parallels with the {doc}`frequentist treatment<../frequentist/linear_models>` we have seen before.

## Decision theory and observation model

We again have a process $p(\mathbf{x},t)$ we would like to model with a **regression function** $y(\mathbf{x})$. In doing that we incur a **loss** $L$ whose expectation is:

$$
\mathbb{E}[L] = \displaystyle\int\int\left[y(\mathbf{x})-t\right]^2p(\mathbf{x},t)\,\mathrm{d}\mathbf{x}\mathrm{d}t
$$(loss)

And we have seen in {doc}`../frequentist/decision_theory` that the best option for $y(\mathbf{x})$ is:

$$
y(\mathbf{x}) = \mathbb{E}[t\vert\mathbf{x}]
$$(regfunc)

We must therefore come up with an expression for $p(t\vert\mathbf{x})$. To make our lives easier, we opt to model this with a Gaussian. It is an attractive choice because its expectation is simply its mean.

This leads to the **observation model**:

$$
p(t\vert\mathbf{x}) = \mathcal{N}\left(t\vert y(\mathbf{x}),\beta^{-1}\right)
$$(obsmodel)

with $\beta^{-1}$ being a variance parameter. Next we specify the shape of $y(\mathbf{x})$.

```{admonition} Further Reading    
:class: tip    
For an extended discussion on decision theory for regression, read the short Section 1.5.5.
+++                         
{bdg-danger}`bishop-prml`     
```

## Graph model and joint distribution

To give $y(\mathbf{x})$ some shape, we again go for a set of **basis functions** $\boldsymbol{\phi}(\mathbf{x})$ which we assume are known and fixed. To make the model trainable we assume these basis functions are scaled by a set of weights $\mathbf{w}$:

$$
y(\mathbf{x},\mathbf{w}) = \displaystyle\sum_jw_j\phi_j(\mathbf{x}) = \mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x})
$$(ybasisfuncs)

We can pick different basis functions for different applications. Click below for a few examples:

`````{tab-set}
````{tab-item} Polynomials

::::{grid}
:gutter: 2

:::{grid-item}
```{figure} ../figures/linmodels0.svg
```
:::

:::{grid-item}
```{card} Functional form
$\phi_j=x^j\quad$
```

```{card} Additional parameters
None
```

```{card} Use cases
Functions with global support
```
:::

::::
````

````{tab-item} Radial basis

::::{grid}
:gutter: 2

:::{grid-item}
```{figure} ../figures/linmodels1.svg
```
:::

:::{grid-item}
```{card} Functional form
$\phi_j=\exp\left[-\displaystyle\frac{(x-\mu_j)^2}{2s_j^2}\right]\quad$ 
```

```{card} Additional parameters
- $\mu_j$: Center of each basis
- $s_j$: Length scale (area of influence)
```

```{card} Use cases
Functions with localized support, transitions.
```
:::

::::
````

````{tab-item} Sigmoids

::::{grid}
:gutter: 2

:::{grid-item}
```{figure} ../figures/linmodels2.svg
```
:::

:::{grid-item}
```{card} Functional form
$\phi_j = \displaystyle\frac{1}{1+\exp\left(\frac{\mu_j-x}{s_j}\right)}\quad$ 
```

```{card} Additional parameters
- $\mu_j$: Location parameter
- $s_j$: Length scale (area of influence)
```

```{card} Use cases
Functions whose influence saturate over $x$
```
:::

::::
````
`````

where we see that they can be nonlinear in $\mathbf{x}$ while still being linear in $\mathbf{w}$.

With this we can already define a {doc}`graph<../preliminaries/graph_models>` for our model:

```{figure} ../figures/regressiongraph1.svg
:scale: 70%
:name: regressiongraph1

A probabilistic graph for regression.
```

Note that $\mathbf{x}$ is modeled as deterministic here. With this in mind, the joint distribution for this graph is simply:

$$
p(\mathbf{w},t\vert\mathbf{x}) = p(\mathbf{w})p(t\vert\mathbf{w},\mathbf{x})
$$(joint1)

and we already decided to make $p(t\vert\mathbf{w},\mathbf{x})$ a Gaussian. 

But what to do about $p(\mathbf{w})$? At this point we can adopt a **prior distribution** over it. To make matters simple, let us also assume it is Gaussian:

$$
p(\mathbf{w}) = \mathcal{N}\left(\mathbf{w}\vert\boldsymbol{0},\alpha^{-1}\mathbf{I}\right)
$$(prior)

where we are assuming a prior with zero mean and diagonal covariance. Let us add that to our graph:

```{figure} ../figures/regressiongraph2.svg
:scale: 70%
:name: regressiongraph2

A probabilistic graph for regression, now with hyperparameters.
```

and we now also add the $\beta$ parameter from the observation model. Since they govern the shapes of our distributions but not the actual values of $\mathbf{w}$, we say $\alpha$ and $\beta$ are **hyperparameters**.

Our model is in principle ready to be used. But since the above prior is not very informative, we first need to observe some data to get a better idea of what the weights actually are.

## Observing some data

Suppose we now observe $N$ samples of $t$ values and their associated $\mathbf{x}$ values. We gather these observations in a matrix $\mathbf{X}=\left[\mathbf{x}_1,\dots,\mathbf{x}_N\right]$ and a column vector $\mathbf{t}=\left[t_1,\dots,t_N\right]^\mathrm{T}$.

This changes our graph to:

```{figure} ../figures/regressiongraph3.svg
:scale: 70%
:name: regressiongraph3

A probabilistic graph for regression, now with observations.
```

where the *plate* with the $N$ inside means we should imagine these two nodes repeated $N$ times.

Assuming these $N$ observations are made *independently and identically distributed (i.i.d.)* from $p(t\vert\mathbf{w},\mathbf{x})$, we can get a joint distribution for our whole dataset:

$$
p(\mathbf{t}\vert\mathbf{X},\mathbf{w},\beta) = \displaystyle\prod_{n=1}^N\mathcal{N}\left(t_n\vert\mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n),\beta^{-1}\right)
$$(bayesreg-likelihood)

We are now standing at a crossroads. We can either take a shortcut and get a single estimate for $\mathbf{w}$ or we can go fully Bayesian and compute the posterior with:

$$
p(\mathbf{w}\vert\mathbf{t}) =
\displaystyle\frac
{p(\mathbf{t}\vert\mathbf{w})p(\mathbf{w})}
{p(\mathbf{t})}
$$(bayes1)

## Two deterministic shortcuts to $\mathbf{w}$

First we take two possible shortcuts to a quick estimate for the weights. We also discuss why they are not always a good choice.

For the first one we note that Eq. {eq}`bayesreg-likelihood` is a multivariate Gaussian for $\mathbf{t}$ (the product of two or more Gaussians is also a Gaussian). We can therefore go the frequentist way:

````{card}
**Maximum Likelihood Estimation (MLE)**
^^^
Given a likelihood function $p(\mathbf{t}\vert\mathbf{w})$, compute $\mathbf{w}$ that maximizes this likelihood. The prior over $\mathbf{w}$ is ignored and only a single point estimate $\mathbf{w}_\mathrm{MLE}$ remains.
````

Taking the logarithm of Eq. {eq}`bayesreg-likelihood` turns the product into a sum and gets rid of the exponentials inside ({doc}`recall the form of a univariate Gaussian<../preliminaries/probability_distributions>`):

$$
\ln p(\mathbf{t}\vert\mathbf{X},\mathbf{w},\beta)
=
\displaystyle\frac{N}{2}\ln\beta - \frac{N}{2}\ln(2\pi)-{\color{red}\frac{\beta}{2}\sum_{n=1}^{N}\left[t_n-\mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)\right]^2}
$$(loglikelihood)

Note the term in red above. Maximizing Eq. {eq}`loglikelihood` is exactly equivalent to minimizing this term. We therefore get the **least squares** solution from {doc}`frequentist regression<../frequentist/linear_models>`. Recall that this works well when we have a lot of data but tends to overfit to small datasets.

The second shortcut already uses Eq. {eq}`bayes1`, so it is a better compromise, although it is still a single point estimate for the weights:

````{card}
**Maximum A Posteriori (MAP)**
^^^
Given a likelihood and a prior, use Bayes' theorem to compute a posterior $p(\mathbf{w}\vert\mathbf{t})$ but keep only the most likely value of $\mathbf{w}$ as estimate. The prior is taken into account but the uncertainty associated with $\mathbf{w}$ is ignored.
````

To get a MAP estimate from Eq. {eq}`bayes1` it suffices to look at the numerator (the unnormalized posterior). Taking the logarithm and isolating only what depends on $\mathbf{w}$ we have:

$$
\ln p(\mathbf{w}\vert\mathbf{t}) \propto
\displaystyle -\frac{\beta}{2}\sum_{n=1}^{N}\left[t_n-\mathbf{w}^\mathrm{T}\boldsymbol{\phi}(\mathbf{x}_n)\right]^2
{\color{red}- \frac{\alpha}{2}\mathbf{w}^\mathrm{T}\mathbf{w}}
$$(bayesreg-map)

Getting the most likely value for $\mathbf{w}$ means maximizing the expression above. Now compare this to the expression for {doc}`regularized least squares<../frequentist/ridge_sgd>` you have seen before, especially the term in red. The MAP solution is exactly equivalent to least squares with $L_2$ regularization!

These results are quite neat, as they unify all the regression approaches we have seen thus far.

```{admonition} Further Reading    
:class: tip    
Make sure you understand how MLE, MAP and the Bayesian posterior for $\mathbf{w}$ are related to each other. Reading Section 3.2 and the first two pages of Section 3.3 will give you a different flavor of this same discussion.
+++                         
{bdg-danger}`bishop-prml`     
```

## The Bayesian way

Finally, we can get a proper probability density for $p(\mathbf{w}\vert\mathbf{t})$ by using the Bayes' theorem in Eq. {eq}`bayes1` consistently. Note that the likelihood $p(\mathbf{t}\vert\mathbf{w})$ is a conditional Gaussian, $p(\mathbf{w})$ is a marginal Gaussian, and the joint distribution (the numerator) is also Gaussian. This means we can directly use {ref}`the standard expressions we derived before<bayes-stdexpressions>` to easily derive a posterior:

$$
p(\mathbf{w}\vert\mathbf{t}) = \mathcal{N}\left(\mathbf{w}\vert\mathbf{m},\mathbf{S}\right)
$$(posterior)

where the mean of the posterior is:

$$
\mathbf{m} = \beta\mathbf{S}\boldsymbol{\Phi}^\mathrm{T}\mathbf{t}
$$(postmean)

and its covariance:

$$
\mathbf{S}^{-1} = \alpha\mathbf{I} + \beta\boldsymbol{\Phi}^\mathrm{T}\boldsymbol{\Phi}
$$(postvar)

where $\Phi_{ij} = \phi_j(x_i)$ already appeared {doc}`in the MLE treatment<../frequentist/linear_models>`.

## Making new predictions

We are almost done! We now have a more informed distribution $p(\mathbf{w}\vert\mathbf{t})$. All that remains now is to finally make new predictions.

Suppose we now have a new input value $\hat{\mathbf{x}}$ and we would like to predict $\hat{t}$. We make one final change to our graph:

```{figure} ../figures/regressiongraph4.svg
:scale: 70%
:name: regressiongraph4

A probabilistic graph for regression, final version.
```

The joint distribution for this final graph is:

$$
p(\hat{t},\mathbf{t},\mathbf{w}\vert\alpha,\beta,\mathbf{X},\hat{\mathbf{x}})
=
p(\mathbf{w}\vert\alpha)p(\mathbf{t}\vert\mathbf{w},\mathbf{X},\beta)p(\hat{t}\vert\hat{\mathbf{x}},\mathbf{w},\beta)
$$(finaljoint)

Conditioning on $\mathbf{t}$ makes the posterior over $\mathbf{w}$ appear:

$$
p(\hat{t},\mathbf{w}\vert\mathbf{t},\alpha,\beta,\hat{\mathbf{x}}) = p(\mathbf{w}\vert\mathbf{t},\alpha,\beta)p(\hat{t}\vert\mathbf{w},\hat{\mathbf{x}},\beta)
$$(finaljoint2)

And finally we can get the distribution over $\hat{t}$ by marginalizing $\mathbf{w}$ out:

$$
p(\hat{t}\vert\hat{\mathbf{x}},\beta) =
\displaystyle\int
p(\mathbf{w}\vert\mathbf{t},\alpha,\beta)p(\hat{t}\vert\mathbf{w},\hat{\mathbf{x}},\beta)
\,\mathrm{d}\mathbf{w}
$$(finalmarginalization)

Considering for a moment only $\mathbf{w}$ and $\hat{t}$, we now have a joint Gaussian with a marginal $p(\mathbf{w}\vert\mathbf{t})$ and a conditional $p(\hat{t}\vert\mathbf{w})$ and we want the marginal $p(\hat{t})$. Using again the {ref}`standard expressions<bayes-stdexpressions>`, we get:

$$
p(\hat{t}\vert\hat{\mathbf{x}},\beta) =
\displaystyle\mathcal{N}\left(
\hat{t}\vert
\mathbf{m}^\mathrm{T}\boldsymbol{\phi}(\hat{\mathbf{x}}),
\frac{1}{\beta} + \boldsymbol{\phi}(\hat{\mathbf{x}})^\mathrm{T}\mathbf{S}\boldsymbol{\phi}(\hat{\mathbf{x}})
\right)
$$(bayesreg-finalmarginal2)

where $\mathbf{m}$ and $\mathbf{S}$ come directly from Eqs. {eq}`postmean` and {eq}`postvar`.

```{admonition} Further Reading    
:class: tip    
Read Section 3.3.2 for more details on this predictive distribution. Study what the two terms of the predictive variance represent, and what happens to them in Figure 3.8.
+++                         
{bdg-danger}`bishop-prml`     
``` 

Click below to see three different radial basis function models fitted to the same dataset using the three strategies we discussed.

`````{tab-set}
````{tab-item} Maximum Likelihood

::::{grid}
:gutter: 2

:::{grid-item}
```{figure} ../figures/linmodels3.svg
```
:::

:::{grid-item}
```{card} Model
- Unregularized least squares fit
- No uncertainty information
```
```{card} Prediction quality
- Overfits noise in data
- Response explodes in extrapolation
```
:::

::::
````

````{tab-item} Maximum A Posteriori

::::{grid}
:gutter: 2

:::{grid-item}
```{figure} ../figures/linmodels4.svg
```
:::

:::{grid-item}
```{card} Model
- Regularized ($L_2$) least squares fit
- No uncertainty information
```
```{card} Prediction quality
- Regularization prevents overfitting
- Behaves well away from observations
```
:::

::::
````

````{tab-item} Bayesian model

::::{grid}
:gutter: 2

:::{grid-item}
```{figure} ../figures/linmodels5.svg
```
:::

:::{grid-item}
```{card} Model
- Consistent Bayesian fit
- Provides uncertainty information
```
```{card} Prediction quality
- Prior distribution prevents overfitting
- Variance increases away from data
```
:::

::::
````
`````

Use the interactive comparison below to see how the prior precision $\alpha$ and
noise precision $\beta$ affect all five polynomial weights, the MAP estimate,
and the full Bayesian predictive distribution. Resample the five observations
to see which conclusions persist across datasets.

<iframe src="../_static/bayesian-linear-models.html" style="width: 100%; height: 720px; border: 0;" loading="lazy" title="Interactive comparison of MLE, MAP and Bayesian polynomial regression"></iframe>

```{admonition} Exam questions    
:class: danger    
For the exam it is crucial that you understand how the Bayesian version of the regression problem arises and how it compares with MLE (and what you have seen in MUDE) and MAP solutions. You might also see formulation questions related to these concepts. All necessary mathematical expressions will be given in the exam, so there is no need to memorize them.
+++    
{bdg-primary}`written-exam`    
``` 
