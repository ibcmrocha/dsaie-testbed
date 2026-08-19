# Probability Distributions

There is a plethora of different probability distributions to choose from when describing discrete and continuous random variables. When linking random variables together in {doc}`graph_models`, these distributions become building blocks for more complicated probabilistic models.

In this page we give a very brief overview of the main distributions we will be using throughout the course.

## Bernoulli distribution

We have been working with binary variables (taking only two possible values, true/false, or equivalently zero/one) but have not formally defined a distribution for them. Given a variable $x$ that can take values of <span style="white-space: nowrap;">$0$ or $1$,</span> we can use a parameter $\mu$ to define


$$
\begin{equation}
p(x=1\vert\mu) = \mu
\end{equation}
$$(prelim_bernoulli1)

and obviously $p(x=0\vert\mu)=1-\mu$. We can have a single expression for $p(x\vert\mu)$ through the **Bernoulli distribution**:

$$
p(x\vert\mu) = \mu^x\left(1-\mu\right)^{1-x}
$$(prelim_bernoulli2)

Check for yourself if the distribution agrees with our two values for $x=0$ and $x=1$. Using {doc}`probability_theory`, you can prove that:

$$
\mathbb{E}[x] = \mu
$$(prelim_bernoulli3)

$$
\text{var}[x] = \mu(1-\mu)
$$(prelim_bernoulli4)

(categoricaldist)=
## Categorical distribution

We would now like to extend the binary treatment from the previous section to discrete variables taking one out of $K$ possible values. To make it easy to treat mathematically, we will represent a variable like this using only zeros and ones. This means that in order to represent all possibilities we resort to a vector representation such as:

$$
\mathbf{x} = [0,1,0,0]^\mathrm{T}
$$(prelim_categorical1)

The above represents a variable with $K=4$ possible values and this specific observation of it has the second possible value. For any value of $\mathbf{x}$, the position of the $1$ indicates the value of the variable, while all other positions should be zero.

The probability distribution for this type of variables can be given by a **categorical distribution**:

$$
p(\mathbf{x}\vert\boldsymbol{\mu}) = \displaystyle\prod_{k=1}^{K}\mu_k^{x_k}
$$(prelim_categorical2)

with $\boldsymbol\mu$ being the set of probabilities corresponding to each possible value. Naturally, probability theory also requires the following constraints:

$$
\mu_k \geq 0
$$(prelim_categorical3)

$$
\displaystyle\sum_k\mu_k=1
$$(prelim_categorical4)

```{admonition} Further Reading    
:class: tip    
You can read Sections 2.1 and 2.2 for more details on the Bernoulli and Categorical distributions, including their relationship with the Binomial and Multinomial distributions.  
+++                         
{bdg-danger}`bishop-prml`     
```

## Gaussian distribution

The Gaussian is arguably the most widely used probability distribution. It is also one of the main building blocks of a large number of machine learning models and we must therefore give it special attention.

For a scalar variable $x$, the Gaussian probability density takes the form:

$$
\mathcal{N}(x\vert\mu,\sigma^2) = \displaystyle\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left[-\displaystyle\frac{1}{2\sigma^2}\left(x-\mu\right)^2\right]
$$(prelim_gaussian1)

where we use the special symbol $\mathcal{N}$ to denote a Gaussian. We can see that the distribution is completely determined by a **mean** $\mu$ and a **variance** $\sigma^2$. For a vector $\mathbf{x}$ of $D$ variables we can generalize a **multivariate** Gaussian as:

$$
\mathcal{N}(\mathbf{x}\vert\boldsymbol{\mu},\boldsymbol{\Sigma}) = \displaystyle\frac{1}{(2\pi)^{D/2}}\frac{1}{\sqrt{\vert\boldsymbol{\Sigma}\vert}}\exp\left[-\displaystyle\frac{1}{2}\left(\mathbf{x}-\boldsymbol{\mu}\right)^\mathrm{T}\boldsymbol{\Sigma}^{-1}\left(\mathbf{x}-\boldsymbol{\mu}\right)\right]
$$(prelim_gaussian2)

where the mean is now a vector $\boldsymbol{\mu}$ of size $D$ and the variance is now a matrix $\boldsymbol{\Sigma}$ of size $D\times D$.

The Gaussian is based on a **quadratic form**, so it has ellipsoidal isoprobability contours in $D$-dimensional space that depend on what goes into $\boldsymbol{\Sigma}$. The contents of the covariance matrix also encode how correlated the different variables in the distribution are.

In the following we see some important results for the Gaussian we will keep using throughout the course.

### Conditioning

Consider a multivariate Gaussian for a vector variable $\mathbf{x}$. Imagine we **split** this vector into two groups $\mathbf{x}_a$ and $\mathbf{x}_b$ of variables:

$$
\mathbf{x} = \begin{bmatrix}\mathbf{x}_a\\\mathbf{x}_b\end{bmatrix}
\quad\Rightarrow\quad
p(\mathbf{x}) = \mathcal{N}\left(
\begin{bmatrix}\mathbf{x}_a\\\mathbf{x}_b\end{bmatrix}
\Bigg\vert
\begin{bmatrix}\boldsymbol{\mu}_a\\\boldsymbol{\mu}_b\end{bmatrix}
,
\begin{bmatrix}\boldsymbol{\Sigma}_{aa} & \boldsymbol{\Sigma}_{ab}\\\boldsymbol{\Sigma}_{ba} & \boldsymbol{\Sigma}_{bb}\end{bmatrix}
\right)
$$(prelim_gaussian3)

where we have just split our mean $\boldsymbol{\mu}$ and covariance $\boldsymbol{\Sigma}$ accordingly. 

A nice property of the Gaussian distribution is that the distribution of part of our variables **conditioned** on the other part is still a Gaussian. More specifically, we can arrive at the expressions:

$$
p(\mathbf{x}_a\vert\mathbf{x}_b) = \mathcal{N}\left(\mathbf{x}_a\vert\boldsymbol{\mu}_{a\vert b},\boldsymbol{\Sigma}_{a\vert b}\right)
$$(prelim_gaussian4)

$$
\boldsymbol{\mu}_{a\vert b} = \boldsymbol{\mu}_a + \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\left(\mathbf{x}_b-\boldsymbol{\mu}_b\right)
$$(prelim_gaussian5)

$$
\boldsymbol{\Sigma}_{a\vert b} = \boldsymbol{\Sigma}_{aa} - \boldsymbol{\Sigma}_{ab}\boldsymbol{\Sigma}_{bb}^{-1}\boldsymbol{\Sigma}_{ba}
$$(prelim_gaussian6)

with a mean that is linear in $\mathbf{x}_b$ and covariance that is independent of $\mathbf{x}_b$.

(gaussmarginalization)=
### Marginalization

Considering the same partitioning as above, we might also be interested in the **marginal** distribution of only part of our variables. Starting from the joint $p(\mathbf{x})$ marginalizing $\mathbf{x}_b$ means integrating over it (from the Sum Rule in {doc}`probability_theory`):

$$
p(\mathbf{x}_a) = \displaystyle\int p(\mathbf{x}_a,\mathbf{x}_b)\,\mathrm{d}\mathbf{x}_b
$$(prelim_gaussian7)

Working out the math leads to:

$$
p(\mathbf{x}_a) = \mathcal{N}\left(\mathbf{x}_a\vert\boldsymbol{\mu}_a,\boldsymbol{\Sigma}_{aa}\right)
$$(prelim_gaussian8)

which is a simple and intuitive result.

You can play around with the plot below by dragging the slider for $x_b$ to get a feel for the effect of marginalizion and conditioning on normally distributed random variables. You can also alter the covariance matrix by pulling the sliders for $\Sigma_{aa}$, $\Sigma_{bb}$, and $\rho$. The parameter $\rho$ is the correlation between $x_a$ and $x_b$, and is given by

$$
\rho_{x_a, x_b} = \operatorname{corr}(x_a, x_b) = \frac{\operatorname{cov}(x_a, x_b)}{\sigma_a \sigma_b} = \frac{\Sigma_{ab}}{\sqrt{\Sigma_{aa}\Sigma_{bb}}}
$$(prelim_gaussian9)

and can be interpreted as a normalized version of the covariance $\Sigma_{ab}$. We chose this parameterization to ensure a positive semidefinite covariance matrix $\boldsymbol{\Sigma}$.

<iframe src="../_static/gaussian-conditioning.html" width="800" height="800" frameborder="0"></iframe>

(bayes-stdexpressions)=
### Bayesian inversion

The last important result has to do with {doc}`bayes_theorem`. We show above how to get conditionals and marginals of partitioned Gaussians. Imagine we use those expressions to obtain the following graph model:

```{figure} ../figures/graph3.svg
:scale: 75%
:name: graph3

A two-node graph with linear-Gaussian relationships
```

From the graph above we have the conditional $p(\mathbf{y}\vert\mathbf{x})$ and the marginal $p(\mathbf{x})$ and they are both Gaussians. But now that we **observed** $\mathbf{y}$ we can see $p(\mathbf{x})$ as a **prior** and $p(\mathbf{y}\vert\mathbf{x})$ as a **likelihood**. This means we can use Bayes' Theorem to get a **posterior** over $\mathbf{x}$:

$$
p(\mathbf{x}\vert\mathbf{y}) = \displaystyle\frac{p(\mathbf{y}\vert\mathbf{x})p(\mathbf{x})}{p(\mathbf{y})}
$$(prelim_gaussbayes1)

From {numref}`graph3`, our joint distribution is $p(\mathbf{x},\mathbf{y})= p(\mathbf{x})p(\mathbf{y}\vert\mathbf{x})$. Since this is **again a Gaussian**, the distributions we are looking for are also Gaussian. This means we can use the conditioning and marginalization expressions from before to first compute the marginal over $\mathbf{y}$:

$$
p(\mathbf{y}) = \mathcal{N}\left(
\mathbf{y}\vert\mathbf{A}\boldsymbol{\mu}+\mathbf{b},
\mathbf{L}^{-1} + \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\mathrm{T}
\right)
$$(prelim_gaussbayes2)

and finally our posterior:

$$
p(\mathbf{x}\vert\mathbf{y}) = 
\mathcal{N}\left(
\mathbf{x}\vert\boldsymbol{\Sigma}^*\left[\mathbf{A}^\mathrm{T}\mathbf{L}\left(\mathbf{y}-\mathbf{b}\right)+\boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}\right],\boldsymbol{\Sigma}^*
\right)
$$(prelim_gaussbayes3)

$$
\boldsymbol{\Sigma}^* = 
\left(\boldsymbol{\Sigma}^{-1}+\mathbf{A}^\mathrm{T}\mathbf{L}\mathbf{A}\right)^{-1}
$$(prelim_gaussbayes4)

```{admonition} Further Reading    
:class: tip    
You can find a very insightful extended discussion on the Gaussian in Section 2.3. The most important relationships can already be found above, but you are encouraged to derive them by yourself while reading the complete discussion in 2.3.
+++                         
{bdg-danger}`bishop-prml`     
```

## Fitting distributions to data

The distributions above have parameters that can be tweaked, and this is of course how we can use distributions with the same functional forms to explain a very broad range of phenomena. For instance, a toin coss could be represented by a Bernoulli distribution with $\mu=0.5$, while the same distribution was used in our discussion about {doc}`bayes_theorem` with $\mu=0.6$ as prior to our beliefs over how much a person studied.

We quickly show the two main ways to fit these parameters to data here.

### The frequentist way

In a frequentist approach we ask ourselves the question:

````{card}
If I fix $\mu$ and draw a set of samples from my model at random, how likely would it be that those samples are **exactly** equal to the data I am observing?
````

For example, given a Bernoulli distribution $p(x\vert\mu) = \mu^x\left(1-\mu\right)^{1-x}$ and a dataset $\mathcal{D}$ with $N$ observations of $x$, we can imagine these samples are drawn **independently and identically distributed (i.i.d)** from $p(x\vert\mu)$ to compute a **likelihood function**:

$$
p(\mathcal{D}\vert\mu) = \displaystyle\prod^N_{n=1}\mu^{x_n}(1-\mu)^{1-x_n}
$$(prelim_mlefit1)

The higher this function becomes, the more likely it is that we can draw our dataset from our current distribution (given $\mu$).  To have a distribution that fits our data well, we are therefore incentivised to use the value of $\mu$ that **maximizes** this likelihood. We refer to this as a **Maximum Likelihood Estimation (MLE)** approach. Taking the logarithm of the expression above and setting its derivative to zero we get:

$$
\mu_{\text{MLE}} = \displaystyle\frac{1}{N}\displaystyle\sum^N_{n=1}x_n
$$(prelim_mlefit2)

### The Bayesian way

In a Bayesian approach we do not deal with drawing datasets and looking at frequencies, but instead with **prior** and **posterior** beliefs over our parameters. We ask ourselves the question:

````{card}
My prior experience says $\mu$ should be within a certain range. After observing my dataset, how much does my belief change?
````

This means we first adopt a **prior** $p(\mu)$ and after observing a dataset $\mathcal{D}$ we compute the **posterior**:

$$
p(\mu\vert\mathcal{D}) = \displaystyle\frac{p(\mathcal{D}\vert\mu)p(\mu)}{p(\mathcal{D})}
$$(prelim_bayesfit)

which will then move the prior to a new range. Note that here we do not have a single value for $\mu$ but a distribution. By observing more and more data, it stands to reason that the variance of $p(\mu\vert\mathcal{D})$ should gradually decrease, until the observations give such overwhelming evidence that $p(\mu)$ will be highly peaked around its true value.

The widget below compares both approaches as observations arrive. Each coin is sampled from a Bernoulli distribution with $\mu=0.5$. The Bayesian estimate starts from the conjugate $\operatorname{Beta}(2,2)$ prior and updates its density after every observation, while the dashed line shows the maximum likelihood estimate from the same observations.

<iframe src="../_static/bernoulli-fitting.html" width="100%" height="540" frameborder="0" title="Interactive comparison of Bayesian and maximum likelihood Bernoulli fitting"></iframe>

```{admonition} Further Reading    
:class: tip    
Read Section 2.1.1 to see how a Bayesian treatment can be given to the Bernoulli distribution. Pay attention to the important concept of **conjugate priors**.
+++                         
{bdg-danger}`bishop-prml`     
```

```{admonition} Exam questions    
:class: danger    
In the exam you might be asked to derive simple relations from known probability distributions. All the necessary mathematical expressions will be given to you, so no need to memorize anything. You might also be asked conceptual questions about these distributions, such as which variable types they model or how their parameters are calibrated.
+++    
{bdg-primary}`written-exam`    
```
