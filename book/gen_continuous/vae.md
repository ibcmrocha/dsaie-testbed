# Variational Autoencoders

Here we discuss a model that lies at the inferface between Bayesian machine learning and deep learning. We look into how to generalize the PPCA model from before to one where mappings to and from the latent space are allowed to be arbitrarily non-linear. We will benefit from our previous discussion on the EM algorithm for PPCA in order to maximize a lower bound of our data in a stochastic way and obtain a tractable model.

##  Encoder-decoder architectures

We have seen {doc}`PCA<ppca>` through a generative perspective that can be seen as an extension of {doc}`Gaussian Mixtures<../gen_discrete/gaussian_mixtures>`. But we have also seen how it can be cast as an error minimization problem. We can therefore see PCA as an encoder-decoder system:

```{figure} ../figures/autoencoder.svg
:scale: 50%
:name: autoencoder

A dimensionality reduction model seen through an encoder-decoder lens.
```

where a sample $\bx$ is mapped to its latent version $\bz$ and back into a reconstruction $\widetilde{\bx}$. Going through $\bz$ is a bottleneck that usually causes loss of information, so the reconstruction $\widetilde{\bx}$ will often not be perfect. To train this model we minimize the distortion measure:

$$
J = \displaystyle\frac{1}{N}\sum_{n=1}^N\lVert\bx-\widetilde{\bx}\rVert^2
$$(vae1)

which we have already seen results in linear mappings dictated by the eigenvectors $\mbf{U}$ of the data covariance matrix. We can also draw it as a simple neural network:

```{figure} ../figures/pcaautoencoder.svg
:scale: 50%
:name: pcaautoencoder

PCA seen through an encoder-decoder lens, with linear mappings between full and latent spaces.
```

and cast the problem as a **supervised** one, with targets being the same as inputs. Training such a neural network having only **linear activations** for every layer will lead to a model that is equivalent to PCA.

This works well for when the hidden patterns we are trying to capture are well described by a linear subspace. That is, however, often not the case with real-world data. Patterns are often exceedingly complex (e.g. from models that generate human faces) and that is translated to highly non-linear subspaces. A popular example is the so-called *swiss roll* dataset, which we show below together with an attempt to capture its intrinsic one-dimensional nature with PPCA:

```{figure} ../figures/vae0.svg
:figwidth: 750px

Fitting a PPCA model to a challenging dataset with non-linear patterns.
```

## Nonlinear autoencoders

We should therefore move towards nonlinear mappings between the $\bx$ and $\bz$ spaces. The easiest way to do this is with a conventional autoencoder, by including non-linear activation functions for both encoder and decoder layers:

```{figure} ../figures/nonlinautoencoder.svg
:scale: 50%
:name: nonlinautoencoder

Generalizing PCA for nonlinear mappings coming from neural networks.
```

This is however a purely deterministic approach, which means:

- It is prone to overfitting small datasets if the networks are too flexible;
- It loses the probabilistic interpretation we worked hard to get when doing PPCA;
- It cannot be used to generate new data.

In the following we introduce a fusion of deep learning and Bayesian machine learning in order to reach a useful non-linear generative dimensionality reduction model.

## Variational Autoencoder

As always we start with a graph model. Similar to PPCA, we assume real and latent variables $\bx$ and $\bz$, respectively, and a joint distribution combining two Gaussians:

```{figure} ../figures/vaegraph2.svg
:scale: 60%
:name: vaegraph

Graph model for a non-linear probabilistic version of PCA. The orange connections indicate non-linear relationships.
```

with the following densities for a single pair $(\bx,\bz)$:

$$
p(\bx,\bz) = p(\bz)p(\bx\vert\bz)
$$(vae2)

$$
\begin{cases}
p(\bz) = \gauss\left(\bz\vert\mbf{0},\mbf{I}\right)\\
p(\bx\vert\bz) = \gauss\left(\bx\vert f_\theta(\bz),\sigma^2\mbf{I}\right)
\end{cases}
$$(vae3)

where $\bs\theta$ is a set of parameters that dictate how latent data is decoded back to the actual data space. We have seen models like this many times before. The extra complication here is that now the function $f_\theta(\bz)$ is non-linear. As a consequence, the posterior

$$
p(\bz\vert\bx) = \displaystyle\frac{p(\bx\vert\bz)p(\bz)}{p(\bx)}
$$(vae4)

is **not Gaussian** anymore. This also means we cannot use the EM Algorithm to train this model, since it relies on an exact computation of $p(\bz\vert\bx)$. 

We should therefore find another way to make the model tractable. We start by assuming an **approximate posterior** $q_\varphi(\bz)$ of $p(\bz\vert\bx)$ which we can assume is Gaussian with mean $\bs\mu_q(\bx)$ and diagonal covariance matrix with components $\bs\sigma_q(\bx)$ both dictated by a set of parameters $\bs\varphi$. We can then make the pragmatic choice of using neural networks to learn both of these densities:

```{figure} ../figures/vae.svg
:scale: 75%
:name: vae

A probabilistic autoencoder architecture with neural nets on both sides. For now the two halves are not connected.
```

Note that for the density $p(\bx\vert\bz)$ we only need one output for the mean and recover the variance $\sigma^2$ by looking at the reconstruction error.

```{admonition} Further Reading    
:class: tip    
Recovering $\sigma^2$ directly from the reconstruction loss is exactly what we did when fitting $\beta$ using MLE for regression models. Look at Eq. (3.21) and see how our approximation for $\beta$ arises directly from the remaining loss. The logic here is the same but now with the loss coming from the terms $\lVert\bx_n-\widetilde{\bx}_n\rVert^2$.
+++                         
{bdg-danger}`bishop-prml`     
``` 

Looking at the model above, we see that for now the two parts do not exactly fit together: the encoder defines the approximate Gaussian but the decoder requires $\bz$ to be known (it is a **conditional** density). We will see below that writing down the expressions for maximizing the likelihood of our dataset will open up a nice path for allowing us to couple the two halves and train them together.

### The Evidence Lower Bound (ELBO)

As always, we will be maximizing $p(\mbf{X})$, but now we are in deep learning territory and will therefore be maximizing it in mini-batches with a small segment of our data. This is however not an issue, as we can do:

$$
\ln p(\mbf{X}) = \displaystyle\sum_{n=1}^N \ln p(\mbf{x}_n)
$$(vae5)

and maximize the likelihood of different parts of the dataset separately. In the following, we therefore look at only **a single sample** $\bx_n$ and will drop the subscript $n$ from now on for compactness.

We start with the joint density:

$$
\ln p(\mbf{x},\mbf{z}) = \ln p(\mbf{x}) + \ln p(\mbf{z}\vert\mbf{x})
$$(vae6)

We then isolate the likelihood and introduce our approximate posterior $q_\varphi(\bz)$:

$$
\ln p(\bx) = \ln p(\bx,\bz) - \ln p(\bz\vert\bx) + \ln q_\varphi(\bz) - \ln q_\varphi(\bz)
$$(vae7)

Similar to how we did it for PPCA, we rearrange these terms and take the expectation w.r.t. $q_\varphi(\bz)$ to arrive at:

$$
\displaystyle\ln p(\mbf{x}) = \underbrace{\int_\mbf{z} q_\varphi(\mbf{z})\ln\left[\frac{p(\mbf{x},\mbf{z})}{q_\varphi(\mbf{z})}\right]\,\mrm{d}\mbf{z}}_{\mathcal{L}} - \underbrace{\int_\mbf{z}q_\varphi(\mbf{z})\ln\left[\frac{p(\mbf{z}\vert\mbf{x})}{q_\varphi(\mbf{z})}\right]\,\mrm{d}\mbf{z}}_{-\mrm{KL}(q_\varphi\Vert\mrm{post})}
$$(vae8)

where we use $\int_\mbf{z} q_\varphi(\mbf{z}) \ln p(\mbf{x})\,\mrm{d}\mbf{z}=\ln p(\mbf{x})$, and we again see the **Evidence Lower Bound** $\mathcal{L}$ appearing.

### Maximizing the ELBO

For {doc}`PPCA<ppca>` we maximized $\mathcal{L}$ by iterating between E-steps and M-steps. Here this is not possible anymore because we cannot move $q(\bz)$ to $p(\bz\vert\bx)$ exactly, and in fact we only have a Gaussian approximation $q_\varphi(\bz)$ at our disposal. We will instead directly maximize $\mathcal{L}$ with respect to both $\bs\theta$ and $\bs\varphi$ at the same time with an SGD algorithm. This means $q_\varphi(\bz)$ will **indirectly** move as close as possible to $p(\bz\vert\bx)$, so we will also be implicitly minimizing the KL term. Recall that the same happened for PPCA, but there we could start from both directions (i.e. set $\mrm{KL}(q_\varphi\Vert\mrm{post})=0\Rightarrow\mathcal{L}$ increases; or maximize $\mathcal{L}$ w.r.t. $q(\bz) \Rightarrow \mrm{KL}(q_\varphi\Vert\mrm{post})$ moves to zero).

Looking just at the lower bound we can rewrite it as:

$$
\mathcal{L} = \mathbb{E}_{q_\varphi}\left[\ln p(\bx,\bz) - \ln q_\varphi(\bz)\right]
$$(vae9)

and decomposing the joint into $p(\bx,\bz) = p(\bz)p_\theta(\bx\vert\bz)$ we get:

$$
\mathcal{L}(\bs\theta,\bs\varphi) = \mathbb{E}_{q_\varphi}\left[\ln p_\theta(\bx\vert\bz)\right] + \underbrace{\mathbb{E}_{q_\varphi}\left[\ln p(\bz) - \ln q_\varphi(\bz)\right]}_{-\mrm{KL}(q_\varphi\Vert\mrm{prior})}
$$(vae10)

where the dependency of $\mathcal{L}$ with respect to both $\bs\theta$ and $\bs\varphi$ is now explicit. The two terms have a clear interpretation:

- A **reconstruction** term $\mathbb{E}_{q_\varphi}\left[\ln p_\theta(\bx\vert\bz)\right]$. Since $p_\theta(\bx\vert\bz)$ has mean $f_\theta(\bz)$ which is computing the reconstruction $\widetilde\bx$, this term increases the closer $\widetilde{\bx}$ is to the original $\bx$;
- A **regularization** term $-\mrm{KL}(q_\varphi\Vert\mrm{prior})$ which measures how similar the approximate posterior $q_\varphi(\bz)$ is to the prior $p(\bz)=\gauss(\bz\vert\mbf{0},\mbf{I})$. This term cannot be higher than zero since $\mrm{KL}(a\Vert b)\geq 0$.

It is usually not possible to maximize both parts, so the KL term tends to balance the reconstruction term and help prevent overfitting. It is also common practice to fine-tune the model by scaling the KL term with an extra parameter $\beta$:

$$
\mathcal{L}(\bs\theta,\bs\varphi) = \mathbb{E}_{q_\varphi}\left[\ln p_\theta(\bx\vert\bz)\right] - \beta\,\mrm{KL}(q_\varphi\Vert\mrm{prior})
$$(vae11)

which is analogous to scaling up the observation noise of a PPCA model, shifting $p(\bz\vert\bx)$ towards $p(\bz)$.

### Taking expectations, and the reparametrization trick

To compute $\mathcal{L}$ we still need to compute expectations over the complete $q_\varphi(\bz)$. Since both $p(\bz)$ and $q_\varphi(\bz)$ are Gaussian, we can actually compute the KL term exactly:

$$
\mrm{KL}(q_\varphi\Vert\mrm{prior}) = -\displaystyle\frac{1}{2}\sum_{m=1}^M\left(1+\ln(\sigma_m^2) - \mu_m^2 - \sigma_m^2\right)
$$(vae12)

and for the reconstruction term we can be pragmatic and use a crude Monte Carlo approximation with **a single sample**:

$$
\mathbb{E}_{q_\varphi}\left[\ln p_\theta(\bx\vert\bz)\right] \approx \ln p_\theta(\bx\vert\widetilde{\bz})
$$(vae13)

which means we need to sample from $q_\varphi(\bz)$. This creates one final issue: when backpropagating through the network we do not want to include this sampling operation on the automatic differentiation graph. To resolve this we resort to the so-called **reparametrization trick**. The complete forward pass through the network therefore becomes:

- Feed a sample through the encoder and get values for $\bs\mu_q$ and $\bs\sigma_q$;
- Sample from an auxiliary variable $\bs\epsilon\sim\gauss(\mbf{0},\mbf{I})$ which lies outside the autograd graph;
- With a sample $\widetilde{\bs\epsilon}$ of the auxiliary variable, compute a sample $\widetilde{\bz} = \bs\mu_q + \bs\sigma_q\odot\widetilde{\bs\epsilon}$;
- Feed $\widetilde{\bz}$ through the decoder and get a reconstruction $\widetilde{\bx}$;
- Add both the KL term (Eq. {eq}`vae12`) and the error term $\lVert\bx-\widetilde{\bx}\rVert^2$ of $p_\theta(\bx\vert\bz)$ to the loss function.

This effectively allows us to couple the two halves of the model and train them together:

```{figure} ../figures/vaereparametrization.svg
:scale: 75%
:name: vaereparametrization

The reparametrization trick allows us to link the two halves of the model and avoid problems with backpropagation.
```

and the expressions for a mini-batch of data are simply a sum (or mean) of the ones for individual samples.

### Swiss Roll example

We train a Variational Autoencoder on the *swiss roll* dataset we tried to describe with PPCA before. Here both encoder and decoder have 6 hidden layers of 50 neurons each and SELU activation (Scaled Exponential Linear Unit). We set the regularization weight $\beta=1$ and train the autoencoder for 2000 epochs with a dataset of $N=500$ observations and a mini-batch size of 40. We use the Adam optimizer with standard settings.

With the trained model, we can try to reconstruct the original training data after mapping from and to a one-dimensional latent space:

```{figure} ../figures/vae1.svg
:figwidth: 750px

Fitting the *swiss roll* dataset with a VAE: reconstructing data
```

where we see that $z$ captures the intrinsic one-dimensional nature of the data quite well. Compare this with the PPCA result above by considering that the complete reconstructed dataset would lie on top of the straight line PPCA fits. We can also use the VAE to generate fake swiss roll samples. As always we use ancestral sampling, by first taking samples from $p(\bz)$ and then using the decoder to reconstruct them:

```{figure} ../figures/vae2.svg
:figwidth: 750px

Fitting the *swiss roll* dataset with a VAE: generating new samples
```

<iframe src="../_static/vae-latent-explorer.html" style="width: 100%; height: 1040px; border: 0;" loading="lazy" title="Explore a real MNIST VAE through its two-dimensional latent space and replay its training"></iframe>

```{admonition} Exam questions    
:class: danger    
You might find exam questions about variational autoencoders. You will not be asked to reproduce the stochastic lower-bound formulation above, although you should be familiar with the approximations we make and how we can still arrive at a tractable model. The conceptual interpretation of the two terms that arise in the final loss function are important. The similarities and differences between VAEs and PPCA are also crucial. You should also be aware of when the usage of a VAE model is warranted (instead of e.g. PPCA or Gaussian mixtures).
+++    
{bdg-primary}`written-exam`    
``` 