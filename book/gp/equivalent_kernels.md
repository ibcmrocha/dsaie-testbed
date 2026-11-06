# From Weights to Functions

For this part of the course we keep working with the familiar basis function models we have been using:

$$
y(\bx) = \bw^\T\basis(\bx)
$$(kernelyfunc)

where for regression problems we would use it directly together with some observation noise to define our targets, while in classification we would first feed it into a logistic sigmoid before using it. 

For now let us leave the application aside and just assume we have a vector with values of $y(\bx)$ for several $\bx$. Using the expression above and defining $\Basis_{nk}=\basis_k(\bx_n)$ we have:

$$
\mbf{y} = \Basis\bw
$$(kernelyvec)
 
and since $p(\bw)=\mathcal{N}(\mbf{0},\alpha^{-1}\mbf{I})$ and we can therefore grab as many samples of $\bw$ from our bag as we want, that means $\mbf{y}$ is also **jointly Gaussian**:

$$
p(\mbf{y}) = \gauss\left(\mbf{y}\vert\mbf{0},\displaystyle\frac{1}{\alpha}\Basis\Basis^\T\right)
\equiv
\gauss\left(\mbf{y}\vert\mbf{0},\mbf{K}\right)
$$(kerneljointgaussy)

where you can see that $\bw$ vanished, and we can instead just sample from $\mbf{y}$ instead of $\bw$!

<iframe src="../_static/weight-function-sampling.html" style="width: 100%; height: 1100px; border: 0;" loading="lazy" title="Equivalent Gaussian prior samples from weight space and function space"></iframe>

```{admonition} Further Reading    
:class: tip    
If you are interested in the exact derivation of going from $\bw$ to $\mbf{y}$, read Section 6.1 and the beginning of 6.4.1.
+++                         
{bdg-danger}`bishop-prml`     
```

We have actually already seen an example of this before when looking at {doc}`../bayesian/active_learning`. For a set of radial basis functions and sampling from the prior over $\bw$ we got the following samples:

```{figure} ../figures/equivalentkernels0.svg
:width: 500px

Samples from a basis function model obtained by sampling from the prior $p(\bw)$ and plotting the resulting $y(x)$. The same curves could also have been directly sampled from $p(\mbf{y})$ instead.
```

If we instead sampled from Eq. {eq}`kerneljointgaussy` we would get very similar curves. But how does one **sample a function**? That might feel a bit weird, but it is actually straightforward: each curve above is composed of 1000 $y$ values for different $x$. To get one of them we just need to compute the $1000\times 1000$ matrix $\mbf{K}$ and sample from the size-1000 version of Eq. {eq}`kerneljointgaussy`.

Each entry in $\mbf{K}$ relates two different values of $\bx$. We can write one of these entries as:

$$
K_{mn} = k(\bx_m,\bx_n) = \displaystyle\frac{1}{\alpha}\basis(\bx_m)^\T\basis(\bx_n)
$$(kernelbasisfunctionkernel)

and we call $k(\bx_m,\bx_n)$ a **kernel function**.