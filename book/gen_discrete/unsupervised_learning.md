# Unsupervised Learning

In regression, our goal was to fit a regression function based on observed targets which we could then use to make predictions for new input values. In classification, we inferred posterior densities to our classes again based on targets (labeled data). 

For the next two parts of the course, we will move away from fitting targets/labels/observations and on to **unsupervised learning**. In this class of models, we attempt to learn patterns based on **unlabeled data**. The goal is to use a dataset to find hidden structures that will help us better understand our data.

In unsupervised learning we assume that whatever patterns our data has are probably related to some variable (or group of variables) we cannot directly observe. To indicate their hidden nature, we give them the name **latent variables**.

To make the discussion more concrete, let us look at two practical examples:

(clustering-unsupervised-clustering)=
## Clustering

Imagine a teacher grades an exam, plots the grade distribution for the whole class and obtains:

```{figure} ../figures/1dgrades.svg
:scale: 50%
:name: 1dgrades

Distribution of grades after a hypothetical exam. A concerning pattern can be clearly seen.
```

Our eyes can clearly see there is something very wrong with these grades: There is a clear split in the class, some did very well, others very badly. Suppose the teacher then finds out that about half of the class did not have access to reading material for review due to a computer bug. This information, previously **hidden**, explains the distribution quite well.

This is a case where the **latent** variable (bug/no bug) could in the end be observed. But could we still find the pattern even without observing it? This is the goal of clustering:

````{card}
Clustering
^^^
Assume the existence of one or more **discrete** latent variables. For each data point of observable information, infer its associated latent value(s). This divides the data into two or more **clusters**.
````

After employing a clustering algorithm, the dataset from before can for instance look like this:

```{figure} ../figures/1dgradesclustered.svg
:scale: 50%
:name: 1dgradesclustered

Distribution of grades after a hypothetical exam, after clustering. The pattern is clearly captured. The circles represent cluster means.
```

The pattern our eyes could already identify can also be clearly captured by machine learning. Note that this creates two clusters, red and green, but it does **not** tell us what causes the data to be split. If the teacher never realises there was a bug in the study material, the cause for the grade distribution will not be elucidated. Nevertheless, identifying the pattern already points to the existence of an issue, and is therefore crucial. In general data is high dimensional and it is usually impossible to rely on identifying patterns visually.

One last question about the plot above: to which cluster should we assign the one student who got a 5.8? It is unlikely that the latent variable we learned helps explain their situation well (for instance, they might have found study material online independently; or they might have had the material but had a personal issue and could not study). Many clustering techniques only allow for a **hard** cluster assignment, which can be too reductionist.

## Dimensionality reduction

Let us stay with the example of the grades. Imagine a teacher gives a written exam and a project and plots the grade distribution of the class:

```{figure} ../figures/2dgrades.svg
:scale: 50%
:name: 2dgrades

Distribution of grades for a hypothetical course with two assignments. Each data point represents a student.
```

Now we have two-dimensional data, but still unlabeled. Again, a clear pattern can be seen by looking at the data. It again seems like there is a **latent variable** that explains why the grades are so correlated. If we could observe this variable, understanding the grade distribution would be much easier. But in practice we usually cannot. This is where dimensionality reduction comes in:

````{card}
Dimensionality Reduction
^^^
Assume the existence of a **continuous** latent variable with **less dimensions** than the observation space. Infer a mapping that associates every data point in the observation space to a position in latent space. Exploit the latent representation of the data to make decisions or perform further inference.
````

With a dimensionality reduction technique, we can reduce the original 2D observation space to a 1D one, shown in red below:

```{figure} ../figures/2dgradespca.svg
:scale: 50%
:name: 2dgradespca

Distribution of grades for a hypothetical course with two assignments. The 2D data can be very well explained in a 1D latent space.
```

As with clustering, the latent variable $\alpha$ we learn from the data does not have an immediate interpretation. It could represent for instance hours of study, as shown above, but that is up to the user's interpretation: by itself, machine learning can merely point out that a pattern exists, not immediately find a reason for it.

The figure above also shows that some spread remains around the $\alpha$ coordinate. In the 1D latent space, this variation is therefore considered unexplained and discarded. This is an important facet of dimensionality reduction: we are always trying to minimize it, but information is almost always lost when going to for lower-dimensional representations.

## Why automate clustering?

Clustering a handful of observations by eye may seem easy in one or two dimensions. Try making the same two-cluster decision below as the number of dimensions grows. In four dimensions, each observation appears in all six pairwise views, so selecting it in one plot updates it everywhere.

<iframe src="../_static/manual-clustering.html" title="Interactive manual clustering in one, two, and four dimensions" width="100%" height="590" frameborder="0" loading="lazy"></iframe>

```{admonition} Exam questions    
:class: danger    
The contents here are quite important, and might appear in the exam in the form of conceptual questions.
+++    
{bdg-primary}`written-exam`    
```
