# Decision Theory

We now focus on a very different class of problems. In constrast to making continuous predictions for regression problems, we now want to make an informed choice between two or more discrete **classes** given some input values.

This is a popular problem in machine learning, e.g. in computer vision, a classification model can be trained to classify pictures of animals, or to identify the occurence of cancer based on x-ray images. In civil engineering, we can think of a number of relevant situations on which classification problems arise:

| Problem | Inputs | Classes |
| :-- | :-- | :-- |
| Structural health monitoring of a bridge | Displacements at several locations | Healthy / Needs repair / Needs demolition |
| Predicting flood | Time series of rainfall in Delft | Will flood / Will not flood |
| Self-driving cars | Images from several cameras | Traffic light is green / yellow / red |
| Monitoring air quality | Time series of Particulate Matter data | Air quality will be good / bad |


Note that many of these problems could be recast as regression problems, for instance we could recast the problem of predicting flood or no flood in Delft to one of predicting the average water level of our canals.

Regardless, we must change the way in which we make decisions based on data in case we are dealing with classification problems. Here it is useful to start with a **generative** form for the model:

```{figure} ../figures/classificationgraph1.svg
:scale: 60%
:name: classificationgraph1

Graph model for a classification problem in its generative form.
```

where the joint distribution is given by:

$$
p(\mathcal{C},\mathbf{x}) = p(\mathcal{C})p(\mathbf{x}\vert\mathcal{C})
$$(classificationjoint)

and sampling from this graph means first sampling a class $\mathcal{C}_k$ and then sampling from the conditional $p(\mathbf{x}\vert\mathcal{C}_k)$. It makes sense that different classes should generate different distributions of input features (an x-ray of a person with cancer should look quite different than an x-ray from a healthy person), and these are the patterns we would like to capture with our classification problem.

Imagine we have a dataset of $N$ observations of $\mathbf{x}$ and their corresponding classes $\mathcal{C}_k$. For the priors $p(\mathcal{C}_k)$ we could keep it simple and approximate them by their proportions on our dataset, i.e. $p(\mathcal{C}_k)=n_k/N$. We then keep it simple and fit a Gaussian distribution to each of our $p(\mathbf{x}\vert\mathcal{C}_k)$.

Now we observe a new $\mathbf{x}$ for which we do not know what class it belongs to. Since we now have a joint distribution, we can use Bayes' Theorem to help us:

$$
p(\mathcal{C}_k\vert\mathbf{x}) =
\displaystyle\frac
{p(\mathbf{x}\vert\mathcal{C}_k)p(\mathcal{C}_k)}
{p(\mathbf{x})}
$$(bayesclassification)

and now we have posterior probabilities $p(\mathcal{C}_k\vert\mathbf{x})$ for each class. Finally we need to make a decision, to which class should we assign our new point?

## Decision regions, decision boundaries

To get to the answer it is easier to look at the problem in 1D and with just two classes. Imagine we have the exact joints for our two classes, and since $\mathcal{C}$ is discrete it suffices to plot both together as functions of $x$:

<iframe src="../_static/decision-boundary.html" width="100%" height="510" frameborder="0" loading="lazy" title="Interactive two-class decision boundary and confusion matrix"></iframe>

<p id="decisionboundaries1" style="text-align: center;"><em>A two-class classification problem in 1D. Move the decision boundary in the interactive figure and notice this is not always the best choice.</em></p>

We define a **decision boundary** as shown above: to the left of it we classify an observation of $x$ as $\mathcal{C}_1$, to the right of it we classify as $\mathcal{C}_2$. Now we investigate if this position for the boundary makes sense:

- When we classify a point as $\mathcal{C}_1$, there is a chance we made a mistake. This chance is proportional to the combination of the dark and light green areas above;
- When we classify a point as $\mathcal{C}_2$, there is a chance it was actually from the other class. This chance is represented by the dark blue area above.

These areas can be put into a **confusion matrix** which you also see above. Naturally we want to minimize the sum of these areas. But look again at the figure above: the sum of the dark green and dark blue areas is always constant irrespective of where we put the decision boundary. But by moving the boundary we can actually get rid of the light green area. By doing that we arrive at:

```{figure} ../figures/decisionboundaries2.svg
:scale: 50%
:name: decisionboundaries2

The problem from above again, but now with optimum decision boundary. This is equivalent to picking the class with the highest posterior probability.
```

which is the model with the minimum misclassification rate. The boundary lies at the point where the two joints have the same value. In other words, we should assign a new unlabeled point to the class with the highest value of joint probability. And because $p(\mathbf{x})$ is a common denominator for all classes in Eqs. {eq}`bayesclassification`, this is equivalent to **picking the class with the highest posterior probability**:

```{figure} ../figures/introclassificationposteriors.svg
:scale: 50%
:name: introclassificationposteriors

Example of resulting posterior distributions over classes, plotted against input feature $x$. Note that plotted against $x$ this is not actually a probability density.
```

This is of course a very intuitive result: given a new point, we compute the probability that it belongs to each of our classes, and finally make the pragmatic choice of assigning it to the class with the highest probability. It is reassuring to see that, on average, this approach will lead to the least amount of misclassifications.

```{admonition} Further Reading
:class: tip
What if different misclassifications have different consequences? If a bridge is about to collapse, there is a high cost involved in classifying it as healthy, while classifying a healthy bridge as defective is much less harmful. You can find out how to compensate for that by reading the short Sections 1.5.2 and 1.5.3.
+++
{bdg-danger}`bishop-prml`
```

## Modeling approaches

As usual, we now stand at a crossroads. There are three main approaches to solving classification problems:

````{card}
**Generative models**
^^^
This is the approach we have seen above. Given some data, we fit distributions $p(\mathbf{x}\vert\mathcal{C}_k)$ to each class and adopt priors $p(\mathcal{C}_k)$. For a new $\mathbf{x}$, we use Bayes' Theorem to compute posteriors $p(\mathcal{C}_k\vert\mathbf{x})$ and assign it to the class with the highest posterior probability.

We say these models are generative because we end up with distributions for $\mathbf{x}$ which we can sample from. For instance, if we are classifying images, our final model can also generate new synthetic (fake) images.
````

````{card}
**Posterior inference**
^^^
We take a shortcut and directly infer the posteriors $p(\mathcal{C}_k\vert\mathbf{x})$ for each class. We then do the same as above and classify a new $\mathbf{x}$ to the class with highest posterior probability. These models are not generative, we can use them to make probabilistic decisions but not to generate synthetic data.
````

````{card}
**Discriminant models**
^^^
We take an even shorter path and fit a function $f(\mathbf{x})$ that directly gives us a class label. This approach is completely deterministic.
````

```{admonition} Exam questions
:class: danger
You should not expect numerical questions related to this topic in the exam. You should instead focus on the concepts: relating the discussion here with decision theory for regression, arriving at a decision rule for classification, and the different modeling approaches.
+++
{bdg-primary}`written-exam`
```
