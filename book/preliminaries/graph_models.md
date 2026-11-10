# Graph Models

Graph representations will be used during the course to allow for more insight into the structure of a range of probabilistic models we will consider. Here we briefly review the basic concepts of graphs and in particular of directed graphs.

## Basic symbology, quick reference

Graphs are composed of **nodes** and **links**. Each node represents a (possibly random) variable, and each link represents probabilistic relationships between nodes.

The graph models we will see here, also called *Bayesian Networks*, are widely used in science and industry not just to describe machine learning models but also for instance in decision making and risk estimation applications.

The figure below shows the graph symbology we will be using throughout the course. Come back to this page if you need a recap when you come across a new graph.

```{figure} ../figures/graphsymbols.svg
:scale: 45%
:name: graphsymbols

Types of nodes in a probabilistic graph
```

## From graphs to joint distributions

Directed graphs encode dependencies between variables. That means that from a given graph we can extract an expression for the **joint distribution** of its variables in a way that makes these dependencies clear.

Consider the following graph:

```{figure} ../figures/graph1.svg
:scale: 50%
:name: graph1

A simple two-node probabilistic graph
```

We have two probabilistic variables $a$ and $b$ and the link indicates that $b$ depends on $a$. This means we can factorize $p(a,b)$ as:

$$
p(a,b) = p(a)p(b\vert a)
$$(prelim_graphfactorization1)

But **why** would we do that? Often the reason is that $p(a,b)$ can be too complex to express with simple (e.g. Gaussian) distributions, while both $p(a)$ and $p(b\vert a)$ might have more tractable forms.

The basic rule is to express dependencies by conditional distributions. Take now the more complicated graph:

```{figure} ../figures/graph2.svg
:scale: 50%
:name: graph2

A more complicated probabilistic graph
```

The joint distribution for this model is:

$$
p(a,b,c,d,e,f) = p(a)p(c)p(b\vert a)p(d\vert c)p(e\vert b,d)p(f\vert e)
$$(prelim_graphfactorization2)

Try to get this expression for the joint distribution by yourself and see if it matches the one above. Start from the independent nodes and gradually move towards nodes for which all dependencies are taken into account until you reach the end of the graph.

Practice writing the factorized joint distribution for a random graph below. Each arrow points from a parent to a variable that depends on it.

<iframe src="../_static/graph-factorization.html" style="width: 100%; height: 720px; border: 0;" loading="lazy" title="Practice factorizing the joint distribution of a random Bayesian network"></iframe>

```{admonition} Further Reading    
:class: tip    
Is it possible to obtain a more general expression valid for any graph? You can find out by reading Section 8.1 up until before 8.1.1. 
+++                         
{bdg-danger}`bishop-prml`     
```

(sec-ancestralsampling)=
## Generative models, ancestral sampling

You will come across **generative** models during the course. You can already think of popular examples such as ChatGPT. If you send ChatGPT the same question several times, different answers will be given every single time. Internally, this often involves sampling from one or more probability distributions.

Consider the following graph representing a hypothetical AI image generator:

```{figure} ../figures/ancestralsampling.svg
:scale: 50%
:name: ancestralsampling

Probabilistic graph from a hypothetical image generator
```

The joint probability density associated with this graph is:

$$
p(a,b,c,d) = p(a)p(b)p(c\vert a,b)p(d\vert c)
$$(prelim_factorization3)

Suppose we would like to take samples from $p(a,b,c,d)$. In **ancestral sampling** we would:

- Draw samples from $p(a)$ and $p(b)$ (independently);
- Compute $p(c\vert a,b)$ and draw a sample from it;
- Finally, compute $p(d\vert c)$ and draw a sample from it;

Imagine $d$ represents the final image we are interested in. Since in that case we are only interested in sampling from the marginal $p(d)$, we just do the above and ignore the samples we get from $a$, $b$ and $c$ in the process.

```{admonition} Exam questions    
:class: danger    
In the exam you might be asked to work with probabilistic graphs and their associated joint probability densities. We might ask you to translate a graph into a joint density, or to work out other related properties. You do not need to memorize any mathematical expressions.
+++    
{bdg-primary}`written-exam`    
``` 