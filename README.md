# Path Optimization on Multi-Weighted Graphs

This is a toy implementation of a path optimizer for graphs where each edge has
multiple weights.

## Edge Weights

Each edge can have multiple weights representing different network metrics.
These can be standard types (like `float` or `int`) or user-defined types. The
key requirement is that we can define an **aggregation** method to combine the
weights along a path.

This demo includes three weight types:

- **Latency**: real number; aggregated by **sum**
- **Throughput**: real number; aggregated by **minimum**
- **Integrity**: binomial opinion; aggregated by **binomial multiplication** (from subjective logic)

### Binomial opinions

In subjective logic, an opinion is a formal representation of uncertain belief.
It expresses the belief, disbelief, and uncertainty about a proposition. An
opinion is defined as a tuple of three real numbers $\omega = (b, d, u)$, where $b$ is
the belief, $d$ is the disbelief, and $u$ is the uncertainty. Each number is in
the range $[0, 1]$ and they sum up to 1, i.e. $b + d + u = 1$.

To combine opinions along a path, we use **binomial multiplication** $\omega_x
∧ \omega_y$, which is defined as follows:

$b_{x∧y} = b_x b_y + \frac{b_x u_y + b_y u_x}{3}$

$d_{x∧y} = d_x + d_y - d_x d_y$

$u_{x∧y} = 1 - b_{x∧y} - d_{x∧y}$


## Optimization via Simulated Annealing


In order to find the optimal path, we use a **simulated annealing** algorithm.
First, we need to define a cost function or **energy** which we want to minimize.

### Cost function

For this purpose, we define a weighted sum of the three weights:

$$
E = \alpha \text{LatencyCost}({\text{path}}) + \beta \text{ThroughputCost}({\text{path}}) + \gamma \text{IntegrityCost}({\text{path}})
$$

- **LatencyCost**: sum of edge latencies
- **ThroughputCost**: minimum edge throughput
- **IntegrityCost**: logit of the belief from the aggregated binomial opinion:
  $\text{logit}(b) \equiv \log\left(\frac{b}{1 - b}\right)$

Weights $\alpha$, $\beta$, and $\gamma$ tune the relative importance of each
term. Their signs indicate if the metric should be minimized (positive) or
maximized (negative).

### Annealing steps

The simulated annealing algorithm starts with an initial path and iteratively
modifies it to find a better one. At each step, a new path is created by
modifying the current one using one of three operations:

- REPLACE: Given a segment $s \rightarrow m \rightarrow t$, replace with $s \rightarrow m' \rightarrow t$ if $\exists m' \in N(s) \cap N(t)$ and $m' \not\in \text{PATH}$.
- INSERT: Given a segment $s \rightarrow t$, replace with $s \rightarrow m \rightarrow t$ if $\exists m \in N(s) \cap N(t)$  and $m' \not\in \text{PATH}$.
- BYPASS: Given a segment $s \rightarrow m \rightarrow t$, replace with $s \rightarrow t$ if $s \in N(t)$.

### Results

We test the algorithm on a random graph with:

- 100 nodes
- Edge probability $p = 0.2$
- Random weights for all edges
- 6000 annealing iterations

Parameters:

- $\alpha = 0.005$ (minimize latency)
- $\beta = -10$ (maximize throughput)
- $\gamma = -10$ (maximize belief in integrity)

The results are displayed in the figure below.

![Results](annealer_plot.png)
