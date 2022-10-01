# Reproducing Kernel Hilbert Spaces

## Kernels 🍿
A function is in essence a mapping from one space to another. In our case, we want to map our random variables $x$ and $y$ from a probability space to a new space that makes it easier to compare the two associated densities. 

For inspiration, let's first consider $x$ and $y$ as points in $R^n$. Suppose we want to calculate the discrepancy of these two points following some transformation. Kernel methods are well-established to perform this exact task:

For $x \in R^n$ and $y \in R^n$:

$$k_{\phi}(x, y) = \langle \phi(x), \phi(y) \rangle$$
<!-- where,

$$X = \{x_i: x_i \in R^n, i=1,...,N\}, Y = \{y_j: y_j \in R^n, j=1,...,M\}$$

and  -->

where $\phi$ is our mapping $R^n \rightarrow R^m$ and $k_{\phi}$ is known as the kernel function associated with $\phi$. 

The kernel function $k_{\phi}$ condenses the two step process of mapping and inner product evaluation to a single direct comparison of the two data points. By defining some non-linear transformation, we can greatly improve the seperability of our data. Kernel methods have become effective tools for exposing the discrepancy between sets of points. 

Could we adopt these kernels for our probability densities  $\mathbb{P}$ and $\mathbb{Q}$? Instead of points in $R^n$, we would want $\phi$ to map the pdfs for our random variables from a distribution space to some new space where we can more easily compare them. One example of a mapping function for a density function could be the expectation function:

$$\mathbb{E}_{x \sim \mathbb{P}}[x] = \int_X x p(x) dx$$

Our resulting kernel function:

$$k(x,y) = \langle \mathbb{E}_{x \sim \mathbb{P}}[x], \mathbb{E}_{y \sim \mathbb{Q}}[y]\rangle$$

It can be proved that:

$$\langle \mathbb{E}_{x \sim \mathbb{P}}[x], \mathbb{E}_{y \sim \mathbb{Q}}[y]\rangle = \mathbb{E}_{x \sim \mathbb{P}, y \sim \mathbb{Q}}[\langle x, y\rangle]$$

Leaving us with the the expecation of a linear kernel:

$$k(x,y) = \mathbb{E}_{x \sim \mathbb{P}, y \sim \mathbb{Q}}[\langle x, y\rangle]$$

It seems like we've been able to extend our notion of kernels to probability densities! 

## One Kernel One Function Space

In the last section, we were able to define transformations on pdfs from our probability space with kernel functions. But what is the space defined by our kernel?

a kernel defines a function space

reproducing kernel

resulting RKHS


## Keep it smooth

One last problem, we need the function $f$ to be smooth. This is acheived by restricting our function space to the unit ball of a reproducing kernel Hilbert space (RKHS). This ensures that the coefficients of transformations of higher frequencies will decay guaranteeing smoothness of our kernels.
