---
title: "Near-Optimal Approximations for Bayesian Inference in Function Space"
collection: publications
permalink: /publication/2025-02-25-near-optimal-approximations-for-bayesian-inference-in-function-space
excerpt: 'A scalable inference algorithm for Bayes posteriors defined on a reproducing kernel Hilbert space (RKHS).'
date: 2025-02-25
paperurl: 'https://www.arxiv.org/abs/2502.18279'
citation: 'Your Name, You. (2009). &quot;Paper Title Number 1.&quot; <i>Journal 1</i>. 1(1).'
---
We propose a scalable inference algorithm for Bayes posteriors defined on a reproducing kernel Hilbert space (RKHS).
Given a likelihood function and a Gaussian random element representing the prior, the corresponding Bayes posterior measure $\Pi_{\text{B}}$ can be obtained as the stationary distribution of an RKHS-valued Langevin diffusion.
We approximate the infinite-dimensional Langevin diffusion via a projection onto the first $M$ components of the Kosambi–Karhunen–Loève expansion.
Exploiting the thus obtained approximate posterior for these $M$ components, we perform inference for $\Pi_{\text{B}}$ by relying on the law of total probability and a sufficiency assumption.
The resulting method scales as $O(M^3+JM^2)$, where $J$ is the number of samples produced from the posterior measure $\Pi_{\text{B}}$.
Interestingly, the algorithm  recovers the posterior arising from the sparse variational Gaussian process (SVGP) (see [Titsias, 2009](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf)) as a special case---owed to the fact that the sufficiency assumption underlies both methods.
However, whereas the SVGP is parametrically constrained to be a Gaussian process, our method is based on a non-parametric variational family $\mathcal{P}(\mathbb{R}^M)$ consisting of all  probability measures on $\mathbb{R}^M$.
As a result, our method is provably close to the optimal $M$-dimensional variational approximation of the Bayes posterior $\Pi_{\text{B}}$ in $\mathcal{P}(\mathbb{R}^M)$ for convex and Lipschitz continuous negative log likelihoods, and coincides with SVGP for the special case of a Gaussian error likelihood.

We also present our implementations available on <a href="https://github.com/jswu18/projected-langevin-sampling">GitHub</a>.

[Download paper here](https://www.arxiv.org/abs/2502.18279)
