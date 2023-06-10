---
title: 'Expectation Maximisation'
date: 2023-05-21
permalink: /posts/2023/05/expectation-maximisation/
tags:
  - Expectation Maximisation
---
Expectation maximisation is a powerful algorithm that can be applied to a wide variety of problems, including clustering, mixture models, and hidden Markov models. In this post, I will present the general formulation of the algorithm, and applying it to the k-means clustering problem as an example.

Consider a $$\textit{general}$$ model/distribution $$P(\mathcal{X}, \mathcal{Z}\vert \theta)$$ where $$\mathcal{X}$$ is the observation data space, $$\mathcal{Z}$$ is the latent or missing data space, and $$\theta$$ are the model/distribution parameters from a parameter space $$\Theta$$. As an example, the k-means clustering model fits into this general structure:

$$P(\mathcal{X}, \mathcal{Z}\vert \theta) = \prod_{n=1}^{N} \left( \sum_{k=1}^K \frac{1}{K} \delta \left(z_{n, k}, \arg\min_{\ell \in \{1, \dots, K\}} \|\mathbf{x}_n - \mathbf{\mu}_{\ell}\|_2^2\right)\right)$$

where $$\delta(i, j)$$ is the Kronecker delta function:

$$\delta(i, j) = \begin{cases}
1 ,  & \text{if } i=j\\
0 , & \text{if } i \neq j\\
\end{cases}$$

For $$N$$ observations and $$\mathbf{x}_n \in \mathbb{R}^{D}$$, we can choose $$\mathcal{X}$$ as $$\mathbb{R}^{N\times D}$$, $$\mathcal{Z}$$ as $$\{0, 1\}^{K \times N}$$ for $$K$$ cluster allocations, and $$\Theta$$ as $$\mathbb{R}^{K\times D}$$ for the $$\mu_{\ell}$$'s, the mean vectors of each cluster.

We wish to find $$\hat{\theta}$$, parameters $$\textit{maximising}$$ $$P(\mathcal{X}, \mathcal{Z}\vert \theta)$$, the parameters for which the data is most likely. In $$P(\mathcal{X}\vert \theta)$$ settings without latent variables, a standard approach would be to solve for $$\hat{\theta}$$ by setting $$\frac{\partial}{\partial \theta}P(\mathcal{X}\vert \theta) = 0$$. But for $$\mathcal{Z}$$, a latent space without observations, we can't perform the same maximisation. We need to maximise the distribution after $$\textit{marginalisation}$$ $${\theta \in \Theta} \int_{\mathcal{Z}}P(\mathcal{X}, z\vert \theta) dz$$ which is often an intractable integral or computationally intractable. In the case of k-means, we see that an integral over $$\mathcal{Z}$$ is a summation over $$K^N$$ possible configurations, which becomes computationally intractable as $$N$$ and $$K$$ grow. To circumvent this issue, we define a $$\textit{lower bound}$$ on the integral. This begins by first reformulating the problem, defining the loss function $$\ell (\theta) = \log \int_{\mathcal{Z}}P(\mathcal{X}, z\vert \theta) dz$$. The logarithm is monotonic, so solving for $$\theta$$ which maximises $$\ell$$ will also maximise our marginalised distribution.

We can choose any distribution on $$\mathcal{Z}$$ parameterised by $$\theta' \in \Theta'$$, $$Q_{\theta'}(\mathcal{Z})$$ such that:

$$\begin{align}
    \ell({\theta}) = \log \int_{\mathcal{Z}}Q_{\theta'}(z) \frac{P(\mathcal{X}, z\vert \theta)}{Q_{\theta'}(z)} dz & = \log \left(\mathbb{E}_{z \sim Q_{\theta'}}\left[ \frac{P(\mathcal{X}, z\vert \theta)}{Q_{\theta'}(z)} \right]\right)\\
    & \geq  \mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(\frac{P(\mathcal{X}, z\vert \theta)}{Q_{\theta'}(z)}\right) \right] = \mathcal{F}(\theta', \theta)
\end{align}$$

We have a lower bound on $$\ell(\theta)$$ by Jensen's inequality given that $$\log$$ is concave. $$\mathcal{F}(\theta', \theta)$$ is known as the free energy or evidence lower bound (ELBO). Instead of trying to maximise an intractable loss, we maximise our loss indirectly by finding the parameters $$\theta'$$ and $$\theta$$ that maximise the free energy lower bound:

$$\max_{\theta' \in \Theta', \theta \in \Theta} \mathcal{F}(\theta', \theta) \leq \max_{ \theta \in \Theta} \ell(\theta)$$

Rewriting the free energy:

$$\mathcal{F}(\theta', \theta) = \mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(P(\mathcal{X}, z\vert \theta)\right) \right] - \mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(Q_{\theta'}(z)\right) \right]$$

we see that optimising with respect to $$\theta'$$ and $$\theta$$ $$\textit{simultaneously}$$ is complicated due to the coupling of $$Q_{\theta'}(z)$$ and $$P(\mathcal{X}, z\vert \theta)$$ through $$z$$. For example, attempting to optimise $$\theta'$$ will change the expectation over $$P(\mathcal{X}, z\vert \theta)$$, and thus changing the optimal parameters of $$\theta$$. To maximise the free energy, we use the $$\textbf{expectation maximisation (EM)}$$ algorithm, which $$\textit{iteratively}$$ optimises for $$\theta'$$ or $$\theta$$ at each step $$t$$, while the other remains fixed. 

The $$\textbf{expectation}$$ (E) step optimises $$\theta'$$ while holding $$\theta$$ fixed such that $$\theta'^{(t)} = \arg\max_{\theta' \in \Theta'} \mathcal{F}(\theta', \theta^{(t-1)})$$.

For our k-means model, this involves maximising the probability when holding $$\mu_{\ell}$$'s fixed by choosing $$z_{n, k}$$ to be the one hot encoding of $$\arg\min_{\ell \in \{1, \dots, K\}} \|\mathbf{x}_n - \mathbf{\mu}_{\ell}\|_2^2$$. We can see that:

$$\begin{align}
    \ell(\theta) \geq \mathcal{F}(\theta', \theta) &= \mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(\frac{P(\mathcal{X}, z\vert \theta)}{Q_{\theta'}(z)}\right) \right]\\
    &= \mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(\frac{P(z \vert \mathcal{X}, \theta) P(\mathcal{X}\vert \theta)}{Q_{\theta'}(z)}\right) \right] \\
    &= \int_{\mathcal{Z}} Q_{\theta'}(z) \log P(\mathcal{X}\vert \theta)dz +  \int_{\mathcal{Z}} Q_{\theta'}(z) \log \left(\frac{P(z \vert \mathcal{X}, \theta)}{Q_{\theta'}(z)}\right) dz \\
    &= \log P(\mathcal{X}\vert \theta) -  \mathbf{KL}\left[Q_{\theta'}(z) \| P(z \vert \mathcal{X}, \theta) \right]\\
    &= \ell(\theta) -  \mathbf{KL}\left[Q_{\theta'}(z) \| P(z \vert \mathcal{X}, \theta) \right]
\end{align}$$

The E step as minimising the Kullback-Leiberg divergence between $$Q_{\theta'}(z)$$ and $$P(z \vert \mathcal{X}, \theta)$$, $$\textit{raises}$$ the free energy lower bound on $$\ell(\theta)$$.

The $$\textbf{maximisation}$$ (M) step optimises $$\theta$$ while holding $$\theta'$$ fixed, $$\theta^{(t)} = \arg\max_{\theta \in \Theta} \mathcal{F}(\theta'^{(t)}, \theta)$$. For our k-means model, this involves recalculating the cluster means $$\mu_{\ell}$$ given the cluster assignments $$z_{n, k}$$ from $$\theta'^{(t)}$$. To understand the M step, we can see:

$$\begin{align}
\arg\max_{\theta \in \Theta} \mathcal{F}(\theta'^{(t)}, \theta) &= \arg\max_{\theta \in \Theta}\left( \mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(P(\mathcal{X}, z\vert \theta)\right) \right] - \mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(Q_{\theta'}(z)\right) \right]\right)\\
&= \arg\max_{\theta \in \Theta}\mathbb{E}_{z \sim Q_{\theta'}}\left[ \log \left(P(\mathcal{X}, z\vert \theta)\right) \right] 
\end{align}$$

Unlike in the E step, where we chose $$\theta'$$ to reach the upper bound on the free energy $$\ell(\theta)$$, in the M step, we are $$\textit{raising}$$ the upper bound $$\ell(\theta)$$ by $$\textit{maximising}$$ the loss under expectation of $$Q_{\theta'}$$. Combining, we can see that:

$$\ell(\theta^{(t-1}) \stackrel{(i)}{=} \mathcal{F}(\theta'^{(t)}, \theta^{(t-1)}) \stackrel{(ii)}{\leq} \mathcal{F}(\theta'^{(t)}, \theta^{(t)}) \stackrel{(iii)}{\leq} \ell(\theta^{(t)})$$

where $$(i)$$ is the E step, choosing $$\theta'^{(t)}$$ to match the current upper bound $$\ell(\theta^{(t-1})$$, $$(ii)$$ is the M step, choosing $$\theta^{(t)}$$ to raise the upper bound to $$\ell(\theta^{(t)})$$ by Jensen's inequality in $$(iii)$$. This guarantees that the EM algorithm  monotonically increases the loss. However, it should be noted that these inequalites are not strict, thus there is no guarantee that EM will find the global optimum.