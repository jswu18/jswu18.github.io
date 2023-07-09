---
title: 'Generalised Variational Inference'
date: 2023-07-09
permalink: /posts/2023/07/generalised-variational-inference/
tags:
  - Generalised Variational Inference
  - Bayesian Inference
---

## The Bayesian Posterior
Statistical modelling is traditionally focused on characterising an underlying data generation process. In a Bayesian context, this involves updating the beliefs on a model's parameterisation. Given a model parameterised by $\theta$, Bayesian inference can be viewed as an update rule on $\pi(\theta)$, the prior belief of $\theta$. For new observations $x_{1:N}$ and a likelihood function $p(x_{1:N}|\theta)$, the belief for $\theta$ is updated as:
\begin{align}
q_B^*(\theta) = \frac{p(x_{1:N}|\theta) \pi(\theta)}{\int_{\Theta} p(x_{1:N}|\theta) d \pi(\theta)}
\label{bayesian-posterior}
\end{align}
where $q_B^*(\theta)$ is known as the \textit{Bayesian posterior}. The validity of the Bayesian Posterior relies on three assumptions concerning the prior, the likelihood, and the normaliser. 

### The Prior Assumption
Bayesian inference assumes a prior $\pi(\theta)$ that is well-specified and informative of $\theta^*$, the `true' model parameters. The prior is interpreted as embodying \textit{all} previous knowledge about the data generating process such as previously observed data. Alternatively, the prior can also be interpreted as representing \textit{pseudo} observations about how we believe the data behaves.

### The Likelihood Assumption

Bayesian inference assumes that there exists $\theta^* \in \Theta$, such that $x_i \sim p(x_n | \theta^*)$ for some unknown but fixed $\theta^*$. In other words $p(x_n | \theta^*)$, the likelihood function that is chosen is \textit{exactly} the true data generating process and is parameterised by $\theta$. In this case, the problem is simply a matter of finding $\theta^*$. 

### The Normaliser Assumption

Bayesian inference assumes that the normaliser $\int_{\Theta} p(x_{1:N}|\theta) d \pi(\theta)$ is a tractable integral or is computationally tractable. Computational tractability assumes access to  adequate computational resources and time to reasonably approximate the integral. This means that in traditional Bayesian inference, the computational complexities of evaluating $q_B^*(\theta)$ can be ignored. 

## The Bayesian Posterior Breaks Down 
In contrast to traditional statistical modelling, larger-scaled models like Bayesian Neural Networks are typically focused on \textit{predictive performance} rather than \textit{model specification}. In these settings, the three  assumptions for the Bayesian posterior quickly breakdown, making it no longer reasonable to view $q_B^*(\theta)$ as a Bayesian belief update. 

### The Prior is Mis-specified
Larger-scaled models are often over-parameterised black box models, such as the weights of a Bayesian neural network. These parameters are essentially uninterpretable and priors are chosen out of convenience (i.e. Gaussians) with little thought given to their true parameterisation. In these settings, it is no longer practical to view $\pi(\theta)$ as a prior belief in the parameters of the data generating model as it is most definitely mis-specified.

### The Likelihood Model is Mis-specified
Although model mis-specification occurs in traditional Bayesian inference, techniques such as hypothesis testing, residual analysis, and domain expertise can help guide the construction of a reasonably well-specified setting. However, the intentions behind using larger-scaled models is completely different. It is not to \textit{understand} the data generating process, but rather to have superior \textit{predictive performance}. With over-parameterisation, these black box models are most definitely mis-specified but often provide high prediction accuracy. As such, model parameters are typically chosen through an optimisation process (i.e. gradient descent), no longer adhering to the spirit of traditional Bayesian inference. For larger-scaled models, it is almost never fair to assume that $x_n \sim p(x|\theta)$ for \textit{any} $\theta \in \Theta$.

### The Normaliser is Intractability
The use of conjugate priors is the only case when there exists closed form expressions for $\int_{\Theta} p(x_{1:N}|\theta) d \pi(\theta)$ to ensure tractable evaluation of $q_B^*(\theta)$. For over-parameterised black-box models, $q_B^*(\theta)$ will need to be approximated either through sampling approximations of the normaliser or variational approximations of $q_B^*(\theta)$.
\newline
\\Samplers such as Metropolis Hastings or Markov Chain Monte Carlo only have convergence guarantees in the infinite limit. Acheiving this limit would require access to infinite computational resources and time, clearly impractical. 
\newline
\\Approximating $q_B^*(\theta)$ involves solving for $q_A^*(\theta) \in \mathcal{Q}_{A}$, where $\mathcal{Q}_{A}$ is often viewed as distributions of a simpler form. For example mean field approximations define a family of distributions $\mathcal{Q}_{MF} = \left\{\prod_i q_i(\theta_i)\right\}$, a product of independent distributions. Variational inference is motivated to finding a $q_A^*(\theta) \in \mathcal{Q}_{A}$ that \textit{approximates} $q_B^*(\theta)$, through the minimisation of some divergence between the two, $D(q_A^*(\theta)\| q_B^*(\theta))$. However the space of distributions $\mathcal{Q}_{A}$ is usually severely restrictive in its expressiveness and $q_A^*(\theta)$ is almost never a fair depiction of the structure of $q_B^*(\theta)$. Realistically, $\mathcal{Q}_{A}$ is chosen purely for computational convenience. With larger-scaled models, it is often no longer reasonable to assume that the normaliser of the Bayesian posterior will be tractable or that $q_B^*(\theta)$ can be reasonably approximated in a tractable manner.

## The Generalised Posterior 
Interpreting the mechanism behind calculating the Bayesian posterior in the context of optimisation can provide a more reasonable depiction of $q_B^*(\theta)$ for larger-scaled models. It can be shown that $q_B^*(\theta)$ solves a special case of a general variational inference (GVI) problem:
\begin{align}
q^*(\theta) = \argmin_{q \in \Pi} \left\{ \mathbb{E}_{q(\theta)}\left[\sum_{n=1}^N \ell(\theta, x_n)\right] + D(q\|\pi)\right\}
\label{general-posterior}
\end{align}
where $q_B^*(\theta)$ is recovered by choosing the negative log-likelihood loss $\ell(\theta, \cdot) = -\log p(\cdot | \theta)$, the Kullback-Leibler divergence $D(\cdot \| \pi) = \KLD(\cdot \| \pi)$, and the feasible set $\Pi = \mathcal{P}(\Theta)$. No longer deriving $q_B^*(\theta)$ from a belief update, we are no longer burdened to fulfill the assumptions required for the Bayesian inference interpretation of $q_B^*(\theta)$. We can re-interpret the role of the prior, likelihood, and choosing tractable normaliser approximations, in the context of an optimisation problem.

### The Prior is a Regulariser
In the optimisation context of (\ref{general-posterior}), we can see that the prior $\pi$ only exists in the divergence term. As such, $\pi$ defines the regulariser of an empirical risk minimisation optimisation problem which is solved by the Bayesian posterior $q_B^*(\theta)$. The choice of prior controls model complexity and prevents overfitting to the empirical risk. Unlike in the Bayesian interpretation, in this optimisation setup $\pi$ is no longer required to be a well-specified prior. Thus in larger-scaled models where prior mis-specification is almost guaranteed, it is more appropriate to view the prior as a regulariser on model complexity rather than a prior belief in the model parameters.

### The Likelihood is a Loss
From (\ref{general-posterior}), the likelihood term exists only in the expectation. Note that the empirical risk is defined as:
\begin{align}
\mathcal{E}(\theta) = \mathbb{E}_{q(\theta)}\left[\sum_{n=1}^N \ell\left(x_n, \theta\right)\right]
\label{empirical-risk}
\end{align}
where $\ell$ is some loss function. We can see that defining $\ell$ as the negative log-likelihood, we recover an empirical risk of the model over empirical data that is equivalent to the expectation term of (\ref{general-posterior}). This interprets the likelihood function as a special loss definition for an optimisation problem. In other words, $q_B^*(\theta)$ is the minimiser of a regularised empirical risk with a log-likelihood loss, defined with respect to its predictive performance rather than its belief updates on model parameters. By pivoting from the Bayesian interpretation of $q_B^*(\theta)$, we no longer need to have a well-specified likelihood function because we can view the posterior as empirical risk minimisation for a special loss definition.

### Model Approximations as Optimisation Constraints
Rather than viewing $q_A^*(\theta)$ as an approximation of $q_B^*(\theta)$, it is more practical to view $q_A^*(\theta)$ as the solution to an optimisation problem, where we are \textit{constrained} to $\mathcal{Q}_{A}$. In other words, we are not attempting to \textit{approximate} $q_B^*(\theta)$ but rather we are finding the \textit{optimal} solution $q_A^*(\theta)$ in the space $\mathcal{Q}_{A}$. With mis-specified priors and likelihood functions, $q_B^*(\theta)$  is no longer a true Bayesian posterior anyways, and so there's little meaning behind these approximations. Especially with the reframing in optimisation, we are more concerned with finding the model in our feasible set $\mathcal{Q}_{A}$ with the best predictive performance rather than the model that most accurately depicts the data generation process.
\subsection{The Bayesian Posterior in a Wider Context}
By $\textit{generalising}$ the Bayesian posterior update mechanism to an optimisation problem, we can understand more general posteriors of the form $q^*(\theta)$. Although it is defined as a solution to an optimisation problem, $q^*(\theta)$ can still be viewed as a form of posterior. The optimisation in ($\ref{general-posterior}$) still provides a mechanism to generate an updated belief of $\theta$ given new data $x_{1:N}$. Generalised Variational Inference provides a flexible framework for constructing these \textit{pseudo}-posteriors where the \textit{Bayesian} posterior $q_B^*(\theta)$ can be recovered as a special case in a wider context.

## Theoretical Guarantees from GVI 
Loss minimisation of larger-scaled machine learning models is typically over a highly non-convex optimisation problem. The parameters of these models $f_{\theta}$ are typically trained through the minimisation:
\begin{align}
\min_{\theta \in \Theta} \sum_{n=1}^N\ell_n(x_n, \theta)
\label{loss-minimisation}
\end{align}
where $\ell_n(x_n, \theta)$ quantifies the predictive performance of a model's parameterisation $\theta$ for training observation $(x_n, y_n)$, such as the squared loss $\ell_{sq}(\theta) = \sum_{n=1}^N \left(y_n - f_{\theta}(x_n)\right)^2$. Typically $\theta^*$, the minimiser of (\ref{loss-minimisation}), is in $ \mathbb{R}^J$ a finite dimensional space where $J$ is the number of parameters in $f_{\theta}$.\\
\newline
In practice a reasonable local minima can achieve high predictive performance, and so the non-convex nature of the parameter space is often ignored. However without the guaranteed existence of a unique minimiser, learning theory is unable to make theoretical claims about these larger-scaled models. By convexifying (\ref{loss-minimisation}), we recover the minimisation problem of the form ($\ref{general-posterior}$). Thus the GVI posterior is also a reframing of modern machine learning models sp that we can understand them in the context of learning theory.
### Probabilistic Lifting
To convexify ($\ref{loss-minimisation}$), we begin by lifting the problem from a finite-dimensional parameter space $\mathbb{R}^J$ to an infinite-dimensional probability space $\mathcal{P}(\mathbb{R}^J)$, the space of measures on $\mathbb{R}^J$:
\begin{align}
    \min_{Q \in \mathcal{P}(\mathbb{R}^J)} \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) dq(\theta)
\label{risk-minimisation}
\end{align}
where $\hat{q}$, minimisers of (\ref{risk-minimisation}), can correspond to  $\hat{\theta}$, minimisers of (\ref{loss-minimisation}), through the Dirac measure $\hat{q}(\theta) = \delta_{\hat{\theta}} (\theta)$. This first reformulation changes a non-convex problem with respect to $\theta$ to a linear problem with respect to $q$. 
\\To show this, consider two minimisers $\theta_A$ and $\theta_B$ such that:
\begin{align}
    \sum_{n=1}^N\ell_n(x_n, \theta_A) = \sum_{n=1}^N\ell_n(x_n, \theta_B) = \min_{\theta \in \Theta} \sum_{n=1}^N\ell_n(x_n, \theta), \text{ where } \theta_A \neq \theta_B
\end{align}
with corresponding measures $\delta_{\theta_A}, \delta_{\theta_B} \in \mathcal{P}(\mathbb{R}^J)$ such that:
\begin{align}
    \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) d\delta_{\theta_A} = \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) d\delta_{\theta_B} = \min_{q \in \mathcal{P}(\mathbb{R}^J)} \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) dq(\theta)
    \label{ex-risk-minimisers}
\end{align}
By defining $q_t = (1-t)\delta_{\theta_A} + t\delta_{\theta_B}$ for $t \in [0, 1]$:
\begin{align}
    \label{show-linear-defn}
    \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) dq_t(\theta) &= \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) d\left\big((1-t)\delta_{\theta_A} + t\delta_{\theta_B}\right\big)\\
    \label{show-linear-linear-operator}
    &= (1-t)\int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) d\delta_{\theta_A} + t \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) d\delta_{\theta_B}\\
    \label{show-linear-minimisers}
    &= \min_{Q \in \mathcal{P}(\mathbb{R}^J)} \int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) dq(\theta)
\end{align}
where ($\ref{show-linear-linear-operator}$) follows by linearity and ($\ref{show-linear-minimisers}$) follows from ($\ref{ex-risk-minimisers}$). Thus ($\ref{risk-minimisation}$) is a linear problem in $q$. 
### Convexification through Regularisation
By adding a strictly convex and positive regulariser $D_r(q\| \pi)$ to our linear objective ($\ref{risk-minimisation}$), we ensure a strictly convex objective, guaranteeing the existence of a $\textit{unique}$ minimiser:
\begin{align}
    q^* = \argmin_{q \in \mathcal{P}(\mathbb{R}^J)} \left\{\int \left( \sum_{n=1}^N\ell_n(x_n, \theta)\right) dq(\theta) + \lambda D_r(q \| \pi)\right\}
\label{regularised-risk-minimisation}
\end{align}
where $\lambda > 0$. The solution of ($\ref{regularised-risk-minimisation}$) is no longer a minimiser of ($\ref{risk-minimisation}$). But rather, $\lambda$ balances the tradeoff between the empirical risk minimisation of ($\ref{risk-minimisation}$) and deviance from a prior measure $\pi$, which in this context we can view as a reference measure.\\
\newline 
Choosing $\Pi =\mathcal{P}(\mathbb{R}^J)$, $\ell(\theta) = \sum_{n=1}^N\ell_n(x_n, \theta)$, and $D(q\| \pi) = \lambda D_r(q\| \pi)$, we see that ($\ref{regularised-risk-minimisation}$) fits into the general form of ($\ref{general-posterior}$), recovering the GVI posterior. 
### Uniqueness of the GVI posterior
Through probabilistic lifting and convexification, we can formulate a GVI posterior that guarantees a unique minimiser for the non-convex problem in (\ref{loss-minimisation}). This posterior is a unique weighted averaging of the local and global minima of (\ref{loss-minimisation}), and equivalently (\ref{risk-minimisation}) where each minima is weighted by the discrepancy from the prior reference measure $\lambda D_r(q \| \pi)$. By guaranteeing a unique minimiser, the GVI framework can provide theoretical guarantees for learning larger-scaled machine learning models.

## References
<a id="1">[1]</a>
Anastasiou, A., Barp, A., Briol, F. X., Ebner, B., Gaunt, R. E., Ghaderinezhad, F., ... & Swan, Y. (2021). Stein's Method Meets Statistics: A Review of Some Recent Developments. arXiv preprint arXiv:2105.03481.

<a id="2">[2]</a>
Barp, A., Briol, F. X., Duncan, A., Girolami, M., & Mackey, L. (2019). Minimum stein discrepancy estimators. Advances in Neural Information Processing Systems, 32.