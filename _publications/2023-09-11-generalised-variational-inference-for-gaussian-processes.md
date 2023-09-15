---
title: "Generalised Variational Inference for Gaussian Processes (Msc. Thesis)"
collection: publications
permalink: /publication/2023-09-11-generalised-varational-inference-for-gaussian-processes
excerpt: 'Achieved linear time inference objectives for variational Gaussian processes compared to cubic complexity approaches in the literature.'
date: 2023-09-11
paperurl: '/files/james-wu-msc-thesis.pdf'
citation: 'Your Name, You. (2009). &quot;Paper Title Number 1.&quot; <i>Journal 1</i>. 1(1).'
---
Proposed by <a href="https://arxiv.org/pdf/1904.02063.pdf">Knoblauch et al. (2022)</a>, generalised variational inference (GVI) is a learning framework motivated by an optimisation-centric interpretation of Bayesian inference. Extending GVI to infinite dimensions, <a href="https://arxiv.org/pdf/2205.06342.pdf">Wild et al. (2022)</a> introduces Gaussian Wasserstein inference (GWI) in function spaces. GWI demonstrates a new inference approach for variational Gaussian processes (GPs), circumventing many limitations of previous approaches. Our work introduces various improvements to GWI for GPs, including new kernel parameterisations such as the neural network GP (NNGP) kernels from <a href="https://arxiv.org/pdf/1912.02803.pdf">Novak et al. (2019)</a>. We also introduce a new learning framework that we call projected GVI (pGVI) for GPs. pGVI weakens the GVI assumption of a definite regulariser. Instead, we propose regularising between scalar projections of the stochastic processes, an approach we call projected regularisation. We demonstrate that pGVI is a highly flexible and well-performing variational inference framework with significantly cheaper linearly time computational costs compared to the cubic costs of existing approaches. We also present our learning frameworks through a comprehensive software implementation available on <a href="https://github.com/jswu18/gvi-gaussian-process">GitHub</a>.

[Download paper here](/files/james-wu-msc-thesis.pdf)
