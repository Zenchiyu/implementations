## Monte Carlo methods

<details>
<summary>Markov Chain Monte Carlo: Metropolis, Metropolis-Hastings, Glauber</summary>

* For 1D:

![img](./images/mcmc.png)

* For 2D normal distribution (Metropolis):

![img](./images/mcmc_metropolis_2d.png)
</details>


<details>
<summary>Importance sampling</summary>

* Importance sampling to estimate an expectation by using samples coming
from another distribution (proposal distribution $g(x)$) instead of sampling
from $f=\rho$. Note that importance sampling is not a sampling method.
The importance weights are used to correct the bias introduced by sampling from
the wrong distribution.

![importance_sampling](./images/importance_sampling.png)
</details>

<details>
<summary>Sampling importance resampling</summary>

* Sampling importance resampling can be used to get samples approximately from
the distribution. The idea is to get samples from the proposal distribution,
obtain the importance weights of each of these samples $x_i$ as in
importance sampling and sampling from these samples with probabilities specified
by the normalized importance weights.

![sampling importance resampling](./images/sampling_importance_resampling.png)
</details>

<details>
<summary>Ising model</summary>

* Ising model: not working as expected
</details>
