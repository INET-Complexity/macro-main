# Final consumption substitution

Assume in this case that we put again goods into **bundles**. This means that we have $N$ goods labeled $i=1,\ldots, N$, and if goods $i$ and $j$ belong to bundle $b_k$  with $k=1,\ldots, B$ , then they can be substitutted. This can mean energy, for instance.



We assume a CES utility function (constant elasticity of substitution) for the households. Their utility within bundle $k$ reads
$$
U_k(\{x_i, i\in b_k\}) = \left(\sum_{i\in b_k}\alpha_i^{1/\sigma}x_i^{(\sigma-1)/\sigma} \right)
$$
where $x_i$ is the amount consumed of good $i$. 

A standard result is that this leads to the following shares,
$$
\gamma_i(t) = \frac{\alpha_i (p_i(t)(1+\tau_i(t)))^{-\sigma}}{\sum_{j\in b_k}\alpha_j (p_j(t)(1+\tau_j(t)))^{-\sigma}}
$$
within each bundle (here $\tau_j$ is the tax rate on good $j$).

Calling then $C_k$ the fraction of the budget allocated to bundle $k$, then the fraction spent in good $i$ is $\gamma_i(t) C_k=c_i(t)$.

For the simulation, we will assume that $C_k$ is fixed for each bundle (e.g. the household's budget fraction for energy will be fixed, and only relative shares will change). $c_i(0)$ is measured from real data, and we logically choose
$$
C_k = \sum_{i\in b_k}c_i(0),\quad \gamma_i(0) = \frac{c_i(0)}{\sum_{j\in b_k} c_j(0)}
$$


and then since $p_i(0)=p_j(0),\quad \forall(i,j)$ in the simulation, we can select 
$$
\alpha_i = c_i(0)(1+\tau_i(0))^\sigma
$$
and therefore we end up with the following consumption share:
$$
c_i(t)=c_i(0) \left(\frac{1+\tau_i(0)}{1+\tau_i(t)}\right)^\sigma p_i(t)^{-\sigma}\frac{\sum_{j\in b_k}c_j(0)}{\sum_{j\in b_k}c_j(0)p_j(t)^{-\sigma}\left(\frac{1+\tau_j(0)}{1+\tau_j(t)}\right)^{\sigma}} := c_i(0)C_k \gamma_i(t)
$$
and so everything can be computed dynamically by updating $\gamma_i(t)$. 



Note that $\sigma>1$ means there is a lot of substitution, $\sigma<1$ is the opposite.

## NOTES ON CURRENT IMPLEMENTATION

- The values of $c_i(0)$ are computed in the macro-main package. They are the consumption coefficients, saying how much of the disposable income is (initally) alocated to each good
- The way the bundles are built should be substantially similar to the substitution bundles that are used in firm substitution, but they should be a separate thing. Things should work in the same way as they do for the bundles (in macro-simulation, not macro-data):
  - The user specifies the bundles in the configuration
  - the bundles are then stored in the household object
  - bundles are then used in the funcitons that compute consumption
- These functions that compute consumption need to be updated to take into account the price. At this moment they don't. The price should include hte markup from taxes. This means the initial tax rate should also be somehow recorded by the households in this case. 
- If no bundles are passed then bundles should be singletons, as they are for the firms. In that case, the behaviour is the same as without a substitution.
- The consumption behaviour should be achieved by adding a function in the relevant function in the `func` folder of the households.