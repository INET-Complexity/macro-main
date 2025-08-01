# Notes on substitution

The base Leontief production function we use reads:
$$
y_i = \min_{j}\left(J_{ij}x_{ij}\right)
$$
where $J_{ij}$ is the intermediate inputs productivity matrix and $x_{ij}$ the intermediate inputs stock.

### Target intermediate inputs

Given a target production $\hat{y}_i$, the target intermediate inputs required for the production of that are given by 
$$
\widehat{x}_{ij} = \frac{\widehat{y}_i}{J_{ij}},
$$
and the firm then decides how much to effectively buy depending on how much it already has in inventories (it aims to keep stable inventories, allowing it to keep producing for some time if there is no delivery of inputs). 



## Good bundles

To introduce substitution, I have introduced good bundles. They will be indexed by the greek letter $\alpha$, and $b_{\alpha}$ denotes bundle $\alpha$ while $|b_{\alpha}|$ denotes the number of goods in that bundle. In this case, the production function now reads
$$
y_i = \min_{\alpha}\left(\frac{1}{\vert b_\alpha \vert} \sum_{j\in b_{\alpha}} J_{ij}x_{ij}\right)
$$
 the normalisation $1/|b_{\alpha}|$ ensures that the same amount is produced initially with both Leontief and bundled Leontief technologies. This is because initial values of $x_{ij}$ are such that $J_{ik}x_{ik} = J_{i\ell }x_{i\ell}$ for all $k,\ell$. 

### Target intermediate inputs

We start from an initial target that is identical to the Leontief target, 
$$
\widehat{x}_{ij}^{\text{L}} = \frac{\widehat{y}_i}{J_{ij}}.
$$
But we multiply them by weights $w_{ij}$ that will make the firm tend to buy cheaper goods. This means the effective amount of goods it wants to buy are 
$$
\widehat{x}_{ij}=w_{ij}\widehat{x}_{ij}^{\text{L}}= \frac{w_{ij}}{J_{ij}}\hat{y_i}.
$$
The resulting production would be
$$
y_i = \min_{\alpha}\left(\frac{1}{|b_{\alpha}|}\sum_{j\in b_{\alpha}} \hat{y}_i w_{ij}\right)
$$
which suggests choosing the weights as to normalise them too
$$
\sum_{j\in b_{\alpha}}w_{ij} = \vert b_{\alpha}|.
$$
Finally, we choose, for $j\in b_{\alpha}$ 
$$
w_{ij}^*(t) = \vert b_{\alpha}\vert \cdot \frac{\exp\left(-\beta \frac{p_j}{\overline{p}}\right)}{\sum_{\ell\in b_{\alpha}}\exp\left(-\beta \frac{p_\ell}{\overline{p}}\right)}
$$

$$
w_{ij}(t) =\alpha w_{ij}^*(t) + (1-\alpha) w_{ij}(t-1)
$$









