# Preprocessing: smoothing and normalization

Before `binned_spikes` (raw spike counts, `n_units x n_bins`) is fed into UMAP, it goes through two steps: Gaussian smoothing along the time axis, then per-unit z-score normalization.

## 1. Gaussian smoothing

Each unit's binned spike count is convolved along time with a normalized Gaussian kernel, to reduce bin-to-bin sampling noise:

$$\hat{x}_i(t) = \sum_{\tau} x_i(t - \tau) \, g(\tau)$$

where the kernel is:

$$g(\tau) = \frac{1}{Z} \exp\left(-\frac{\tau^2}{2\sigma^2}\right), \qquad Z = \sum_{\tau} \exp\left(-\frac{\tau^2}{2\sigma^2}\right)$$

- $x_i(t)$ is the raw spike count for unit $i$ in bin $t$.
- $\sigma$ is the smoothing bandwidth in bins (`sigma_bins = 2`, i.e. 200 ms at a 100 ms bin size).
- $Z$ normalizes the kernel so it sums to 1 (a pure weighted average, not a rescaling).

## 2. Z-score normalization

Each smoothed unit is then centered and scaled by its own mean and standard deviation, so units with very different firing rates contribute comparably to distance calculations in UMAP:

$$\tilde{x}_i(t) = \frac{\hat{x}_i(t) - \mu_i}{\sigma_i}$$

- $\mu_i$ and $\sigma_i$ are the mean and standard deviation of unit $i$'s smoothed activity across the whole session.

This is the standard z-score approach. Note it has less protection than "soft normalization" (dividing by range + a constant, as in Churchland et al. 2012) against near-silent units — a unit with very low variance gets divided by a small $\sigma_i$, which can amplify its noise into large-magnitude values.
