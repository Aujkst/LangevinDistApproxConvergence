# LangevinDistApproxConvergence

## (Unadjusted) Langevin Algorithm

$$
\mathrm{d}X_t = \frac{1}{2}\nabla\log\pi(X_t)\mathrm{d}t + \mathrm{d}W_t,\quad X_0=x_0
$$

### Euler-Maruyama Method

$$
X_{t_n+1} = X_{t_n} + \frac{1}{2}\frac{\mathrm{d}}{\mathrm{d}x}\log\pi(X_{t_n})\Delta + \Delta W_n,
$$

where $\Delta W_n \sim \mathcal{N}(0, \Delta)$ i.i.d..

### Strong Order 1.5 Taylor Method

$$
\begin{aligned}
    X_{t_n+1} = X_{t_n} & + \frac{1}{2}\frac{\mathrm{d}}{\mathrm{d}x}\log\pi(X_{t_n})\Delta + \Delta W_n + \frac{1}{2}\frac{\mathrm{d^2}}{\mathrm{d}x^2}\log\pi(X_{t_n}) \Delta Z_n \\
    & + \frac{1}{8}\left(\frac{\mathrm{d}}{\mathrm{d}x}\log\pi(X_{t_n})\frac{\mathrm{d^2}}{\mathrm{d}x^2}\log\pi(X_{t_n}) + \frac{\mathrm{d^3}}{\mathrm{d}x^3}\log\pi(X_{t_n}) \right) \Delta^2
,\end{aligned}
$$

where $\Delta W_n = \sqrt{\Delta}U_{1, n}$, $\Delta Z_n = \frac{1}{2}\Delta^{\frac{3}{2}}\left( U_{1, n} + \frac{1}{\sqrt{3}}U_{2, n} \right)$ and $U_{1, n}, U_{2, n} \sim \mathcal{N}(0, 1)$ i.i.d..

## Metropolis Adjusted Langevin Algorithm (MALA)

The proposals $x'$ are accepted with probability $\alpha(x'|x)$, where:

$$
\alpha(x'|x) \coloneqq 1 \land \frac{\pi(x')q(x|x')}{\pi(x)q(x'|x)}
$$