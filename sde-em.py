import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

np.random.seed(0)

T, N = 1, 2**8
dt = T / N

# Number of realizations to generate.
m = 1000

# Create an empty array to store the realizations.
W = np.empty((m, N+1))

# Initial values of x.
W[:, 0] = 0

dW = np.random.normal(size=(m, N), loc=0, scale=np.sqrt(dt))
W[:, 1:] = np.cumsum(dW, axis=-1) + np.expand_dims(W[:, 0], axis=-1)

t = np.linspace(0, N * dt, N + 1)

lmbda, mu = 2, 1
X_zero = 1

R = 4
Dt = R * dt
L = int(N / R)
X_true = X_zero * np.exp((lmbda-0.5*mu**2) * t + mu * W)

X_em = X_zero * np.ones((W.shape[0], L+1))
X_temp = np.repeat(X_zero, repeats=m)
for j in range(L):
    W_inc = np.sum(dW[:, R*j+1:R*(j+1)+1], axis=1)
    X_temp = X_temp + Dt * lmbda * X_temp + mu * W_inc * X_temp
    X_em[:, j+1] = X_temp

fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

for i, ax in enumerate([ax1, ax2, ax3]):
    emerr = np.abs(X_em[i, -1] - X_true[i, -1])
    ax.plot(t[::R], X_em[i], 
            marker='*', color='mediumpurple', linestyle='--', 
            label=f'EM approx.')
    ax.plot(t, X_true[i], 'k', label=f'True solution')
    ax.grid()
    print(f'True solution and Euler-Maruyama approximation (N = {N}, L = {L})')
    # ax.set_title(f'True solution and Euler-Maruyama approximation (N = {N}, L = {L})')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$X_t$')
    ax.legend(title=f'emerr = {emerr:.3f}')

plt.tight_layout()
plt.savefig('proj/sde-em.pdf')
plt.show()
