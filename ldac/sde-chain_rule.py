import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True

np.random.seed(0)

alpha, beta = 2, 1
T, N = 1, 200
dt = T / N
X_zero = 1
X_zero2 = 1 / np.sqrt(X_zero)
Dt = dt

m = 3

W = np.empty((m, N+1))
W[:, 0] = 0
dW = np.random.normal(size=(m, N), loc=0, scale=np.sqrt(dt))
W[:, 1:] = np.cumsum(dW, axis=-1) + np.expand_dims(W[:, 0], axis=-1)

X_em1, X_em2 = np.zeros((m, N+1)), np.zeros((m, N+1))
X_em1[:, 0] = X_zero
X_em2[:, 0] = X_zero2
for j in range(N):
    W_inc = np.copy(dW[:, j])
    f1 = alpha - X_em1[:, j]
    g1 = beta * np.sqrt(np.abs(X_em1[:, j]))
    X_em1[:, j+1] = X_em1[:, j] + Dt * f1 + W_inc * g1
    
    f2 = (4 * alpha - beta ** 2) / (8 * X_em2[:, j]) - X_em2[:, j] / 2
    g2 = beta / 2
    X_em2[:, j+1] = X_em2[:, j] + Dt * f2 + W_inc * g2

Dt_vals = np.arange(0, T + Dt, Dt)

# fig = plt.figure(figsize=(10, 7.5))
fig = plt.figure(figsize=(8, 6))
ax1 = plt.subplot(3, 1, 1)
ax2 = plt.subplot(3, 1, 2)
ax3 = plt.subplot(3, 1, 3)

for i, ax in enumerate([ax1, ax2, ax3]):
    ax.plot(Dt_vals, np.sqrt(X_em1[i]), 'k--', label='Direct Solution')
    ax.plot(Dt_vals, X_em2[i], 'o', color='mediumpurple', 
             markerfacecolor='none',# markersize=3,
             label='Solution via Chain Rule')
    ax.grid()
    ax.set_xlabel('$t$')
    ax.set_ylabel('$V(X_t)$')
    ax.legend(title=f'error = {np.linalg.norm(np.sqrt(X_em1[i]) - X_em2[i], ord=np.inf):.5f}')

print(f'T = {T}, N = {N}, dt = {dt:.5f}')
plt.tight_layout()
plt.savefig('proj/sde-chain_rule.pdf')
plt.show()